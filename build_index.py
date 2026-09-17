import argparse
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


DEFAULT_FRONTEND_URL = (
    "https://github.com/umershahbaz04/"
    "shipra-frontend/archive/refs/heads/main.zip"
)

SUPPORTED_EXTENSIONS = {
    ".cs", ".cshtml", ".csproj", ".sln", ".props", ".targets",
    ".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".html",
    ".css", ".scss", ".sql", ".proto", ".yml", ".yaml",
}

IGNORED_DIRECTORIES = {
    ".git", ".github", ".idea", ".vs", ".vscode", "bin", "obj",
    "build", "dist", "coverage", "node_modules", "packages", "logs",
    "uploads", "testresults", "wwwroot", "migrations",
}

IGNORED_EXACT_FILES = {
    ".env", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "secrets.cs", "launchsettings.json",
}

SENSITIVE_SUFFIXES = {".pfx", ".p12", ".pem", ".key", ".cer", ".crt"}
MAX_FILE_SIZE = 600_000
LINES_PER_CHUNK = 120
LINE_OVERLAP = 25


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Shipra frontend + backend FAISS knowledge files."
    )
    parser.add_argument(
        "--frontend-url",
        default=DEFAULT_FRONTEND_URL,
        help="GitHub ZIP URL for the frontend main branch.",
    )
    parser.add_argument(
        "--frontend-zip",
        help="Optional local frontend ZIP. Overrides --frontend-url.",
    )
    parser.add_argument(
        "--backend-zip",
        action="append",
        default=[],
        help="Backend source ZIP. Repeat for multiple archives.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Validate collection without loading the embedding model.",
    )
    return parser.parse_args()


def safe_extract(archive, destination):
    destination = destination.resolve()

    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError(f"Unsafe ZIP member: {member.filename}")

    archive.extractall(destination)


def download_bytes(url):
    print(f"Downloading frontend source: {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Shipra-AI-Indexer/2.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def discover_backend_archives(project_root, supplied):
    if supplied:
        paths = [Path(value).expanduser().resolve() for value in supplied]
    else:
        paths = []
        search_roots = [project_root, Path("/content")]
        for root in search_roots:
            if root.exists():
                paths.extend(root.glob("Shipra.Backend.API.*.zip"))

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(resolved)
            seen.add(resolved)

    missing = [str(path) for path in unique_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Backend ZIP file(s) not found: " + ", ".join(missing)
        )

    if not unique_paths:
        raise FileNotFoundError(
            "No backend ZIPs found. Upload them to /content or pass "
            "--backend-zip for each archive."
        )

    return unique_paths


def is_sensitive_or_generated(relative_path):
    lowered_parts = {part.lower() for part in relative_path.parts}
    if lowered_parts.intersection(IGNORED_DIRECTORIES):
        return True

    filename = relative_path.name.lower()
    if filename in IGNORED_EXACT_FILES:
        return True
    if filename.startswith("appsettings") and filename.endswith(".json"):
        return True
    if filename.startswith(".env"):
        return True
    if relative_path.suffix.lower() in SENSITIVE_SUFFIXES:
        return True

    return False


def redact_high_confidence_secrets(text):
    assignment_pattern = re.compile(
        r"(?i)(api[_-]?key|client[_-]?secret|access[_-]?token|password)"
        r"(\s*[=:]\s*)"
        r"([\"'])([^\"'\r\n]{6,})([\"'])"
    )
    return assignment_pattern.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}"
            f"{match.group(3)}[REDACTED]{match.group(5)}"
        ),
        text,
    )


def detect_symbol(lines):
    patterns = [
        re.compile(
            r"\b(?:class|interface|record|struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
        ),
        re.compile(
            r"\b(?:async\s+)?(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)"
            r"\s*=\s*(?:async\s*)?\("
        ),
        re.compile(
            r"\b(?:public|private|protected|internal)\s+"
            r"(?:static\s+)?(?:async\s+)?[A-Za-z0-9_<>,?\[\].]+\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
        ),
    ]

    for line in lines:
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                return match.group(1)

    return None


def split_source(text):
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + LINES_PER_CHUNK, len(lines))
        selected_lines = lines[start:end]
        content = "\n".join(selected_lines).strip()

        if content:
            chunks.append(
                {
                    "text": content,
                    "start_line": start + 1,
                    "end_line": end,
                    "symbol": detect_symbol(selected_lines),
                }
            )

        if end == len(lines):
            break
        start = end - LINE_OVERLAP

    return chunks


def infer_layer(relative_path):
    lowered = [part.lower() for part in relative_path.parts]
    known_layers = {
        "application", "core", "infrastructure", "web", "api", "components",
        "pages", "services", "repository", "repositories", "features",
    }
    for part in lowered:
        if part in known_layers:
            return part
    return lowered[0] if lowered else "unknown"


def collect_tree(root, project, path_prefix, seen_hashes):
    collected_chunks = []
    collected_metadata = []
    skipped_sensitive = 0
    skipped_large = 0

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(root)
        if is_sensitive_or_generated(relative_path):
            skipped_sensitive += 1
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.stat().st_size > MAX_FILE_SIZE:
            skipped_large += 1
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        text = redact_high_confidence_secrets(text)
        display_path = f"{path_prefix}/{relative_path.as_posix()}"

        for chunk_number, chunk in enumerate(split_source(text)):
            digest = hashlib.sha256(
                f"{display_path}\n{chunk['text']}".encode("utf-8")
            ).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            collected_chunks.append(chunk["text"])
            collected_metadata.append(
                {
                    "project": project,
                    "source_type": "actual_code",
                    "file_path": display_path,
                    "section_title": chunk["symbol"] or display_path,
                    "symbol": chunk["symbol"],
                    "layer": infer_layer(relative_path),
                    "chunk_index": chunk_number,
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                    "content_sha256": digest,
                }
            )

    print(
        f"Collected {len(collected_chunks)} chunks from {project}:{path_prefix}; "
        f"skipped {skipped_sensitive} sensitive/generated and "
        f"{skipped_large} oversized files."
    )
    return collected_chunks, collected_metadata


def analyze_frontend_reachability(chunks, metadata):
    """Mark frontend files reachable from real application entry points."""
    source_by_path = {}

    for chunk, item in zip(chunks, metadata):
        if item.get("project") != "frontend":
            continue
        path = item.get("file_path", "")
        source_by_path.setdefault(path, []).append(chunk)

    all_paths = set(source_by_path)
    import_graph = {path: set() for path in all_paths}
    inbound_counts = Counter()
    import_pattern = re.compile(
        r"(?:from\s+|import\s*\(|require\s*\()"
        r"[\"']([^\"']+)[\"']"
    )

    def resolve_import(importer, specifier):
        if not specifier.startswith("."):
            return None

        base = posixpath.normpath(
            posixpath.join(posixpath.dirname(importer), specifier)
        )
        candidates = [base]
        candidates.extend(
            f"{base}{extension}"
            for extension in (".js", ".jsx", ".ts", ".tsx")
        )
        candidates.extend(
            f"{base}/index{extension}"
            for extension in (".js", ".jsx", ".ts", ".tsx")
        )

        return next(
            (candidate for candidate in candidates if candidate in all_paths),
            None,
        )

    for importer, source_parts in source_by_path.items():
        source = "\n".join(source_parts)
        for specifier in import_pattern.findall(source):
            target = resolve_import(importer, specifier)
            if target and target != importer:
                import_graph[importer].add(target)

    for targets in import_graph.values():
        for target in targets:
            inbound_counts[target] += 1

    entry_suffixes = (
        "/src/index.js",
        "/src/index.jsx",
        "/src/main.js",
        "/src/main.jsx",
        "/src/main.ts",
        "/src/main.tsx",
    )
    entry_points = {
        path for path in all_paths if path.endswith(entry_suffixes)
    }

    reachable = set(entry_points)
    pending = list(entry_points)
    while pending:
        importer = pending.pop()
        for target in import_graph.get(importer, ()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)

    backup_pattern = re.compile(
        r"(?i)(?:^|[/_.-])(backup|bak|copy|old|legacy|deprecated|unused)"
    )

    for item in metadata:
        if item.get("project") != "frontend":
            continue

        path = item.get("file_path", "")
        is_backup_named = bool(backup_pattern.search(path))
        is_reachable = path in reachable
        item["frontend_reachable"] = is_reachable
        item["frontend_inbound_references"] = inbound_counts.get(path, 0)
        item["implementation_status"] = (
            "backup_named"
            if is_backup_named
            else "active_reachable"
            if is_reachable
            else "unreferenced_or_dynamic"
        )

    status_counts = Counter(
        item.get("implementation_status")
        for item in metadata
        if item.get("project") == "frontend"
    )
    print(f"Frontend reachability: {dict(status_counts)}")


def retain_original_backend_docs(project_root):
    chunks_path = project_root / "chunks.json"
    metadata_path = project_root / "metadata.json"

    if not chunks_path.exists() or not metadata_path.exists():
        return [], []

    with chunks_path.open("r", encoding="utf-8") as file:
        old_chunks = json.load(file)
    with metadata_path.open("r", encoding="utf-8") as file:
        old_metadata = json.load(file)

    retained_chunks = []
    retained_metadata = []

    for chunk, item in zip(old_chunks, old_metadata):
        file_path = item.get("file_path", "")
        source_type = item.get("source_type", "")
        is_original_doc = (
            file_path == "Shipra.Backend.API documentation"
            or (
                item.get("project", "backend") == "backend"
                and source_type != "actual_code"
                and not file_path.lower().endswith(
                    (".cs", ".cshtml", ".csproj", ".sln")
                )
            )
        )

        if is_original_doc:
            new_item = dict(item)
            new_item["project"] = "backend"
            new_item["source_type"] = "documentation"
            new_item["file_path"] = "Shipra.Backend.API documentation"
            retained_chunks.append(chunk)
            retained_metadata.append(new_item)

    print(f"Retained {len(retained_chunks)} original backend documentation chunks.")
    return retained_chunks, retained_metadata


def write_json_atomic(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def build_embeddings(project_root, chunks, metadata):
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install requirements first: pip install -r requirements.txt"
        ) from error

    embedding_inputs = []
    for chunk, item in zip(chunks, metadata):
        embedding_inputs.append(
            "\n".join(
                [
                    f"Project: {item.get('project', '')}",
                    f"Source type: {item.get('source_type', '')}",
                    f"File: {item.get('file_path', '')}",
                    f"Layer: {item.get('layer', '')}",
                    f"Symbol: {item.get('symbol', '')}",
                    f"Implementation status: {item.get('implementation_status', '')}",
                    f"Frontend reachable: {item.get('frontend_reachable', '')}",
                    chunk,
                ]
            )
        )

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        embedding_inputs,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    write_json_atomic(project_root / "chunks.json", chunks)
    write_json_atomic(project_root / "metadata.json", metadata)

    temporary_index = project_root / "faiss.index.tmp"
    faiss.write_index(index, str(temporary_index))
    os.replace(temporary_index, project_root / "faiss.index")

    print(
        f"Knowledge base created: {len(chunks)} chunks, "
        f"FAISS vectors: {index.ntotal}."
    )


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    temporary_root = Path(tempfile.mkdtemp(prefix="shipra-index-"))

    try:
        all_chunks, all_metadata = retain_original_backend_docs(project_root)
        seen_hashes = set()

        for chunk, item in zip(all_chunks, all_metadata):
            digest = hashlib.sha256(
                f"{item.get('file_path', '')}\n{chunk}".encode("utf-8")
            ).hexdigest()
            seen_hashes.add(digest)

        frontend_destination = temporary_root / "frontend"
        frontend_destination.mkdir(parents=True)

        if args.frontend_zip:
            frontend_bytes = Path(args.frontend_zip).read_bytes()
        else:
            frontend_bytes = download_bytes(args.frontend_url)

        with zipfile.ZipFile(io.BytesIO(frontend_bytes)) as archive:
            safe_extract(archive, frontend_destination)

        if (frontend_destination / "package.json").is_file():
            frontend_root = frontend_destination
        else:
            frontend_roots = [
                path for path in frontend_destination.iterdir() if path.is_dir()
            ]
            if not frontend_roots:
                raise ValueError("Frontend ZIP contains no project directory.")
            frontend_root = frontend_roots[0]

        new_chunks, new_metadata = collect_tree(
            frontend_root,
            project="frontend",
            path_prefix="Shipra.Frontend",
            seen_hashes=seen_hashes,
        )
        analyze_frontend_reachability(new_chunks, new_metadata)
        all_chunks.extend(new_chunks)
        all_metadata.extend(new_metadata)

        backend_archives = discover_backend_archives(
            project_root,
            args.backend_zip,
        )

        for archive_number, archive_path in enumerate(backend_archives, start=1):
            destination = temporary_root / f"backend-{archive_number}"
            destination.mkdir(parents=True)

            with zipfile.ZipFile(archive_path) as archive:
                safe_extract(archive, destination)

            roots = [path for path in destination.iterdir() if path.is_dir()]
            source_root = roots[0] if len(roots) == 1 else destination

            new_chunks, new_metadata = collect_tree(
                source_root,
                project="backend",
                path_prefix=source_root.name,
                seen_hashes=seen_hashes,
            )
            all_chunks.extend(new_chunks)
            all_metadata.extend(new_metadata)

        counts = Counter(item["project"] for item in all_metadata)
        actual_code_count = sum(
            item.get("source_type") == "actual_code" for item in all_metadata
        )
        print(
            f"Final collection: {len(all_chunks)} chunks; "
            f"projects={dict(counts)}; actual_code={actual_code_count}."
        )

        required_paths = (
            "SaleChannelController.cs",
            "CreateSaleChannelConfigCommandHandler.cs",
            "SaleChannelConfigRepository.cs",
            "SaleChannelConfig.cs",
            "saleChannelConnectModal.js",
            "AxiosInterceptors.js",
        )
        indexed_paths = {item["file_path"] for item in all_metadata}
        missing_requirements = [
            required
            for required in required_paths
            if not any(path.endswith(required) for path in indexed_paths)
        ]
        if missing_requirements:
            raise ValueError(
                "Critical Sales Channel files were not indexed: "
                + ", ".join(missing_requirements)
            )

        print("Critical frontend/backend Sales Channel chain verified.")

        if args.collect_only:
            print("Collection-only validation completed; no files were overwritten.")
            return

        build_embeddings(project_root, all_chunks, all_metadata)

    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
