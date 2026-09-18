from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from groq import Groq

from live_api_registry import READ_ONLY_TOOLS


@dataclass(frozen=True)
class LiveAPIIntent:
    is_live_query: bool
    tool: str | None = None
    parameters: dict[str, Any] | None = None
    language: str = "English"
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification: str = ""


def _json_object(text: str) -> dict[str, Any]:
    start = (text or "").find("{")
    end = (text or "").rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON intent was returned.")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Live intent must be a JSON object.")
    return value


def parse_live_api_intent(
    api_key: str,
    question: str,
    history: list[dict[str, str]],
) -> LiveAPIIntent:
    tools = {
        name: {
            "domain": tool.domain,
            "operation": tool.operation,
            "required_parameters": list(tool.required_parameters),
        }
        for name, tool in READ_ONLY_TOOLS.items()
    }
    recent = [
        {"role": item.get("role"), "content": str(item.get("content", ""))[:400]}
        for item in history[-4:]
        if item.get("role") in {"user", "assistant"}
    ]
    system = f"""You are a strict multilingual Shipra read-only API intent parser.
Return one JSON object only. Never invent a tool or parameter.
Orders, order payment counts, order statuses and order details are handled elsewhere;
for every order-related question return is_live_query=false.

Allowed tools:
{json.dumps(tools, separators=(',', ':'))}

Schema:
{{
  "is_live_query": boolean,
  "tool": string|null,
  "parameters": object,
  "language": "English"|"Roman Urdu"|"Arabic",
  "confidence": number,
  "needs_clarification": boolean,
  "clarification": string
}}

Rules:
- Understand spelling mistakes and English, Roman Urdu, Urdu script and Arabic.
- Count requests use the matching list tool; the executor derives TotalCount safely.
- A specific detail tool requires its ID. If the ID is absent, ask for it.
- Never convert a name into an ID by guessing.
- Never select a mutation operation.
- If no allowed tool matches, return is_live_query=false.
- Match the user's language unless they explicitly request another language.
"""
    response = Groq(api_key=api_key, timeout=20, max_retries=1).chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "system", "content": system}, *recent, {"role": "user", "content": question}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = _json_object(response.choices[0].message.content or "")
    if raw.get("is_live_query") is not True:
        return LiveAPIIntent(is_live_query=False)
    tool_name = str(raw.get("tool") or "")
    if tool_name not in READ_ONLY_TOOLS:
        raise ValueError("The model selected a non-allow-listed tool.")
    confidence = float(raw.get("confidence") or 0)
    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    language = str(raw.get("language") or "English")
    if language not in {"English", "Roman Urdu", "Arabic"}:
        language = "English"
    tool = READ_ONLY_TOOLS[tool_name]
    missing = [key for key in tool.required_parameters if parameters.get(key) in {None, ""}]
    needs_clarification = raw.get("needs_clarification") is True or bool(missing) or confidence < 0.85
    clarification = str(raw.get("clarification") or "").strip()
    if missing and not clarification:
        clarification = f"Please provide {', '.join(missing)}."
    return LiveAPIIntent(
        is_live_query=True,
        tool=tool_name,
        parameters=parameters,
        language=language,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification=clarification,
    )
