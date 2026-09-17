# Shipra Code Assistant

Public, guest-mode code assistant for the Shipra frontend and backend. Chats are stored only in the visitor's Streamlit session and are never shared or persisted.

## Install and test

```bash
pip install -r requirements.txt
python self_check.py --fast
python self_check.py
streamlit run app.py
```

`--fast` validates the supplied FAISS/index/source snapshot without downloading models. The normal check also loads the embedding and reranker models and runs retrieval regression tests.

## Required secret

Create `.streamlit/secrets.toml` locally or add this in Streamlit Cloud Secrets:

```toml
GROQ_API_KEY = "your-key"
```

## Rebuild after Shipra source changes

Run the source-safe indexer with the latest frontend/backend ZIPs. It skips `.env`, `appsettings*.json`, build folders and other sensitive/generated files.

```bash
python build_index.py \
  --frontend-zip Shipra.Frontend.zip \
  --backend-zip Shipra.Backend.API.Application.zip \
  --backend-zip Shipra.Backend.API.Core.zip \
  --backend-zip Shipra.Backend.API.Infrastructure.zip \
  --backend-zip Shipra.Backend.API.Web.zip
python self_check.py
```

Do not commit source ZIPs, `.env`, Streamlit secrets or production configuration.
