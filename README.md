# Puter Reverse OpenAI-Compatible API

FastAPI server that exposes OpenAI Chat Completions endpoints and proxies to `https://api.puter.com/drivers/call`.

## Run locally

```bash
cd "Puter reversed"
pip install -r requirements.txt
python start_server.py
```

Defaults to `http://localhost:8781`. You can override with env vars `HOST` and `PORT`.

## Render deployment

- Use the following settings:
  - Build command: `pip install -r "Puter reversed/requirements.txt"`
  - Start command: `python "Puter reversed/puter_server.py"`
  - Environment: Python 3.11
  - PORT env var: Render injects `PORT`, code reads it

Alternatively, run with uvicorn directly:
```
uvicorn puter_server:app --host 0.0.0.0 --port $PORT
```

## Hugging Face Spaces (Docker)

Use the provided Dockerfile. Spaces use port 7860 automatically.

## Test

```bash
curl http://localhost:8781/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role":"user","content":"Tell me something new about chemistry."}],
    "stream": false
  }'
```


