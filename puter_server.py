#!/usr/bin/env python3
"""
Puter.com Reverse OpenAI-Compatible API Server

Accepts OpenAI Chat Completions requests and forwards them to:
  POST https://api.puter.com/drivers/call

with payload:
  {
    "interface": "puter-chat-completion",
    "driver": "claude",
    "test_mode": false,
    "method": "complete",
    "args": {
      "messages": [{"content": "..."}],
      "model": "claude-sonnet-4-20250514",
      "stream": true
    }
  }
"""
import json
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import os

try:
    from .config import (
        PUTER_HEADERS,
        PUTER_AUTH_BEARER,
        SERVER_CONFIG,
        MODEL_MAPPING,
    )
except ImportError:
    from config import (
        PUTER_HEADERS,
        PUTER_AUTH_BEARER,
        SERVER_CONFIG,
        MODEL_MAPPING,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PUTER_URL = "https://api.puter.com/drivers/call"
REQUEST_TIMEOUT = 120


# ===== OpenAI-compatible models =====
class OpenAIMessage(BaseModel):
    role: Optional[str] = Field(default=None, description="Role")
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def get_text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            parts: List[str] = []
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return str(self.content) if self.content is not None else ""

    class Config:
        extra = "allow"


class OpenAIFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class OpenAITool(BaseModel):
    type: str = Field(default="function")
    function: Optional[OpenAIFunction] = None

    class Config:
        extra = "allow"


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    functions: Optional[List[OpenAIFunction]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class OpenAIChoice(BaseModel):
    index: int = 0
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: Optional[Dict[str, int]] = None


class OpenAIStreamChoice(BaseModel):
    index: int = 0
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class OpenAIStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


def _build_puter_payload(openai_req: OpenAIChatRequest) -> Dict[str, Any]:
    # Map OpenAI messages to Puter format: only 'content' is used
    mapped_messages: List[Dict[str, str]] = []
    for m in openai_req.messages:
        txt = m.get_text()
        mapped_messages.append({"content": txt})

    # Model mapping: map OpenAI model key -> (driver, puter_model)
    mapping = MODEL_MAPPING.get(openai_req.model) or MODEL_MAPPING.get("default")
    driver = mapping["driver"]
    puter_model = mapping["puter_model"]

    payload: Dict[str, Any] = {
        "interface": "puter-chat-completion",
        "driver": driver,
        "test_mode": False,
        "method": "complete",
        "args": {
            "messages": mapped_messages,
            "model": puter_model,
            "stream": True,  # always request streaming upstream; we aggregate if needed
        },
    }
    return payload


def _headers_with_auth() -> Dict[str, str]:
    h = dict(PUTER_HEADERS)
    h["authorization"] = f"Bearer {PuterAuth.token}"
    return h


class PuterAuth:
    token: str = PUTER_AUTH_BEARER


async def _stream_openai_chunks(openai_req: OpenAIChatRequest, request_id: str) -> AsyncGenerator[str, None]:
    headers = _headers_with_auth()
    payload = _build_puter_payload(openai_req)

    with requests.Session() as sess:
        try:
            resp = sess.post(
                PUTER_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Upstream connection error: {e}")

        if resp.status_code != 200:
            detail = resp.text[:500]
            raise HTTPException(status_code=502, detail=f"Upstream error {resp.status_code}: {detail}")

        created = int(time.time())

        # Initial role chunk
        initial = OpenAIStreamChunk(
            id=request_id,
            created=created,
            model=openai_req.model,
            choices=[OpenAIStreamChoice(index=0, delta={"role": "assistant"}, finish_reason=None)],
        )
        yield f"data: {initial.model_dump_json()}\n\n"

        # Stream content
        for raw in resp.iter_lines():
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue

            text_piece: Optional[str] = None
            # Many APIs stream JSON lines; try to parse
            try:
                obj = json.loads(line)
                # Common keys
                for k in ("delta", "text", "content", "output"):
                    if isinstance(obj.get(k), str) and obj.get(k):
                        text_piece = obj.get(k)
                        break
            except Exception:
                # Fallback to raw text
                if line and line != "[DONE]":
                    text_piece = line

            if not text_piece:
                continue

            chunk = OpenAIStreamChunk(
                id=request_id,
                created=created,
                model=openai_req.model,
                choices=[OpenAIStreamChoice(index=0, delta={"content": text_piece}, finish_reason=None)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        final = OpenAIStreamChunk(
            id=request_id,
            created=created,
            model=openai_req.model,
            choices=[OpenAIStreamChoice(index=0, delta={}, finish_reason="stop")],
        )
        yield f"data: {final.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


def _complete_non_streaming(openai_req: OpenAIChatRequest) -> str:
    headers = _headers_with_auth()
    payload = _build_puter_payload(openai_req)
    payload["args"]["stream"] = True

    with requests.Session() as sess:
        try:
            resp = sess.post(
                PUTER_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Upstream connection error: {e}")

        if resp.status_code != 200:
            detail = resp.text[:500]
            raise HTTPException(status_code=502, detail=f"Upstream error {resp.status_code}: {detail}")

        parts: List[str] = []
        for raw in resp.iter_lines():
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue
            try:
                obj = json.loads(line)
                for k in ("delta", "text", "content", "output"):
                    if isinstance(obj.get(k), str) and obj.get(k):
                        parts.append(obj.get(k))
                        break
            except Exception:
                if line and line != "[DONE]":
                    parts.append(line)
        return "".join(parts)


# ===== FastAPI app =====
app = FastAPI(
    title="Puter Reverse OpenAI API",
    version="1.0.0",
    description="OpenAI-compatible API proxying to api.puter.com"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Puter Reverse OpenAI API", "status": "running", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": int(time.time())}


@app.get("/v1/models")
async def models():
    created = int(time.time())
    data = []
    for key in [k for k in MODEL_MAPPING.keys() if k != "default"]:
        data.append({"id": key, "object": "model", "created": created, "owned_by": "puter"})
    if not data:
        data.append({"id": "claude-sonnet-4-20250514", "object": "model", "created": created, "owned_by": "puter"})
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat(request: OpenAIChatRequest):
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    logger.info(f"[{req_id}] model={request.model}, stream={bool(request.stream)}")

    if bool(request.stream):
        return StreamingResponse(
            _stream_openai_chunks(request, req_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )

    content = _complete_non_streaming(request)
    created = int(time.time())
    response = OpenAIChatResponse(
        id=req_id,
        created=created,
        model=request.model,
        choices=[OpenAIChoice(index=0, message={"role": "assistant", "content": content}, finish_reason="stop")],
        usage={
            "prompt_tokens": len(" ".join([m.get_text() for m in request.messages]).split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(" ".join([m.get_text() for m in request.messages]).split()) + len(content.split()),
        },
    )
    return response


@app.post("/v1/chat/completions/raw")
async def raw(req: Request):
    body = await req.body()
    try:
        obj = json.loads(body)
        _ = OpenAIChatRequest(**obj)
        return {"valid": True}
    except Exception as e:
        return JSONResponse(status_code=422, content={"valid": False, "error": str(e)})


if __name__ == "__main__":
    try:
        import uvicorn
        host = os.getenv("HOST", SERVER_CONFIG.get("host", "0.0.0.0"))
        port = int(os.getenv("PORT", SERVER_CONFIG.get("port", 7860)))
        logger.info(f"Starting Puter Reverse API on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
