"""FastAPI application exposing the Copilot proxy endpoints."""
from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .config import (
    get_api_key as get_config_api_key,
    get_base_url as get_config_base_url,
    get_context_length,
    get_model_name as get_config_model_name,
    get_temperature,
)

DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
DEFAULT_MODEL = "GLM-4.7"
API_KEY_ENV_VARS = ("ZAI_API_KEY", "ZAI_CODING_API_KEY", "GLM_API_KEY")
BASE_URL_ENV_VAR = "ZAI_API_BASE_URL"
CHAT_COMPLETION_PATH = "/chat/completions"

THINK_TAGS_ENV_VAR = "COPILOT_PROXY_THINK_TAGS"
STRIP_THINK_TAGS_ENV_VAR = "COPILOT_PROXY_STRIP_THINK_TAGS"
_THINK_OPEN = "<think>\n"
_THINK_CLOSE = "\n</think>\n\n"

_THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>\s*", flags=re.DOTALL)


def _is_truthy_env(env_var: str, default: bool) -> bool:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class _StreamRewriteState:
    def __init__(self, *, wrap_think_tags: bool) -> None:
        self.wrap_think_tags = wrap_think_tags
        self.think_open_by_index: dict[int, bool] = {}
        self.last_meta: dict[str, object] | None = None


def _strip_think_blocks(value: str) -> str:
    stripped = _THINK_BLOCK_RE.sub("", value)
    stripped = stripped.lstrip("\n")
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    if not stripped.strip():
        return value
    return stripped


def _strip_think_tags_from_messages(body: dict) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        content = message.get("content")
        if isinstance(content, str):
            message["content"] = _strip_think_blocks(content)
            continue

        if isinstance(content, list):
            changed = False
            new_parts = []
            for part in content:
                if not isinstance(part, dict):
                    new_parts.append(part)
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    new_text = _strip_think_blocks(text)
                    if new_text != text:
                        changed = True
                        new_parts.append({**part, "text": new_text})
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            if changed:
                message["content"] = new_parts



def _rewrite_reasoning_deltas(data: dict, state: _StreamRewriteState) -> bool:
    """Map reasoning_content into content so clients that ignore it stream tokens."""

    changed = False
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False

    for fallback_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue

        raw_index = choice.get("index", fallback_index)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue

        delta = choice.get("delta")
        if delta is None:
            delta = {}
            choice["delta"] = delta
            changed = True
        if not isinstance(delta, dict):
            continue

        # Copilot ignores reasoning_content; map it to content to avoid silent streams.
        has_reasoning = "reasoning_content" in delta
        reasoning_value = delta.pop("reasoning_content", None) if has_reasoning else None
        if has_reasoning:
            changed = True

        think_open = state.think_open_by_index.get(index, False)

        reasoning_text: str | None
        if reasoning_value is None:
            reasoning_text = None
        elif isinstance(reasoning_value, str):
            reasoning_text = reasoning_value
        else:
            reasoning_text = str(reasoning_value)

        content_value = delta.get("content")
        content_text: str | None
        if content_value is None:
            content_text = None
        elif isinstance(content_value, str):
            content_text = content_value
        else:
            content_text = str(content_value)

        if reasoning_text is not None:
            prefix = ""
            suffix = ""
            if state.wrap_think_tags and not think_open:
                prefix = _THINK_OPEN
                think_open = True

            if content_text:
                # Some backends transition to normal content mid-chunk. Close <think> first so
                # editor/chat UIs can keep reasoning separate from the final answer.
                if state.wrap_think_tags and think_open:
                    suffix = _THINK_CLOSE
                    think_open = False
                delta["content"] = f"{prefix}{reasoning_text}{suffix}{content_text}"
            else:
                delta["content"] = f"{prefix}{reasoning_text}"
            changed = True

        elif content_text and state.wrap_think_tags and think_open:
            # First normal content after reasoning: close the <think> block.
            delta["content"] = f"{_THINK_CLOSE}{content_text}"
            changed = True
            think_open = False

        # If the stream ends without ever producing normal content, close before finishing.
        if state.wrap_think_tags and think_open and choice.get("finish_reason") is not None:
            existing = delta.get("content")
            if existing:
                delta["content"] = f"{existing}{_THINK_CLOSE}"
            else:
                delta["content"] = _THINK_CLOSE
            changed = True
            think_open = False

        state.think_open_by_index[index] = think_open

    return changed


def _build_think_close_sse(state: _StreamRewriteState) -> bytes | None:
    open_indices = [idx for idx, is_open in state.think_open_by_index.items() if is_open]
    if not open_indices:
        return None

    meta = state.last_meta or {}
    payload: dict[str, object] = {
        "object": meta.get("object", "chat.completion.chunk"),
        "choices": [{"index": idx, "delta": {"content": _THINK_CLOSE}} for idx in open_indices],
    }
    for key in ("id", "created", "model"):
        value = meta.get(key)
        if value is not None:
            payload[key] = value

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return b"data: " + data


def _transform_sse_event(event: bytes, state: _StreamRewriteState) -> bytes:
    if not event:
        return event
    lines = event.splitlines()
    changed = False
    new_lines: list[bytes] = []
    for line in lines:
        line = line.rstrip(b"\r")
        if line.startswith(b"data:"):
            payload = line[len(b"data:") :].lstrip()
            payload_stripped = payload.strip()
            if payload_stripped == b"[DONE]":
                if state.wrap_think_tags:
                    close_sse = _build_think_close_sse(state)
                    if close_sse:
                        return close_sse + b"\n\n" + b"data: [DONE]"
                return b"data: [DONE]"
            try:
                payload_text = payload.decode("utf-8")
            except UnicodeDecodeError:
                new_lines.append(line)
                continue
            try:
                data = json.loads(payload_text)
            except json.JSONDecodeError:
                new_lines.append(line)
                continue
            if isinstance(data, dict):
                state.last_meta = {k: data.get(k) for k in ("id", "created", "model", "object")}
            if isinstance(data, dict) and _rewrite_reasoning_deltas(data, state):
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                changed = True
            new_lines.append(b"data: " + payload)
        else:
            new_lines.append(line)
    if not changed:
        return event
    return b"\n".join(new_lines)



def get_model_catalog():
    """Generate the model catalog dynamically based on config."""
    context_length = get_context_length()
    return [
        {
            "name": "GLM-4.7",
            "model": "GLM-4.7",
            "modified_at": "2025-12-21T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.7",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4-Plus",
            "model": "GLM-4-Plus",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4-Plus",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6",
            "model": "GLM-4.6",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5",
            "model": "GLM-4.5",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5-Air",
            "model": "GLM-4.5-Air",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5-Air",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5-AirX",
            "model": "GLM-4.5-AirX",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5-AirX",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5-Flash",
            "model": "GLM-4.5-Flash",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5-Flash",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6V",
            "model": "GLM-4.6V",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6V",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6V-Flash",
            "model": "GLM-4.6V-Flash",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6V-Flash",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.6V-FlashX",
            "model": "GLM-4.6V-FlashX",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.6V-FlashX",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4.5V",
            "model": "GLM-4.5V",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4.5V",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "AutoGLM-Phone-Multilingual",
            "model": "AutoGLM-Phone-Multilingual",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "AutoGLM-Phone-Multilingual",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
        {
            "name": "GLM-4-32B-0414-128K",
            "model": "GLM-4-32B-0414-128K",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "GLM-4-32B-0414-128K",
            "details": {
                "format": "glm",
                "family": "glm",
                "families": ["glm"],
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
        },
    ]


MODEL_CATALOG = get_model_catalog()


def _get_api_key() -> str:
    # First try to get API key from config file
    config_api_key = get_config_api_key()
    if config_api_key:
        return config_api_key

    # Fall back to environment variables
    for env_var in API_KEY_ENV_VARS:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key.strip()

    raise RuntimeError(
        "Missing Z.AI API key. Please set it using 'copilot-proxy config set-api-key <key>' "
        f"or set one of the following environment variables: {', '.join(API_KEY_ENV_VARS)}"
    )


def _get_base_url() -> str:
    # First try to get base URL from config file
    config_base_url = get_config_base_url()
    if config_base_url:
        base_url = config_base_url
    else:
        # Fall back to environment variable or default
        base_url = os.getenv(BASE_URL_ENV_VAR, DEFAULT_BASE_URL).strip()
        if not base_url:
            base_url = DEFAULT_BASE_URL

    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    return base_url.rstrip("/")


def _get_chat_completion_url() -> str:
    base_url = _get_base_url()
    if base_url.endswith(CHAT_COMPLETION_PATH):
        return base_url
    return f"{base_url}{CHAT_COMPLETION_PATH}"


@asynccontextmanager
async def _lifespan(app: FastAPI):  # noqa: D401 - FastAPI lifespan signature
    """Ensure configuration is ready before serving requests."""

    try:
        _ = _get_api_key()
        print("GLM Coding Plan proxy is ready.")
    except Exception as exc:  # pragma: no cover - startup logging
        print(f"Failed to initialise GLM Coding Plan proxy: {exc}")
    yield


def create_app() -> FastAPI:
    """Create and return a configured FastAPI application."""

    app = FastAPI(lifespan=_lifespan)

    @app.get("/")
    async def root():  # noqa: D401 - FastAPI route
        """Return a simple health message."""

        return {"message": "GLM Coding Plan proxy is running"}

    @app.get("/api/ps")
    async def list_running_models():  # noqa: D401 - FastAPI route
        """Return an empty list as we do not host local models."""

        return {"models": []}

    @app.get("/api/version")
    async def get_version():  # noqa: D401 - FastAPI route
        """Expose a version compatible with the Ollama API expectations."""

        return {"version": "0.6.4"}

    @app.get("/api/tags")
    @app.get("/api/list")
    async def list_models():  # noqa: D401 - FastAPI route
        """Return the static catalog of GLM models."""

        return {"models": get_model_catalog()}

    @app.post("/api/show")
    async def show_model(request: Request):  # noqa: D401 - FastAPI route
        """Handle Ollama-compatible model detail queries."""

        try:
            body = await request.json()
            model_name = body.get("model")
        except Exception:
            model_name = DEFAULT_MODEL

        if not model_name:
            model_name = get_config_model_name() or DEFAULT_MODEL

        context_length = get_context_length()

        return {
            "template": "{{ .System }}\n{{ .Prompt }}",
            "capabilities": ["tools"],
            "details": {
                "family": "glm",
                "families": ["glm"],
                "format": "glm",
                "parameter_size": "cloud",
                "quantization_level": "cloud",
            },
            "model_info": {
                "general.basename": model_name,
                "general.architecture": "glm",
                "glm.context_length": context_length,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):  # noqa: D401 - FastAPI route
        """Forward chat completion calls to the Z.AI backend."""

        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid JSON body. In PowerShell, avoid backslash-escaping quotes; use "
                    "ConvertTo-Json and pass it to curl via stdin/file (e.g. `--data-binary '@-'`)."
                ),
            ) from exc

        if not body.get("model"):
            body["model"] = get_config_model_name() or DEFAULT_MODEL

        # Apply temperature override if configured
        config_temp = get_temperature()
        if config_temp is not None:
            body["temperature"] = config_temp

        # Strip <think> blocks from assistant messages before forwarding upstream.
        # This prevents multi-turn prompt bloat when clients store/resend reasoning in `content`.
        # Disable via `COPILOT_PROXY_STRIP_THINK_TAGS=0` if you want the model to see prior <think> blocks.
        if _is_truthy_env(STRIP_THINK_TAGS_ENV_VAR, default=True):
            _strip_think_tags_from_messages(body)

        stream = body.get("stream", False)

        api_key = _get_api_key()
        chat_completion_url = _get_chat_completion_url()

        async def generate_chunks() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    }
                    response = await client.post(
                        chat_completion_url,
                        headers=headers,
                        json=body,
                        timeout=None,
                    )
                    response.raise_for_status()

                    rewrite_state = _StreamRewriteState(
                        wrap_think_tags=_is_truthy_env(THINK_TAGS_ENV_VAR, default=True)
                    )
                    buffer = b""
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        buffer += chunk
                        while True:
                            sep_index = buffer.find(b"\r\n\r\n")
                            sep_len = 4
                            lf_index = buffer.find(b"\n\n")
                            if lf_index != -1 and (sep_index == -1 or lf_index < sep_index):
                                sep_index = lf_index
                                sep_len = 2
                            if sep_index == -1:
                                break
                            event = buffer[:sep_index]
                            buffer = buffer[sep_index + sep_len :]
                            yield _transform_sse_event(event, rewrite_state) + b"\n\n"
                    if buffer:
                        yield _transform_sse_event(buffer, rewrite_state)

                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 401:
                        raise RuntimeError("Unauthorized. Check your Z.AI API key.") from exc
                    raise

        if stream:
            return StreamingResponse(generate_chunks(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            response = await client.post(chat_completion_url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()

    return app


app = create_app()

__all__ = [
    "API_KEY_ENV_VARS",
    "BASE_URL_ENV_VAR",
    "CHAT_COMPLETION_PATH",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MODEL_CATALOG",
    "app",
    "create_app",
]
