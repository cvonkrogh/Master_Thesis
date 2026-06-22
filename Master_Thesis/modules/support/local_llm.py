"""
Local LLM backend (Ollama / llama.cpp-compatible API).

Chintan's requirement: no cloud LLM for extraction. This module calls a model
running locally via Ollama's HTTP API (default http://127.0.0.1:11434).

Setup:
    brew install ollama          # macOS
    ollama serve                 # start server (or run Ollama app)
    ollama pull llama3.1:8b     # download a model once

Environment (optional, in .env):
    OLLAMA_HOST=http://127.0.0.1:11434
    OLLAMA_MODEL=llama3.1:8b
    OLLAMA_TIMEOUT=3600          # seconds per request (default 1800 = 30 min)
"""

from __future__ import annotations

import json
import os
import re
from typing import TypeVar

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 1800.0  # 30 min — local 8B models can be slow on larger decks

T = TypeVar("T", bound=BaseModel)


def ollama_host() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def ollama_model(default: str = DEFAULT_OLLAMA_MODEL) -> str:
    load_dotenv()
    return os.getenv("OLLAMA_MODEL", default)


def ollama_timeout(default: float = DEFAULT_TIMEOUT) -> float:
    load_dotenv()
    raw = os.getenv("OLLAMA_TIMEOUT", "").strip()
    if not raw:
        return default
    return float(raw)


def check_ollama(model: str | None = None) -> None:
    """Raise RuntimeError if Ollama is not reachable or model is missing."""
    host = ollama_host()
    model = model or ollama_model()
    try:
        r = httpx.get(f"{host}/api/tags", timeout=10.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(
            f"Ollama is not running at {host}. Install Ollama and run "
            f"`ollama serve`, then `ollama pull {model}`."
        ) from e

    names = {m.get("name", "") for m in r.json().get("models", [])}
    if not any(n == model or n.startswith(f"{model}:") for n in names):
        raise RuntimeError(
            f"Model {model!r} not found in Ollama. Run: ollama pull {model}"
        )


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _resolve_json_refs(schema: dict) -> dict:
    """Inline $ref / $defs so Ollama gets a flat JSON Schema."""
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                if ref_name in defs:
                    return resolve(dict(defs[ref_name]))
            return {k: resolve(v) for k, v in node.items() if k != "$defs" and k != "definitions"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(dict(schema))
    resolved.pop("$defs", None)
    resolved.pop("definitions", None)
    return resolved


def _ollama_json_format(schema_model: type[BaseModel]) -> dict:
    """JSON Schema for Ollama's structured `format` parameter."""
    return _resolve_json_refs(schema_model.model_json_schema())


_JSON_OUTPUT_RULES = (
    "\n\nOUTPUT FORMAT (critical):\n"
    "Return ONE JSON object with exactly two top-level keys: `fields` and `evidence`.\n"
    "- `fields`: object mapping each schema field name to a short string (use \"\" if unknown).\n"
    "- `evidence`: object mapping each schema field name to a list of slide numbers (use [] if unknown).\n"
    "Do NOT return a JSON Schema, $defs, properties, or $ref. Return actual extracted values only."
)


def _call_ollama_chat(
    host: str,
    model: str,
    messages: list[dict[str, str]],
    json_format: dict,
    temperature: float,
    timeout: float,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": json_format,
        "options": {"temperature": temperature},
    }
    try:
        resp = httpx.post(f"{host}/api/chat", json=body, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e

    raw = resp.json().get("message", {}).get("content", "")
    if not raw:
        raise RuntimeError("Ollama returned empty content.")
    return raw


def chat_json(
    messages: list[dict[str, str]],
    schema_model: type[T],
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float | None = None,
) -> T:
    """Call Ollama /api/chat with JSON output and validate with Pydantic."""
    host = ollama_host()
    model = model or ollama_model()
    timeout = timeout if timeout is not None else ollama_timeout()
    json_format = _ollama_json_format(schema_model)

    payload_messages: list[dict[str, str]] = []
    for i, m in enumerate(messages):
        content = m["content"]
        if i == 0 and m["role"] == "system":
            content = content + _JSON_OUTPUT_RULES
        payload_messages.append({"role": m["role"], "content": content})

    raw = _call_ollama_chat(host, model, payload_messages, json_format, temperature, timeout)

    try:
        data = _extract_json(raw)
        return schema_model.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        # One retry: small models sometimes echo schema on the first attempt.
        retry_messages = payload_messages + [
            {
                "role": "user",
                "content": (
                    "Your last reply was not valid extracted data. "
                    "Reply again with ONLY a JSON object containing `fields` and `evidence` "
                    "with real string values and integer slide lists — not a JSON Schema."
                ),
            }
        ]
        raw = _call_ollama_chat(host, model, retry_messages, json_format, temperature, timeout)
        try:
            data = _extract_json(raw)
            return schema_model.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            snippet = raw[:500].replace("\n", " ")
            raise RuntimeError(f"Invalid JSON from Ollama: {e}. Raw: {snippet!r}") from e
