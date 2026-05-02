"""
openrouter_media_proxy.py

A minimal FastAPI proxy that translates OpenAI-compatible image and audio
requests into OpenRouter's chat/completions API format.

Designed to run as an internal Docker-network sidecar in front of Open WebUI.
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ---- Configuration ---------------------------------------------------------

app = FastAPI()

OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1")
UPSTREAM_TIMEOUT = int(os.getenv("UPSTREAM_TIMEOUT", "120"))
DEFAULT_IMAGE_MODALITIES = os.getenv("DEFAULT_IMAGE_MODALITIES", "image")
IMAGE_EDIT_MASK_MODE = os.getenv("IMAGE_EDIT_MASK_MODE", "reject").strip().lower() or "reject"

# ---- Logging ---------------------------------------------------------------

_raw_level = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, _raw_level, logging.INFO)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("openrouter_media_proxy")

# ---- Constants / mappings --------------------------------------------------

# OpenAI pixel dimensions -> OpenRouter aspect ratios
SIZE_TO_ASPECT: dict[str, str] = {
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "256x256": "1:1",
    "512x512": "1:1",
}

# OpenAI quality labels -> OpenRouter image_size tokens
QUALITY_TO_IMAGE_SIZE: dict[str, str] = {
    "low": "1K",
    "standard": "1K",
    "medium": "2K",
    "hd": "2K",
    "high": "4K",
}

IMAGE_RESPONSE_FORMAT_VALUES = ("b64_json", "url")
IMAGE_SIZE_VALUES = ("auto", "256x256", "512x512", "1024x1024", "1536x1024", "1024x1536", "1792x1024", "1024x1792")
IMAGE_QUALITY_VALUES = ("auto", "low", "standard", "medium", "hd", "high")
IMAGE_STYLE_VALUES = ("natural", "vivid")
IMAGE_BACKGROUND_VALUES = ("transparent", "opaque", "auto")
IMAGE_OUTPUT_FORMAT_VALUES = ("png", "jpeg", "webp")
IMAGE_MODERATION_VALUES = ("low", "auto")
IMAGE_INPUT_FIDELITY_VALUES = ("high", "low")
IMAGE_EDIT_MASK_MODE_VALUES = ("reject", "passthrough")

if IMAGE_EDIT_MASK_MODE not in IMAGE_EDIT_MASK_MODE_VALUES:
    logger.warning(
        "Invalid IMAGE_EDIT_MASK_MODE=%s; defaulting to reject",
        IMAGE_EDIT_MASK_MODE,
    )

DATA_URL_RE = re.compile(r"data:image/[^;]+;base64,(.*)", re.DOTALL)

AUDIO_CONTENT_TYPE_TO_FORMAT: dict[str, str] = {
    "audio/aac": "aac",
    "audio/aiff": "aiff",
    "audio/flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/pcm": "pcm16",
    "audio/wav": "wav",
    "audio/webm": "ogg",
    "audio/x-aac": "aac",
    "audio/x-aiff": "aiff",
    "audio/x-flac": "flac",
    "audio/x-wav": "wav",
}

AUDIO_EXTENSION_TO_FORMAT: dict[str, str] = {
    "aac": "aac",
    "aif": "aiff",
    "aiff": "aiff",
    "flac": "flac",
    "m4a": "m4a",
    "mp3": "mp3",
    "oga": "ogg",
    "ogg": "ogg",
    "opus": "ogg",
    "pcm": "pcm16",
    "raw": "pcm16",
    "wav": "wav",
    "wave": "wav",
}

SPEECH_RESPONSE_FORMAT_TO_UPSTREAM: dict[str, str] = {
    "aac": "aac",
    "flac": "flac",
    "mp3": "mp3",
    "opus": "opus",
    "pcm": "pcm16",
    "wav": "wav",
}

SPEECH_RESPONSE_FORMAT_TO_MEDIA_TYPE: dict[str, str] = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "wav": "audio/wav",
}


# ---- Helpers ---------------------------------------------------------------


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex


def _image_modalities() -> list[str]:
    return [m.strip() for m in DEFAULT_IMAGE_MODALITIES.split(",") if m.strip()]


def _mask_passthrough_enabled() -> bool:
    if IMAGE_EDIT_MASK_MODE in IMAGE_EDIT_MASK_MODE_VALUES:
        return IMAGE_EDIT_MASK_MODE == "passthrough"
    return False


def _mask_unsupported_response() -> JSONResponse:
    return error_response(
        400,
        (
            "Mask-based image edits are not supported by this proxy. "
            "OpenRouter chat/completions does not expose a native mask primitive, "
            "so true OpenAI-style inpainting semantics cannot be translated."
        ),
    )


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_string_option(
    name: str,
    value: Any,
    allowed_values: tuple[str, ...],
) -> tuple[str | None, JSONResponse | None]:
    normalized = _normalize_optional_text(value)
    if normalized is None:
        return None, None

    lowered = normalized.lower()
    if lowered not in allowed_values:
        allowed = ", ".join(allowed_values)
        return None, error_response(
            400,
            f"Invalid {name}. Supported values: {allowed}.",
        )

    return lowered, None


def _validate_integer_option(
    name: str,
    value: Any,
    min_value: int | None = None,
    max_value: int | None = None,
) -> tuple[int | None, JSONResponse | None]:
    if value in (None, ""):
        return None, None

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None, error_response(400, f"Invalid {name}. Expected an integer.")

    if min_value is not None and number < min_value:
        return None, error_response(
            400,
            f"Invalid {name}. Must be greater than or equal to {min_value}.",
        )
    if max_value is not None and number > max_value:
        return None, error_response(
            400,
            f"Invalid {name}. Must be less than or equal to {max_value}.",
        )

    return number, None


def _parse_image_count(value: Any) -> tuple[int, JSONResponse | None]:
    if value in (None, ""):
        return 1, None

    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1, error_response(400, "Invalid n. Expected an integer.")

    return max(1, min(number, 10)), None


def _parse_image_user(value: Any) -> tuple[str | None, JSONResponse | None]:
    if value in (None, ""):
        return None, None
    if not isinstance(value, str):
        return None, error_response(400, "Invalid user. Expected a string.")

    user = value.strip()
    return (user or None), None


def _parse_image_options(
    *,
    size: Any = None,
    quality: Any = None,
    style: Any = None,
    background: Any = None,
    response_format: Any = None,
    output_format: Any = None,
    output_compression: Any = None,
    moderation: Any = None,
    user: Any = None,
    seed: Any = None,
    input_fidelity: Any = None,
) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    normalized_size, error = _validate_string_option("size", size, IMAGE_SIZE_VALUES)
    if error:
        return None, error

    normalized_quality, error = _validate_string_option(
        "quality", quality, IMAGE_QUALITY_VALUES
    )
    if error:
        return None, error

    normalized_style, error = _validate_string_option("style", style, IMAGE_STYLE_VALUES)
    if error:
        return None, error

    normalized_background, error = _validate_string_option(
        "background", background, IMAGE_BACKGROUND_VALUES
    )
    if error:
        return None, error

    normalized_response_format, error = _validate_string_option(
        "response_format", response_format, IMAGE_RESPONSE_FORMAT_VALUES
    )
    if error:
        return None, error

    normalized_output_format, error = _validate_string_option(
        "output_format", output_format, IMAGE_OUTPUT_FORMAT_VALUES
    )
    if error:
        return None, error

    normalized_moderation, error = _validate_string_option(
        "moderation", moderation, IMAGE_MODERATION_VALUES
    )
    if error:
        return None, error

    normalized_input_fidelity, error = _validate_string_option(
        "input_fidelity", input_fidelity, IMAGE_INPUT_FIDELITY_VALUES
    )
    if error:
        return None, error

    parsed_output_compression, error = _validate_integer_option(
        "output_compression", output_compression, 0, 100
    )
    if error:
        return None, error

    parsed_seed, error = _validate_integer_option("seed", seed)
    if error:
        return None, error

    parsed_user, error = _parse_image_user(user)
    if error:
        return None, error

    return {
        "background": normalized_background,
        "input_fidelity": normalized_input_fidelity,
        "moderation": normalized_moderation,
        "output_compression": parsed_output_compression,
        "output_format": normalized_output_format,
        "quality": normalized_quality,
        "response_format": normalized_response_format or "b64_json",
        "seed": parsed_seed,
        "size": normalized_size,
        "style": normalized_style,
        "user": parsed_user,
    }, None


def build_image_config(
    size: str | None,
    quality: str | None,
    background: str | None = None,
    output_format: str | None = None,
    output_compression: int | None = None,
    moderation: str | None = None,
    input_fidelity: str | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if size and size != "auto":
        ar = SIZE_TO_ASPECT.get(size)
        if ar:
            cfg["aspect_ratio"] = ar
    if quality and quality != "auto":
        isz = QUALITY_TO_IMAGE_SIZE.get(quality)
        if isz:
            cfg["image_size"] = isz
    if background:
        cfg["background"] = background
    if output_format:
        cfg["output_format"] = output_format
    if output_compression is not None:
        cfg["output_compression"] = output_compression
    if moderation:
        cfg["moderation"] = moderation
    if input_fidelity:
        cfg["input_fidelity"] = input_fidelity
    return cfg


def upstream_headers(request: Request) -> dict[str, str]:
    """Forward only the Authorization header to OpenRouter."""
    hdrs: dict[str, str] = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        hdrs["Authorization"] = auth
    return hdrs


def error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type}},
    )


def _extract_image_url(candidate: Any) -> str | None:
    if isinstance(candidate, str):
        url = candidate.strip()
        return url or None

    if not isinstance(candidate, dict):
        return None

    url = candidate.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()

    image_url = candidate.get("image_url")
    if isinstance(image_url, str) and image_url.strip():
        return image_url.strip()
    if isinstance(image_url, dict):
        nested_url = image_url.get("url")
        if isinstance(nested_url, str) and nested_url.strip():
            return nested_url.strip()

    source = candidate.get("source")
    if source is not None:
        return _extract_image_url(source)

    return None


def _message_image_urls(message: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def add_url(candidate: Any) -> None:
        url = _extract_image_url(candidate)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    raw_images = message.get("images")
    if isinstance(raw_images, list):
        for item in raw_images:
            add_url(item)

    content = message.get("content")
    content_items: list[Any] = []
    if isinstance(content, list):
        content_items = content
    elif isinstance(content, dict):
        content_items = [content]

    for item in content_items:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() == "text":
            continue
        add_url(item)

    return urls


def _content_shape(content: Any) -> str:
    if content is None:
        return "none"
    if isinstance(content, str):
        return "str"
    if isinstance(content, list):
        item_types: list[str] = []
        for item in content[:4]:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "dict").strip() or "dict"
            else:
                item_type = type(item).__name__
            item_types.append(item_type)
        if len(content) > 4:
            item_types.append("...")
        return f"list[{', '.join(item_types)}]"
    if isinstance(content, dict):
        return "dict"
    return type(content).__name__


def _summarize_image_response(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return f"choices_type={type(choices).__name__}"

    summaries: list[str] = []
    for idx, choice in enumerate(choices[:2]):
        if not isinstance(choice, dict):
            summaries.append(f"choice{idx}: type={type(choice).__name__}")
            continue

        message = choice.get("message") or choice.get("delta") or {}
        if not isinstance(message, dict):
            summaries.append(f"choice{idx}: message_type={type(message).__name__}")
            continue

        image_urls = _message_image_urls(message)
        text = _content_to_text(message.get("content")).strip()
        raw_images = message.get("images")
        raw_image_count = len(raw_images) if isinstance(raw_images, list) else 0
        finish_reason = str(choice.get("finish_reason") or "unknown")
        summaries.append(
            " ".join(
                [
                    f"choice{idx}",
                    f"finish_reason={finish_reason}",
                    f"raw_images={raw_image_count}",
                    f"image_urls={len(image_urls)}",
                    f"content={_content_shape(message.get('content'))}",
                    f"text_len={len(text)}",
                ]
            )
        )

    if len(choices) > 2:
        summaries.append("...")

    return f"choices={len(choices)}; " + "; ".join(summaries)


def extract_images(
    data: dict[str, Any],
    response_format: str = "b64_json",
) -> list[dict[str, str]]:
    """
    Pull base64 images out of an OpenRouter chat/completions response
    and reshape them into OpenAI ImagesResponse.data entries.
    """
    images: list[dict[str, str]] = []
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message") or choice.get("delta") or {}
        if not isinstance(msg, dict):
            continue
        revised: str | None = None
        content = _content_to_text(msg.get("content")).strip()
        if content:
            revised = content
        for url in _message_image_urls(msg):
            entry: dict[str, str] = {}
            if response_format == "url":
                if url:
                    entry["url"] = url
            else:
                match = DATA_URL_RE.match(url)
                if match:
                    entry["b64_json"] = match.group(1)
            if entry:
                if revised:
                    entry["revised_prompt"] = revised
                images.append(entry)
    return images


def _collect_usage_details(details: Any) -> dict[str, int]:
    payload: dict[str, int] = {}
    if not isinstance(details, dict):
        return payload
    for key, value in details.items():
        if isinstance(value, (int, float)):
            payload[str(key)] = int(value)
    return payload


def build_openai_image_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    numeric_tokens = [
        value
        for value in (prompt_tokens, completion_tokens, total_tokens)
        if isinstance(value, (int, float))
    ]
    if not numeric_tokens:
        return None

    result: dict[str, Any] = {
        "input_tokens": int(prompt_tokens or 0),
        "output_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)),
    }

    input_details = _collect_usage_details(usage.get("prompt_tokens_details"))
    if input_details:
        result["input_tokens_details"] = input_details

    output_details = _collect_usage_details(usage.get("completion_tokens_details"))
    if output_details:
        result["output_tokens_details"] = output_details

    return result


def merge_openai_image_usages(usages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not usages:
        return None

    merged: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    input_details: dict[str, int] = {}
    output_details: dict[str, int] = {}

    for usage in usages:
        merged["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        merged["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        merged["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

        usage_input_details = usage.get("input_tokens_details")
        if isinstance(usage_input_details, dict):
            for key, value in usage_input_details.items():
                if isinstance(value, (int, float)):
                    input_details[str(key)] = input_details.get(str(key), 0) + int(value)

        usage_output_details = usage.get("output_tokens_details")
        if isinstance(usage_output_details, dict):
            for key, value in usage_output_details.items():
                if isinstance(value, (int, float)):
                    output_details[str(key)] = output_details.get(str(key), 0) + int(value)

    if input_details:
        merged["input_tokens_details"] = input_details
    if output_details:
        merged["output_tokens_details"] = output_details
    return merged


def build_image_response(
    images: list[dict[str, str]],
    usages: list[dict[str, Any]],
) -> JSONResponse:
    payload: dict[str, Any] = {
        "created": int(time.time()),
        "data": images,
    }
    usage = merge_openai_image_usages(usages)
    if usage:
        payload["usage"] = usage
    return JSONResponse(content=payload)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def extract_text(data: dict[str, Any]) -> str:
    for choice in data.get("choices", []):
        msg = choice.get("message") or choice.get("delta") or {}
        text = _content_to_text(msg.get("content")).strip()
        if text:
            return text
    return ""


def build_openai_usage(data: dict[str, Any]) -> dict[str, Any] | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    numeric_tokens = [
        value
        for value in (prompt_tokens, completion_tokens, total_tokens)
        if isinstance(value, (int, float))
    ]
    if numeric_tokens:
        result: dict[str, Any] = {
            "input_tokens": int(prompt_tokens or 0),
            "output_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)),
            "type": "tokens",
        }
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            detail_payload: dict[str, int] = {}
            audio_tokens = prompt_details.get("audio_tokens")
            text_tokens = prompt_details.get("text_tokens")
            if isinstance(audio_tokens, (int, float)):
                detail_payload["audio_tokens"] = int(audio_tokens)
            if isinstance(text_tokens, (int, float)):
                detail_payload["text_tokens"] = int(text_tokens)
            if detail_payload:
                result["input_token_details"] = detail_payload
        return result

    seconds = usage.get("seconds")
    if isinstance(seconds, (int, float)):
        return {"seconds": float(seconds), "type": "duration"}

    return None


def _augment_prompt(
    prompt: str,
    style: str | None = None,
    background: str | None = None,
) -> str:
    """Append OpenAI-specific style / background hints to the prompt text."""
    parts = [prompt]
    if style == "natural":
        parts.append("Use a natural, realistic style.")
    elif style == "vivid":
        parts.append("Use a vivid, dramatic style.")
    if background == "transparent":
        parts.append("The image should have a transparent background.")
    return " ".join(parts)


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidates: list[str] = []
    stripped = _strip_json_fences(text)
    if stripped:
        candidates.append(stripped)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace:last_brace + 1])

    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _guess_audio_format(upload: Any) -> str:
    content_type = str(getattr(upload, "content_type", "") or "").split(";", 1)[0].strip().lower()
    if content_type in AUDIO_CONTENT_TYPE_TO_FORMAT:
        return AUDIO_CONTENT_TYPE_TO_FORMAT[content_type]

    filename = str(getattr(upload, "filename", "") or "")
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1].strip().lower()
        if ext in AUDIO_EXTENSION_TO_FORMAT:
            return AUDIO_EXTENSION_TO_FORMAT[ext]

    if "/" in content_type:
        subtype = content_type.split("/", 1)[1]
        if subtype in AUDIO_EXTENSION_TO_FORMAT:
            return AUDIO_EXTENSION_TO_FORMAT[subtype]

    return "wav"


def _speech_voice_id(voice: Any) -> str:
    if isinstance(voice, dict):
        candidate = voice.get("id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    if isinstance(voice, str) and voice.strip():
        return voice.strip()
    return "alloy"


def _speech_upstream_format(response_format: str | None) -> str:
    key = str(response_format or "mp3").strip().lower()
    return SPEECH_RESPONSE_FORMAT_TO_UPSTREAM.get(key, "mp3")


def _speech_media_type(response_format: str | None) -> str:
    key = str(response_format or "mp3").strip().lower()
    return SPEECH_RESPONSE_FORMAT_TO_MEDIA_TYPE.get(key, "audio/mpeg")


def _build_audio_instruction(
    task: str,
    response_format: str,
    prompt: str | None = None,
    language: str | None = None,
) -> str:
    parts: list[str] = []

    if task == "translate":
        parts.append("Translate the provided audio into English.")
    else:
        parts.append("Transcribe the provided audio faithfully in the original spoken language.")

    if language and task == "transcribe":
        parts.append(f"The spoken language is likely {language}.")
    if prompt:
        parts.append(f"Follow this spelling, terminology, and style guidance: {prompt}")

    if response_format == "verbose_json":
        if task == "translate":
            parts.append(
                "Return only valid JSON with keys text, language, duration, and optional segments. "
                'Use "english" for the language value. Include duration as a number in seconds when '
                "the model can estimate it. If timing data is unavailable, use an empty segments array."
            )
        else:
            parts.append(
                "Return only valid JSON with keys text, language, duration, and optional segments and words. "
                "Each segment should contain id, seek, start, end, text, tokens, temperature, avg_logprob, "
                "compression_ratio, and no_speech_prob. If word timestamps are unavailable, use an empty words array."
            )
    elif response_format == "diarized_json":
        parts.append(
            "Return only valid JSON with keys task, text, duration, and segments. "
            'Use "transcribe" for task. Each segment must contain id, start, end, speaker, text, and '
            'type set to "transcript.text.segment". If speaker diarization is uncertain, still emit best-effort segments.'
        )
    else:
        parts.append("Return only the transcript text without commentary or markdown.")

    return " ".join(parts)


def _normalize_transcription_verbose(
    payload: dict[str, Any] | None,
    fallback_text: str,
    requested_language: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "duration": 0.0,
        "language": requested_language or "unknown",
        "text": fallback_text,
    }
    if isinstance(payload, dict):
        result["text"] = str(payload.get("text") or fallback_text)
        result["language"] = str(payload.get("language") or result["language"])
        result["duration"] = _as_float(payload.get("duration"), 0.0)
        if isinstance(payload.get("segments"), list):
            result["segments"] = payload["segments"]
        if isinstance(payload.get("words"), list):
            result["words"] = payload["words"]
    return result


def _normalize_translation_verbose(
    payload: dict[str, Any] | None,
    fallback_text: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "duration": 0.0,
        "language": "english",
        "text": fallback_text,
    }
    if isinstance(payload, dict):
        result["text"] = str(payload.get("text") or fallback_text)
        result["duration"] = _as_float(payload.get("duration"), 0.0)
        if isinstance(payload.get("segments"), list):
            result["segments"] = payload["segments"]
    return result


def _normalize_diarized(
    payload: dict[str, Any] | None,
    fallback_text: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "duration": 0.0,
        "segments": [],
        "task": "transcribe",
        "text": fallback_text,
    }
    if isinstance(payload, dict):
        result["text"] = str(payload.get("text") or fallback_text)
        result["duration"] = _as_float(payload.get("duration"), 0.0)
        raw_segments = payload.get("segments")
        if isinstance(raw_segments, list):
            normalized_segments: list[dict[str, Any]] = []
            for idx, segment in enumerate(raw_segments):
                if not isinstance(segment, dict):
                    continue
                normalized_segments.append(
                    {
                        "id": str(segment.get("id") or f"segment_{idx}"),
                        "end": _as_float(segment.get("end"), 0.0),
                        "speaker": str(segment.get("speaker") or chr(65 + (idx % 26))),
                        "start": _as_float(segment.get("start"), 0.0),
                        "text": str(segment.get("text") or ""),
                        "type": "transcript.text.segment",
                    }
                )
            result["segments"] = normalized_segments
    if usage:
        result["usage"] = usage
    return result


def _build_audio_response(
    data: dict[str, Any],
    task: str,
    response_format: str,
    requested_language: str | None = None,
) -> Response:
    normalized_format = response_format.strip().lower() if response_format else "json"
    text = extract_text(data).strip()
    usage = build_openai_usage(data)

    if normalized_format in {"text", "srt", "vtt"}:
        return Response(content=text, media_type="text/plain; charset=utf-8")

    if task == "transcribe" and normalized_format == "diarized_json":
        payload = _parse_json_object(text)
        return JSONResponse(content=_normalize_diarized(payload, text, usage))

    if normalized_format == "verbose_json":
        payload = _parse_json_object(text)
        if task == "translate":
            return JSONResponse(content=_normalize_translation_verbose(payload, text))
        return JSONResponse(
            content=_normalize_transcription_verbose(payload, text, requested_language)
        )

    body: dict[str, Any] = {"text": text}
    if task == "transcribe" and usage:
        body["usage"] = usage
    return JSONResponse(content=body)


async def _call_upstream(
    client: httpx.AsyncClient,
    body: dict[str, Any],
    headers: dict[str, str],
    rid: str,
    idx: int,
) -> tuple[dict[str, Any] | None, tuple[int, dict[str, Any]] | None]:
    """
    POST to OpenRouter chat/completions.
    Returns (success_json, None) or (None, (status_code, error_body)).
    """
    url = f"{OPENROUTER_URL}/chat/completions"
    try:
        resp = await client.post(url, json=body, headers=headers)
        if resp.status_code == 200:
            return resp.json(), None
        logger.warning(
            "rid=%s idx=%s upstream_status=%s body=%s",
            rid, idx, resp.status_code, resp.text[:500],
        )
        try:
            err_body = resp.json()
        except Exception:
            err_body = {
                "error": {"message": resp.text[:500], "type": "upstream_error"}
            }
        return None, (resp.status_code, err_body)
    except httpx.TimeoutException:
        logger.error("rid=%s idx=%s event=timeout", rid, idx)
        return None, (
            504,
            {"error": {"message": "Upstream request timed out", "type": "timeout_error"}},
        )
    except Exception as exc:
        logger.exception("rid=%s idx=%s event=exception", rid, idx)
        return None, (
            502,
            {
                "error": {
                    "message": f"Upstream request failed: {exc}",
                    "type": "proxy_error",
                }
            },
        )


async def _stream_error_payload(resp: httpx.Response) -> dict[str, Any]:
    raw = await resp.aread()
    text = raw.decode("utf-8", errors="ignore")
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"error": {"message": text[:500], "type": "upstream_error"}}


async def _iter_openrouter_audio_deltas(
    resp: httpx.Response,
    rid: str,
) -> Any:
    async for line in resp.aiter_lines():
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            logger.warning("rid=%s event=invalid_sse_chunk payload=%s", rid, payload[:200])
            continue
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        delta = choices[0].get("delta") or {}
        audio = delta.get("audio") or {}
        yield {
            "data": str(audio.get("data") or ""),
            "transcript": str(audio.get("transcript") or ""),
        }


async def _collect_speech_audio(
    body: dict[str, Any],
    headers: dict[str, str],
    rid: str,
) -> tuple[bytes | None, str, tuple[int, dict[str, Any]] | None]:
    url = f"{OPENROUTER_URL}/chat/completions"
    stream_headers = dict(headers)
    stream_headers["Accept"] = "text/event-stream"
    audio_chunks: list[str] = []
    transcript_chunks: list[str] = []

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=body, headers=stream_headers) as resp:
                if resp.status_code != 200:
                    err_body = await _stream_error_payload(resp)
                    logger.warning(
                        "rid=%s endpoint=speech upstream_status=%s body=%s",
                        rid, resp.status_code, json.dumps(err_body)[:500],
                    )
                    return None, "", (resp.status_code, err_body)

                async for delta in _iter_openrouter_audio_deltas(resp, rid):
                    if delta["data"]:
                        audio_chunks.append(delta["data"])
                    if delta["transcript"]:
                        transcript_chunks.append(delta["transcript"])
    except httpx.TimeoutException:
        logger.error("rid=%s endpoint=speech event=timeout", rid)
        return None, "", (
            504,
            {"error": {"message": "Upstream request timed out", "type": "timeout_error"}},
        )
    except Exception as exc:
        logger.exception("rid=%s endpoint=speech event=exception", rid)
        return None, "", (
            502,
            {
                "error": {
                    "message": f"Upstream request failed: {exc}",
                    "type": "proxy_error",
                }
            },
        )

    if not audio_chunks:
        return b"", "".join(transcript_chunks), None

    try:
        audio_bytes = base64.b64decode("".join(audio_chunks))
    except Exception as exc:
        logger.exception("rid=%s endpoint=speech event=decode_failure", rid)
        return None, "", (
            502,
            {
                "error": {
                    "message": f"Failed to decode upstream audio: {exc}",
                    "type": "proxy_error",
                }
            },
        )

    return audio_bytes, "".join(transcript_chunks), None


async def _proxy_speech_events(
    body: dict[str, Any],
    headers: dict[str, str],
    rid: str,
) -> Any:
    url = f"{OPENROUTER_URL}/chat/completions"
    stream_headers = dict(headers)
    stream_headers["Accept"] = "text/event-stream"

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=body, headers=stream_headers) as resp:
                if resp.status_code != 200:
                    err_body = await _stream_error_payload(resp)
                    yield f"data: {json.dumps({'error': err_body.get('error', err_body)})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for delta in _iter_openrouter_audio_deltas(resp, rid):
                    event: dict[str, str] = {}
                    if delta["data"]:
                        event["audio"] = delta["data"]
                    if delta["transcript"]:
                        event["transcript"] = delta["transcript"]
                    if event:
                        yield f"data: {json.dumps(event)}\n\n"
    except httpx.TimeoutException:
        yield (
            'data: {"error": {"message": "Upstream request timed out", '
            '"type": "timeout_error"}}\n\n'
        )
    except Exception as exc:
        logger.exception("rid=%s endpoint=speech_stream event=exception", rid)
        error_event = {
            "error": {
                "message": f"Upstream request failed: {exc}",
                "type": "proxy_error",
            }
        }
        yield f"data: {json.dumps(error_event)}\n\n"

    yield "data: [DONE]\n\n"


async def _call_audio_input_endpoint(request: Request, task: str) -> Response:
    rid = _request_id(request)
    content_type = request.headers.get("content-type", "")
    if "multipart" not in content_type:
        return error_response(
            415,
            "This endpoint expects multipart/form-data with a file field.",
            "unsupported_media_type",
        )

    form = await request.form()
    try:
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return error_response(400, "A multipart file field named 'file' is required.")

        model = str(form.get("model") or "").strip()
        if not model:
            return error_response(400, "A model is required.")

        raw = await upload.read()
        if not raw:
            return error_response(400, "The uploaded audio file is empty.")

        response_format = str(form.get("response_format") or "json").strip().lower()
        prompt = str(form.get("prompt") or "").strip() or None
        language = str(form.get("language") or "").strip() or None
        audio_format = _guess_audio_format(upload)
        temperature = form.get("temperature")
    finally:
        await form.close()

    b64_audio = base64.b64encode(raw).decode()

    instruction = _build_audio_instruction(task, response_format, prompt, language)
    or_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": audio_format,
                        },
                    },
                ],
            }
        ],
    }

    if temperature not in (None, ""):
        or_body["temperature"] = _as_float(temperature, 0.0)

    headers = upstream_headers(request)
    logger.info(
        "rid=%s endpoint=%s model=%s response_format=%s audio_format=%s bytes=%s",
        rid, task, model, response_format, audio_format, len(raw),
    )

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        success, error = await _call_upstream(client, or_body, headers, rid, 0)

    if error:
        status, err_body = error
        return JSONResponse(status_code=status, content=err_body)
    if success is None:
        return error_response(502, "No upstream response received.", "upstream_error")

    logger.info("rid=%s endpoint=%s event=success", rid, task)
    return _build_audio_response(success, task, response_format, language)


# ---- Routes ----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---- /images/generations ---------------------------------------------------


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def generations(request: Request) -> Response:
    rid = _request_id(request)
    body = await request.json()

    prompt = body.get("prompt", "")
    model = body.get("model", "")
    n, error = _parse_image_count(body.get("n"))
    if error:
        return error

    image_options, error = _parse_image_options(
        background=body.get("background"),
        moderation=body.get("moderation"),
        output_compression=body.get("output_compression"),
        output_format=body.get("output_format"),
        quality=body.get("quality"),
        response_format=body.get("response_format"),
        seed=body.get("seed"),
        size=body.get("size"),
        style=body.get("style"),
        user=body.get("user"),
    )
    if error:
        return error

    size = image_options["size"]
    quality = image_options["quality"]
    style = image_options["style"]
    background = image_options["background"]
    response_format = image_options["response_format"]

    full_prompt = _augment_prompt(prompt, style, background)
    image_config = build_image_config(
        size,
        quality,
        background=background,
        output_format=image_options["output_format"],
        output_compression=image_options["output_compression"],
        moderation=image_options["moderation"],
    )

    or_body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": full_prompt}],
        "modalities": _image_modalities(),
    }
    if image_config:
        or_body["image_config"] = image_config
    if image_options["seed"] is not None:
        or_body["seed"] = image_options["seed"]
    if image_options["user"]:
        or_body["user"] = image_options["user"]

    headers = upstream_headers(request)
    logger.info(
        "rid=%s endpoint=generations model=%s n=%s size=%s quality=%s response_format=%s",
        rid, model, n, size, quality, response_format,
    )

    all_images: list[dict[str, str]] = []
    all_usages: list[dict[str, Any]] = []
    last_error: tuple[int, dict[str, Any]] | None = None
    success_summaries: list[str] = []

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        tasks = [_call_upstream(client, or_body, headers, rid, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        for success, error in results:
            if success:
                extracted_images = extract_images(success, response_format=response_format)
                if extracted_images:
                    all_images.extend(extracted_images)
                else:
                    success_summaries.append(_summarize_image_response(success))
                usage = build_openai_image_usage(success)
                if usage:
                    all_usages.append(usage)
            elif error:
                last_error = error

    if not all_images:
        status, err_body = last_error or (
            502,
            {"error": {"message": "No images returned by upstream", "type": "upstream_error"}},
        )
        if success_summaries:
            logger.error(
                "rid=%s endpoint=generations error=no_images status=%s upstream_shapes=%s",
                rid,
                status,
                " || ".join(success_summaries[:3]),
            )
        else:
            logger.error("rid=%s endpoint=generations error=no_images status=%s", rid, status)
        return JSONResponse(status_code=status, content=err_body)

    logger.info("rid=%s endpoint=generations images_returned=%s", rid, len(all_images))
    return build_image_response(all_images, all_usages)


# ---- /images/edits ---------------------------------------------------------


@app.post("/v1/images/edits")
@app.post("/images/edits")
async def edits(request: Request) -> Response:
    rid = _request_id(request)
    content_type = request.headers.get("content-type", "")

    if "multipart" in content_type:
        form = await request.form()
        try:
            prompt = str(form.get("prompt") or "")
            model = str(form.get("model") or "")
            n, error = _parse_image_count(form.get("n"))
            if error:
                return error

            image_options, error = _parse_image_options(
                background=form.get("background"),
                input_fidelity=form.get("input_fidelity"),
                moderation=form.get("moderation"),
                output_compression=form.get("output_compression"),
                output_format=form.get("output_format"),
                quality=form.get("quality"),
                response_format=form.get("response_format"),
                seed=form.get("seed"),
                size=form.get("size"),
                user=form.get("user"),
            )
            if error:
                return error

            expected_file_fields = {"image", "image[]", "images", "images[]", "mask", "mask[]"}

            image_urls: list[str] = []
            for key, value in form.multi_items():
                if hasattr(value, "read") and key in expected_file_fields:
                    if key.startswith("mask") and not _mask_passthrough_enabled():
                        logger.warning(
                            "rid=%s endpoint=edits mask_behavior=rejected_unsupported",
                            rid,
                        )
                        return _mask_unsupported_response()
                    raw = await value.read()
                    if raw:
                        ct = getattr(value, "content_type", None) or "image/png"
                        b64 = base64.b64encode(raw).decode()
                        if key.startswith("mask"):
                            logger.warning(
                                "rid=%s endpoint=edits mask_behavior=best_effort message=%s",
                                rid,
                                "Mask inputs are forwarded as additional image context.",
                            )
                        image_urls.append(f"data:{ct};base64,{b64}")
        finally:
            await form.close()
    else:
        body = await request.json()
        prompt = body.get("prompt", "")
        model = body.get("model", "")
        n, error = _parse_image_count(body.get("n"))
        if error:
            return error

        image_options, error = _parse_image_options(
            background=body.get("background"),
            input_fidelity=body.get("input_fidelity"),
            moderation=body.get("moderation"),
            output_compression=body.get("output_compression"),
            output_format=body.get("output_format"),
            quality=body.get("quality"),
            response_format=body.get("response_format"),
            seed=body.get("seed"),
            size=body.get("size"),
            user=body.get("user"),
        )
        if error:
            return error

        image_urls = []
        for img in body.get("images", []):
            url = img.get("image_url")
            if url:
                image_urls.append(url)
        mask = body.get("mask")
        if mask:
            if not _mask_passthrough_enabled():
                logger.warning(
                    "rid=%s endpoint=edits mask_behavior=rejected_unsupported",
                    rid,
                )
                return _mask_unsupported_response()

            mask_url = mask.get("image_url") if isinstance(mask, dict) else None
            if mask_url:
                logger.warning(
                    "rid=%s endpoint=edits mask_behavior=best_effort message=%s",
                    rid,
                    "Mask inputs are forwarded as additional image context.",
                )
                image_urls.append(mask_url)

    size = image_options["size"]
    quality = image_options["quality"]
    background = image_options["background"]
    response_format = image_options["response_format"]

    prompt_text = _augment_prompt(prompt, background=background)
    content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    content_parts.extend(
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    )

    image_config = build_image_config(
        size,
        quality,
        background=background,
        output_format=image_options["output_format"],
        output_compression=image_options["output_compression"],
        moderation=image_options["moderation"],
        input_fidelity=image_options["input_fidelity"],
    )

    or_body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content_parts}],
        "modalities": _image_modalities(),
    }
    if image_config:
        or_body["image_config"] = image_config
    if image_options["seed"] is not None:
        or_body["seed"] = image_options["seed"]
    if image_options["user"]:
        or_body["user"] = image_options["user"]

    headers = upstream_headers(request)
    logger.info(
        "rid=%s endpoint=edits model=%s n=%s input_images=%s response_format=%s",
        rid, model, n, len(image_urls), response_format,
    )

    all_images: list[dict[str, str]] = []
    all_usages: list[dict[str, Any]] = []
    last_error: tuple[int, dict[str, Any]] | None = None
    success_summaries: list[str] = []

    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        tasks = [_call_upstream(client, or_body, headers, rid, i) for i in range(n)]
        results = await asyncio.gather(*tasks)
        for success, error in results:
            if success:
                extracted_images = extract_images(success, response_format=response_format)
                if extracted_images:
                    all_images.extend(extracted_images)
                else:
                    success_summaries.append(_summarize_image_response(success))
                usage = build_openai_image_usage(success)
                if usage:
                    all_usages.append(usage)
            elif error:
                last_error = error

    if not all_images:
        status, err_body = last_error or (
            502,
            {"error": {"message": "No images returned by upstream", "type": "upstream_error"}},
        )
        if success_summaries:
            logger.error(
                "rid=%s endpoint=edits error=no_images status=%s upstream_shapes=%s",
                rid,
                status,
                " || ".join(success_summaries[:3]),
            )
        else:
            logger.error("rid=%s endpoint=edits error=no_images status=%s", rid, status)
        return JSONResponse(status_code=status, content=err_body)

    logger.info("rid=%s endpoint=edits images_returned=%s", rid, len(all_images))
    return build_image_response(all_images, all_usages)


# ---- /audio/transcriptions -------------------------------------------------


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def audio_transcriptions(request: Request) -> Response:
    return await _call_audio_input_endpoint(request, "transcribe")


# ---- /audio/translations ---------------------------------------------------


@app.post("/v1/audio/translations")
@app.post("/audio/translations")
async def audio_translations(request: Request) -> Response:
    return await _call_audio_input_endpoint(request, "translate")


# ---- /audio/speech ---------------------------------------------------------


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def audio_speech(request: Request) -> Response:
    rid = _request_id(request)
    body = await request.json()

    input_text = str(body.get("input") or "").strip()
    model = str(body.get("model") or "").strip()
    if not input_text:
        return error_response(400, "An input string is required.")
    if not model:
        return error_response(400, "A model is required.")

    response_format = str(body.get("response_format") or "mp3").strip().lower()
    voice = _speech_voice_id(body.get("voice"))
    instructions = str(body.get("instructions") or "").strip() or None
    stream_format = str(body.get("stream_format") or "audio").strip().lower()
    speed = _as_float(body.get("speed"), 1.0)

    system_parts = [
        "You are a text-to-speech engine.",
        "Generate spoken audio for the user's text only.",
        "Do not add commentary, role-play, or extra words.",
    ]
    if instructions:
        system_parts.append(f"Voice and delivery guidance: {instructions}")
    if abs(speed - 1.0) > 1e-9:
        system_parts.append(f"Target speaking speed: {speed}x compared to a natural baseline.")

    or_body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": " ".join(system_parts)},
            {"role": "user", "content": input_text},
        ],
        "modalities": ["text", "audio"],
        "audio": {
            "voice": voice,
            "format": _speech_upstream_format(response_format),
        },
        "stream": True,
    }

    headers = upstream_headers(request)
    logger.info(
        "rid=%s endpoint=speech model=%s voice=%s response_format=%s stream_format=%s",
        rid, model, voice, response_format, stream_format,
    )

    if stream_format == "sse":
        return StreamingResponse(
            _proxy_speech_events(or_body, headers, rid),
            media_type="text/event-stream",
        )

    audio_bytes, _transcript, error = await _collect_speech_audio(or_body, headers, rid)
    if error:
        status, err_body = error
        return JSONResponse(status_code=status, content=err_body)
    if not audio_bytes:
        return error_response(502, "No audio returned by upstream", "upstream_error")

    return Response(
        content=audio_bytes,
        media_type=_speech_media_type(response_format),
    )
