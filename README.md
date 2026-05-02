# OpenRouter Media Proxy

A minimal, transparent FastAPI proxy that lets **Open WebUI** or any other
client expecting parts of the **OpenAI Images API** and **OpenAI Audio API**
talk to **OpenRouter** models exposed through `/chat/completions`.

## Motivation

Open WebUI expects OpenAI-style endpoints such as:

- `/v1/images/generations`
- `/v1/images/edits`
- `/v1/audio/transcriptions`
- `/v1/audio/translations`
- `/v1/audio/speech`

OpenRouter exposes both image generation and audio input/output through
`/api/v1/chat/completions`, using multimodal `messages`, `input_audio`, image
parts, and streamed audio deltas.

This proxy sits in between and translates on the fly:

```text
Open WebUI / OpenAI client
    -> /v1/images/*, /v1/audio/*
    -> Proxy
    -> OpenRouter /chat/completions
```

## Supported Endpoints

### Images

- `POST /v1/images/generations`
- `POST /v1/images/edits`

Image requests are translated into OpenRouter `modalities: ["image", "text"]`
style chat requests. Returned base64 data URLs are converted back into OpenAI
style image responses.

Current image response behavior:

- Default response shape remains `{ created, data: [{ b64_json, revised_prompt? }] }`
  for compatibility with existing Open WebUI usage.
- If `response_format: "url"` is explicitly requested, the proxy returns the
  upstream OpenRouter `data:` URL in the OpenAI-style `url` field.
- If upstream usage data is present, the proxy includes an OpenAI-style image
  `usage` object in the response.

### Audio

- `POST /v1/audio/transcriptions`
- `POST /v1/audio/translations`
- `POST /v1/audio/speech`

Audio transcription and translation requests accept multipart uploads,
base64-encode the file, and send it to OpenRouter as an `input_audio` content
part. Speech requests translate OpenAI TTS-style JSON into OpenRouter audio
output requests and collect the upstream SSE audio chunks back into a binary
audio response.

Bare `/images/*` and `/audio/*` routes are also available for convenience.

## Request Mapping

### Images

| OpenAI-style input | OpenRouter request |
|--------------------|-------------------|
| `prompt` | `messages[0].content` |
| `model` | Passed through unchanged |
| `size` | mapped to `image_config.aspect_ratio` |
| `quality` | mapped to `image_config.image_size` |
| `style` / `background` | Prompt hints |
| `n` | Parallel upstream requests |
| uploaded images | inline `image_url` parts |
| `mask` | rejected by default; optional best-effort passthrough |
| `user` / `seed` | Passed through unchanged |
| `output_format`, `output_compression`, `moderation`, `input_fidelity` | best-effort `image_config` pass-through |

Image request validation:

- Invalid enum-style values for `size`, `quality`, `style`, `background`,
  `response_format`, `output_format`, `moderation`, and `input_fidelity`
  return OpenAI-style `400 invalid_request_error` responses.
- Invalid integer values for `n`, `seed`, or `output_compression` also return
  `400 invalid_request_error` responses.
- `mask` inputs on image edits are rejected by default, because OpenRouter's
  documented chat-completions image flow does not expose a native mask
  primitive.

### Audio Input

| OpenAI-style input | OpenRouter request |
|--------------------|-------------------|
| multipart `file` | `input_audio.data` base64 |
| file type / extension | `input_audio.format` |
| `model` | Passed through unchanged |
| `prompt` / `language` | Instruction text |
| `response_format` | Instruction shaping + response normalization |
| `temperature` | Passed through when present |

### Audio Output

| OpenAI-style input | OpenRouter request |
|--------------------|-------------------|
| `input` | user message content |
| `model` | Passed through unchanged |
| `voice` | `audio.voice` |
| `response_format` | `audio.format` |
| `instructions` / `speed` | system prompt hints |
| `stream_format: "audio"` | proxy collects SSE and returns audio bytes |
| `stream_format: "sse"` | proxy returns a simple SSE stream of audio chunks |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base URL. |
| `DEFAULT_IMAGE_MODALITIES` | `image` | Modalities for image-capable models. Set `image,text` for models that require both. |
| `IMAGE_EDIT_MASK_MODE` | `reject` | `reject` returns 400 for edit masks. `passthrough` keeps the old best-effort behavior and forwards masks as extra image context. |
| `UPSTREAM_TIMEOUT` | `120` | Request timeout in seconds. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

## Quick Start

1. Start the service:

   ```bash
   docker compose up -d openrouter-media-proxy
   ```

2. Point your client at:

   ```text
   http://openrouter-media-proxy:8080/v1
   ```

3. Use an OpenRouter API key in the normal OpenAI `Authorization: Bearer ...`
   header.

4. Configure model IDs in the client as **OpenRouter model IDs**, not native
   OpenAI IDs.

## Usage Examples

### Generate an image

```bash
curl -s http://openrouter-media-proxy:8080/v1/images/generations \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cozy cabin in the mountains at sunset",
    "model": "google/gemini-2.5-flash-image-preview",
    "size": "1792x1024",
    "quality": "hd",
    "n": 1
  }'
```

### Transcribe audio

```bash
curl -s http://openrouter-media-proxy:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -F "file=@sample.wav" \
  -F "model=google/gemini-2.5-flash" \
  -F "response_format=json"
```

### Translate audio into English

```bash
curl -s http://openrouter-media-proxy:8080/v1/audio/translations \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -F "file=@sample.mp3" \
  -F "model=google/gemini-2.5-flash" \
  -F "response_format=verbose_json"
```

### Generate speech

```bash
curl -s http://openrouter-media-proxy:8080/v1/audio/speech \
  -H "Authorization: Bearer sk-or-v1-YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello from OpenRouter through an OpenAI-compatible proxy.",
    "model": "openai/gpt-4o-audio-preview",
    "voice": "alloy",
    "response_format": "wav"
  }' > speech.wav
```

## Limitations

- Models are passed through unchanged. The client must use OpenRouter model IDs
  that actually support the requested modality.
- Image `response_format` is only partially compatible:
  omitted `response_format` still defaults to `b64_json` in this proxy,
  and explicit `response_format: "url"` returns an upstream `data:` URL rather
  than a temporary externally hosted URL.
- `n > 1` for images still means multiple parallel upstream requests and costs
  `n` times the credits.
- Image `size` remains an approximation, because OpenRouter image generation is
  controlled through aspect ratio and provider-specific size buckets rather than
  guaranteed OpenAI pixel-exact output dimensions.
- Image `quality` is also approximate. OpenAI quality labels are mapped to
  OpenRouter image-size buckets, then provider-specific image controls are
  passed through on a best-effort basis.
- Image fields such as `output_format`, `output_compression`, `moderation`,
  and edit `input_fidelity` are forwarded on a best-effort basis. Whether they
  affect the final image depends on the selected OpenRouter model/provider.
- Audio transcription and translation are prompt-shaped onto chat completions.
  Structured responses such as `verbose_json` and `diarized_json` are best
  effort and depend on model behavior.
- Audio transcription and translation streaming is not implemented. The proxy
  always returns a final response body.
- `stream_format: "sse"` for speech is supported only as a simple SSE stream of
  `{ audio, transcript? }` chunks, not a guaranteed byte-for-byte OpenAI event
  schema.
- `instructions` and `speed` for speech are translated into prompt guidance,
  because OpenRouter exposes only `voice` and `format` as structured audio
  output controls in the documented flow.
- Masks on image edits are rejected by default. Set `IMAGE_EDIT_MASK_MODE=passthrough`
  only if you explicitly want the old best-effort fallback, where masks are
  forwarded as additional image context rather than a true native OpenAI mask
  primitive.

[oai-images]: https://platform.openai.com/docs/api-reference/images
