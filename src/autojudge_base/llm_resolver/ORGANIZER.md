# Organizer Guide: Available Models Configuration

This guide explains how to configure the model pool that participants can request.

## How It Works: End-to-End Flow

### 1. Organizer sets up `available_models.yml`

Location: `AUTOJUDGE_AVAILABLE_MODELS` env var or `~/.autojudge/available_models.yml`

### 2. Participant writes `llm-config.yml`

```yaml
model_preferences:
  - "llama-3.3-70b-instruct"  # First choice
  - "gpt-4o"                   # Fallback
on_no_match: "use_default"
```

### 3. Judge is invoked with `--llm-config`

```bash
./my-judge.py --llm-config llm-config.yml --rag-responses runs/ --output out.txt
```

### 4. Resolution at startup

In `click_plus.py:_resolve_llm_config()`:
- Load participant preferences from `llm-config.yml`
- Load organizer's available models via `ModelResolver.from_env()`
- Match first available preference (resolving aliases)
- Return ready-to-use `MinimaLlmConfig`

### 5. Judge receives resolved config

```python
# The judge's judge() method receives:
judge.judge(rag_responses, rag_topics, llm_config)
# Where llm_config contains:
#   - base_url: "http://gpu-server:3001/v1"
#   - model: "llama-3.3-70b-instruct-tp4"
#   - api_key: (resolved from env var)
```

### Resolution Logic

1. Loop through participant's `model_preferences` in order
2. Resolve aliases (e.g., `llama-3.3-70b-instruct` → `llama-3.3-70b-instruct-tp`)
3. Check if resolved name exists in organizer's enabled models
4. Return first match as `MinimaLlmConfig`
5. If no match:
   - `on_no_match: "use_default"` (default) → Use organizer's `default_model`
   - `on_no_match: "error"` → Raise `ModelResolutionError`

## Quick Start

Create `~/.autojudge/available_models.yml`:

```yaml
models:
  llama-3.3-70b-instruct:
    base_url: "http://localhost:3001/v1"
    model_id: "llama-3.3-70b-instruct"
    api_key_env: ""
    enabled: true

default_model: "llama-3.3-70b-instruct"
```

## Configuration File Location

The framework searches for `available_models.yml` in this order:

1. **`AUTOJUDGE_AVAILABLE_MODELS` env var** - Explicit path to config file
   ```bash
   export AUTOJUDGE_AVAILABLE_MODELS=/path/to/available_models.yml
   ```

2. **`~/.autojudge/available_models.yml`** - Default user location

3. **Legacy env vars** - Falls back to `OPENAI_MODEL`/`OPENAI_BASE_URL` for backwards compatibility

## Configuration Format

```yaml
models:
  # Each key is the canonical model name participants can request
  model-name:
    base_url: "https://api.example.com/v1"  # OpenAI-compatible endpoint
    model_id: "actual-model-identifier"      # Model ID sent to API
    api_key_env: "ENV_VAR_NAME"              # Env var containing API key
    enabled: true                            # Set false to disable

# REQUIRED: Default model when participant preferences don't match
# Must reference an enabled model
default_model: "model-name"

# Aliases let participants use common names
aliases:
  "gpt-4": "gpt-4-turbo"
  "llama": "llama-3.3-70b-instruct"
```

## Validation

The organizer config is validated on load:
- `default_model` must be set (required)
- `default_model` must reference an enabled model

This guarantees that participant submissions always resolve to a valid model.

## Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `base_url` | Yes | OpenAI-compatible API endpoint (no trailing slash) |
| `model_id` | Yes | Model identifier sent in API requests |
| `api_key_env` | No | Environment variable name for API key (default: `OPENAI_API_KEY`) |
| `enabled` | No | Whether model is available (default: `true`) |

## TIRA Integration

For TIRA deployments, set the env var to point to the runtime config:

```bash
# In TIRA container environment
export AUTOJUDGE_AVAILABLE_MODELS=/tira/config/available_models.yml
```

TIRA overrides the example config with actual endpoints at runtime.

## Resolution Behavior

When a participant's `llm-config.yml` is processed:

1. Each preference is checked in order against enabled models
2. Aliases are resolved to canonical names
3. First matching model is used
4. If no match:
   - `on_no_match: "use_default"` (default) → Uses `default_model`
   - `on_no_match: "error"` → Raises `ModelResolutionError`

## Example: Multi-Provider Setup

```yaml
models:
  # Local inference (fastest, no cost)
  llama-3.3-70b-instruct-tp:
    base_url: "http://gpu-server.local:3001/v1"
    model_id: "llama-3.3-70b-instruct-tp4"
    api_key_env: ""
    enabled: true

  # OpenAI (fallback)
  gpt-4o:
    base_url: "https://api.openai.com/v1"
    model_id: "gpt-4o-2024-08-06"
    api_key_env: "OPENAI_API_KEY"
    enabled: true

  gpt-4o-mini:
    base_url: "https://api.openai.com/v1"
    model_id: "gpt-4o-mini"
    api_key_env: "OPENAI_API_KEY"
    enabled: true

  # Anthropic via OpenRouter
  claude-3-5-sonnet:
    base_url: "https://openrouter.ai/api/v1"
    model_id: "anthropic/claude-3.5-sonnet"
    api_key_env: "OPENROUTER_API_KEY"
    enabled: true

default_model: "llama-3.3-70b-instruct-tp"

aliases:
  "llama": "llama-3.3-70b-instruct-tp"
  "gpt-4": "gpt-4o"
  "claude": "claude-3-5-sonnet"
```

## Debugging

### CLI

```bash
# List available models from current config
trec-auto-judge list-models

# List from specific config file
trec-auto-judge list-models --config /path/to/available_models.yml

# Test resolution against a participant's config
trec-auto-judge list-models --resolve participant/llm-config.yml
```

### Programmatic

```python
from trec_auto_judge.llm_resolver import AvailableModels

available = AvailableModels.from_env()
print("Enabled:", available.get_enabled_models())
print("Default:", available.default_model)
print("Aliases:", available.aliases)
print("Disabled:", available.disabled)
```