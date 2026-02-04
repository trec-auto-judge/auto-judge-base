# Participant-requested LLM Model Configuration

This guide explains how to request specific LLM models for your AutoJudge implementation.

## Two Config Formats

### 1. Submission Mode (`llm-config.yml`)

For TIRA submission - declares model preferences resolved against organizer's pool:

```yaml
model_preferences:
  - "llama-3.3-70b-instruct"
  - "gpt-4o"
  - "claude-3-5-sonnet"
on_no_match: "use_default"
```

### 2. Dev Mode (`llm-config.dev.yml`)

For local development - uses your endpoint directly (no resolution):

```yaml
base_url: "http://localhost:11434/v1"
model: "llama3.2"
api_key: ""  # optional
```

## Usage

```bash
# Development: use your local endpoint
./my-judge.py --llm-config llm-config.dev.yml --rag-responses runs/ --output output.txt

# Submission: resolve against organizer's available models
./my-judge.py --llm-config llm-config.yml --rag-responses runs/ --output output.txt
```

## How It Works

1. You declare an **ordered list** of preferred models in `llm-config.yml`
2. At runtime, the system checks which models are available from the organizer's pool
3. The **first available** model from your list is selected
4. Your judge receives a ready-to-use `MinimaLlmConfig` with the resolved model

## Configuration Format

### Basic Configuration

```yaml
model_preferences:
  - "gpt-4o"           # First choice
  - "gpt-4-turbo"      # Fallback if gpt-4o unavailable
  - "llama-3.1-70b"    # Second fallback
```

### With Fallback Behavior

```yaml
model_preferences:
  - "gpt-4o"
  - "claude-3-5-sonnet"

# What happens if none of your preferences are available?
# Options: "use_default" (default) or "error"
on_no_match: "use_default"
```

- `on_no_match: "use_default"` (default) - Fall back to the organizer's default model
- `on_no_match: "error"` - Fail with an error message listing available models

## Available Models

The available models are controlled by the evaluation organizers. Common models include:

| Model Name | Description |
|------------|-------------|
| `gpt-4o` | OpenAI GPT-4o |
| `gpt-4o-mini` | OpenAI GPT-4o Mini (faster, cheaper) |
| `llama-3.1-70b` | Meta Llama 3.1 70B |

**Note:** Actual availability depends on the evaluation environment. If your preferred model is unavailable, the system will try your fallback choices.

## Using the Resolved Config in Your Judge

Your judge receives the resolved `MinimaLlmConfig` as the third parameter:

```python
from trec_auto_judge import AutoJudge
from trec_auto_judge.llm import MinimaLlmConfig, OpenAIMinimaLlm

class MyJudge(AutoJudge):
    def judge(self, rag_responses, rag_topics, llm_config: MinimaLlmConfig):
        # Create LLM backend from resolved config
        llm = OpenAIMinimaLlm(llm_config)

        # Use llm for your judging logic...
        # llm_config.model contains the resolved model name
        # llm_config.base_url contains the endpoint URL
```

## Fallback: Environment Variables

If no `--llm-config` is provided, the system falls back to environment variables:

- `OPENAI_BASE_URL` - LLM endpoint URL
- `OPENAI_MODEL` - Model identifier
- `OPENAI_API_KEY` - API key/token

This ensures backwards compatibility with existing judges.

## Example Directory Structure

```
my-judge/
├── my-judge.py           # Your judge implementation
├── llm-config.yml        # Submission: model preferences
├── llm-config.dev.yml    # Dev: your local endpoint
├── requirements.txt
└── Dockerfile
```

## Debugging

Use the CLI to check available models and test resolution:

```bash
# List available models
trec-auto-judge list-models

# Test resolution against your config
trec-auto-judge list-models --resolve llm-config.yml
```

## Troubleshooting

### "No model available from preferences"

Your preferred models are not in the available pool. Either:
1. Add more fallback options to your `model_preferences` list
2. Remove `on_no_match: "error"` to use organizer's default (the default behavior)
3. Check the error message for the list of available models

### Config not being read

Ensure you're passing the `--llm-config` flag:
```bash
./my-judge.py --llm-config llm-config.yml ...
```

### YAML parsing errors

Verify your YAML syntax. Common issues:
- Use spaces, not tabs
- List items need `- ` prefix with space
- Strings with special characters need quotes