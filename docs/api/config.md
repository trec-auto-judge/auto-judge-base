# Config & protocol

An AutoJudge implements the [`AutoJudge`](#autojudge_base.AutoJudge) protocol (`judge`, `create_nuggets`, `create_qrels`) and receives its LLM endpoint through the injected [`LlmConfigBase`](#autojudge_base.llm_config.LlmConfigBase). The framework builds the config from environment variables (`OPENAI_BASE_URL`/`OPENAI_MODEL`/`OPENAI_API_KEY`/`CACHE_DIR`) — see [Configure your LLM endpoint](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/02-configure-llm-endpoint.md).

## AutoJudge protocol

::: autojudge_base.AutoJudge

## LLM config

::: autojudge_base.llm_config.LlmConfigBase

::: autojudge_base.llm_config.LlmConfigProtocol
