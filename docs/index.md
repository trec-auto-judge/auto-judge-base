# auto-judge-base API reference

This site is the generated API reference for **auto-judge-base**, the core library behind [TREC AutoJudge](https://github.com/trec-auto-judge) — protocols, data models, the workflow runner, and the leaderboard/qrels/nugget builders. Every page is generated from the source, so it tracks the installed version rather than a hand-maintained copy.

For task instructions — setting up an environment, developing a judge, running workflows, and submitting to TIRA — see the [Participant HowTo](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/README.md), which links back into these reference pages for field-level detail. The [TREC AutoJudge task site](https://trec-auto-judge.cs.unh.edu/) covers the shared task itself (schedule, data, participation), and the [GitHub organization](https://github.com/trec-auto-judge) hosts the code and developer documentation.

## Reference pages

- [Data models](api/data-models.md) — `Report`, `Request`, `Document`, and the citation-carrying sentence types a judge reads.
- [Nuggets](api/nuggets.md) — `NuggetBanks` and the per-topic nugget containers a judge produces.
- [Leaderboard](api/leaderboard.md) — `LeaderboardSpec`/`LeaderboardBuilder` and the verified result table.
- [Qrels](api/qrels.md) — `QrelsSpec`/`build_qrels` and TREC-format relevance judgments.
- [Config & protocol](api/config.md) — the `AutoJudge` protocol methods and the injected `LlmConfigBase`.
