# Leaderboard

A judge's `judge()` returns a [`Leaderboard`](#autojudge_base.leaderboard.Leaderboard). Declare its schema once with `LeaderboardSpec` (one `MeasureSpec` per measure), assemble rows with `LeaderboardBuilder`, and let `build(expected_topic_ids=..., on_missing=...)` verify coverage before the table is trusted.

## Leaderboard and entries

::: autojudge_base.leaderboard.Leaderboard

::: autojudge_base.leaderboard.LeaderboardEntry

::: autojudge_base.leaderboard.LeaderboardFormat

## Specs and builder

::: autojudge_base.leaderboard.MeasureSpec

::: autojudge_base.leaderboard.LeaderboardSpec

::: autojudge_base.leaderboard.LeaderboardBuilder

## Verification

::: autojudge_base.leaderboard.LeaderboardVerification

::: autojudge_base.leaderboard.LeaderboardVerificationError
