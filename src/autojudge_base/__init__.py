"""
AutoJudge Base - Core infrastructure for implementing TREC AutoJudge systems.

This package provides:
- Protocols for implementing auto-judges (AutoJudge, LeaderboardJudgeProtocol, etc.)
- Data models (Report, Request, Document)
- Output containers (Leaderboard, Qrels, NuggetBanks)
- Workflow orchestration (judge_runner, workflow.yml support)
- CLI utilities (click_plus, auto_judge_to_click_command)
"""

from typing import Iterable, Protocol, Sequence, Optional, Type

# Core data models
from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests_from_file

# Output containers
from .leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    MeasureSpec,
    LeaderboardSpec,
    LeaderboardBuilder,
    LeaderboardVerification,
    LeaderboardVerificationError,
)
from .qrels import (
    QrelsSpec,
    QrelRow,
    Qrels,
    build_qrels,
    QrelsVerification,
    QrelsVerificationError,
    write_qrel_file,
    doc_id_md5,
)

# Nugget data
from .nugget_data import NuggetBanks, NuggetBanksProtocol
from .nugget_doc_models import NuggetDocEntry, TopicNuggetDocs, write_nugget_docs_collaborator

# LLM configuration
from .llm_config import LlmConfigProtocol, LlmConfigBase, load_llm_config

# Utilities
from .utils import format_preview

__version__ = '0.1.0'


# === The interface for AutoJudges to implement ====
#
# Three separate protocols allow mixing implementations:
#   - LeaderboardJudgeProtocol: produces leaderboard scores
#   - QrelsCreatorProtocol: creates relevance judgments
#   - NuggetCreatorProtocol: creates nugget banks
#
# A single class can implement all three (common case), or different
# classes can be used for each phase via workflow.yml configuration.


class LeaderboardJudgeProtocol(Protocol):
    """Protocol for leaderboard generation."""

    def judge(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        **kwargs
    ) -> "Leaderboard":
        """
        Judge RAG responses against topics and produce a leaderboard.

        Args:
            rag_responses: RAG system outputs to judge
            rag_topics: Evaluation topics/queries
            llm_config: LLM configuration
            nugget_banks: Optional nuggets for judgment
            qrels: Optional qrels from create_qrels() phase
            **kwargs: Additional settings

        Returns:
            Leaderboard with rankings/scores for runs
        """
        ...


class QrelsCreatorProtocol(Protocol):
    """Protocol for qrels creation."""

    def create_qrels(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional["Qrels"]:
        """
        Create relevance judgments (qrels) for RAG responses.

        Args:
            rag_responses: RAG system outputs to judge
            rag_topics: Evaluation topics/queries
            llm_config: LLM configuration for qrels generation
            nugget_banks: Optional nuggets to use for judgment
            **kwargs: Additional settings (e.g., grade_range=[0, 3])

        Returns:
            Qrels with relevance judgments, or None if not supported
        """
        ...


class NuggetCreatorProtocol(Protocol):
    """Protocol for nugget creation."""

    nugget_banks_type: Type[NuggetBanksProtocol]
    """The NuggetBanks container type this creator produces."""

    def create_nuggets(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[NuggetBanksProtocol]:
        """
        Create or refine nugget banks based on RAG responses.

        Args:
            rag_responses: RAG system outputs to analyze for nugget creation/refinement
            rag_topics: Evaluation topics/queries
            llm_config: LLM configuration for nugget generation
            nugget_banks: Optional existing nuggets to refine/extend

        Returns:
            NuggetBanks container, or None if not supported
        """
        ...


class AutoJudge(LeaderboardJudgeProtocol, QrelsCreatorProtocol, NuggetCreatorProtocol, Protocol):
    """
    Combined protocol for judges that implement all three phases.

    This is a convenience protocol for the common case where a single class
    handles nugget creation, qrels creation, and leaderboard generation.

    For modular configurations, use the individual protocols:
    - LeaderboardJudgeProtocol
    - QrelsCreatorProtocol
    - NuggetCreatorProtocol
    """
    pass


# CLI utilities
from .click_plus import option_rag_responses, option_rag_topics, option_ir_dataset, auto_judge_to_click_command

__all__ = [
    # Version
    "__version__",
    # Core data models
    "Report",
    "load_report",
    "Request",
    "load_requests_from_irds",
    "load_requests_from_file",
    # Output containers
    "Leaderboard",
    "LeaderboardEntry",
    "MeasureSpec",
    "LeaderboardSpec",
    "LeaderboardBuilder",
    "LeaderboardVerification",
    "LeaderboardVerificationError",
    "Qrels",
    "QrelsSpec",
    "QrelRow",
    "build_qrels",
    "QrelsVerification",
    "QrelsVerificationError",
    "write_qrel_file",
    "doc_id_md5",
    # Nugget data
    "NuggetBanks",
    "NuggetBanksProtocol",
    "NuggetDocEntry",
    "TopicNuggetDocs",
    "write_nugget_docs_collaborator",
    # LLM configuration
    "LlmConfigProtocol",
    "LlmConfigBase",
    "load_llm_config",
    # Protocols
    "LeaderboardJudgeProtocol",
    "QrelsCreatorProtocol",
    "NuggetCreatorProtocol",
    "AutoJudge",
    # CLI utilities
    "option_rag_responses",
    "option_rag_topics",
    "option_ir_dataset",
    "auto_judge_to_click_command",
    # Utilities
    "format_preview",
]
