"""Document models and helpers.

Regular package (not a namespace package) so ``setuptools`` bundles it into
the wheel and static tooling (griffe/mkdocstrings) can resolve the submodule.
"""

from .document import Document

__all__ = ["Document"]
