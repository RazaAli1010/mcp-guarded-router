"""mcpr - a fine-tuned MCP tool router with a three-layer guardrail.

This package is pure Python by contract: nothing under ``src/mcpr/`` may import a package
from the forbidden list in SPEC.md 2.4 (torch, transformers, peft, ...). That constraint is
what lets the Kaggle notebooks ``pip install -e`` this very package and reuse the identical
prompt-building and guardrail code, making train/eval/production prompts identical by
construction rather than by discipline.
"""

__version__ = "0.1.0"
