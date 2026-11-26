"""Tigo (.tg): Minimal language & durable runtime for safe, typed AI agents."""
from .checker import check_program
from .runtime import ApprovalPaused, ApprovalRejected, DurableStore, Runtime, validate_type
from .syntax import Program, TigoError, compile_file, compile_source

__all__ = [
    "ApprovalPaused",
    "ApprovalRejected",
    "DurableStore",
    "Program",
    "Runtime",
    "TigoError",
    "check_program",
    "compile_file",
    "compile_source",
    "validate_type",
]
