from contextvars import ContextVar


strict_llm_errors = ContextVar("strict_llm_errors", default=False)


class LLMServiceError(RuntimeError):
    """Safe user-facing error; does not contain provider responses or secrets."""


class LLMOutputError(LLMServiceError):
    """A completed stream without a usable, complete answer."""

    def __init__(self, message, reason):
        super().__init__(message)
        self.reason = reason
