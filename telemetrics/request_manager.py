import contextvars
import uuid

# Context variable to hold the request ID
_request_id_ctx_var = contextvars.ContextVar("request_id", default=None)

class RequestIdManager:
    @staticmethod
    def set(request_id: str = None):
        """Set a request ID in the context. Generate a new one if not provided."""
        if request_id is None:
            request_id = str(uuid.uuid4())
        _request_id_ctx_var.set(request_id)

    @staticmethod
    def get() -> str:
        """Get the current request ID from context."""
        return _request_id_ctx_var.get()

    @staticmethod
    def clear():
        """Clear the request ID from context."""
        _request_id_ctx_var.set(None)
