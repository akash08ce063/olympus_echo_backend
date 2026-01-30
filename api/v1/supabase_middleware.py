from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from data_layer.supabase_client import get_supabase_client
from telemetrics.logger import logger


class SupabaseAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate Bearer JWTs with Supabase and attach user to request.state.user.

    - Skips OPTIONS and listed excluded paths (docs, openapi, health, etc.)
    - Uses Supabase's auth.get_user(token) to validate access tokens
    - Sets `request.state.user` to the returned user object on success
    """

    def __init__(self, app):
        super().__init__(app)
        self.excluded_paths = [
            "/docs",
            "/openapi.json",
            "/health",
            # Add any other endpoints that should bypass Supabase auth here
        ]

    async def dispatch(self, request: Request, call_next):
        # Skip authentication for OPTIONS requests
        if request.method == "OPTIONS":
            logger.debug(f"✅ SupabaseAuth: Skipping auth for OPTIONS request to {request.url.path}")
            return await call_next(request)

        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            logger.debug(f"✅ SupabaseAuth: Skipping auth for excluded path {request.url.path}")
            return await call_next(request)

        # If already authenticated by another middleware, continue
        if hasattr(request.state, "user") and request.state.user:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")

        if not auth_header or not auth_header.startswith("Bearer "):
            logger.debug(f"❌ SupabaseAuth: Missing or invalid Authorization header for {request.url.path}")
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})

        token = auth_header.replace("Bearer ", "").strip()

        try:
            supabase_client = await get_supabase_client()
            # The underlying client exposes auth.get_user
            user_response = await supabase_client.get_client().auth.get_user(token)
            user = getattr(user_response, "user", None)

            if not user:
                logger.warning(f"❌ SupabaseAuth: Invalid or expired token for path {request.url.path}")
                return JSONResponse(status_code=401, content={"detail": "Invalid or expired token"})

            # Attach user object to request.state
            request.state.user = user
            logger.debug(f"✅ SupabaseAuth: Authenticated user {getattr(user, 'id', '<no-id>')} for {request.url.path}")

        except Exception as e:
            logger.exception(f"❌ SupabaseAuth: Authentication failed: {e}")
            return JSONResponse(status_code=401, content={"detail": f"Authentication failed: {e}"})

        return await call_next(request)
