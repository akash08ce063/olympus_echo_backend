import inspect
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from telemetrics.request_manager import RequestIdManager


class RichLogger:
    """Rich text logger with clean, colorful output."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Create console for rich output
        self.console = Console() if RICH_AVAILABLE else None

        # Rich handler for console output
        if RICH_AVAILABLE:
            rich_handler = RichHandler(
                console=self.console,
                show_time=False,  # We format time ourselves
                show_level=True,  # Keep RichHandler's level display
                show_path=True,  # We'll override pathname/lineno in formatter
                enable_link_path=False,
                markup=True,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
            )
            rich_handler.setLevel(level)
            rich_handler.setFormatter(self._get_rich_formatter())
            self.logger.addHandler(rich_handler)
        else:
            # Fallback to basic handler if rich not available
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(self._get_basic_formatter())
            self.logger.addHandler(console_handler)

        # Also add file handler for persistent logging
        self._setup_file_handler(level)

    def _setup_file_handler(self, level: int):
        """Setup file handler for persistent logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(self._get_file_formatter())
        self.logger.addHandler(file_handler)

    def _get_rich_formatter(self):
        """Rich formatter for console output."""

        class RichFormatter(logging.Formatter):
            def __init__(self, logger_instance):
                super().__init__()
                self.logger_instance = logger_instance

            def format(self, record):
                # Add custom fields
                record.request_id = getattr(record, "request_id", RequestIdManager.get())
                record.process_id = os.getpid()
                record.thread_id = threading.get_ident()
                record.tag = getattr(record, "tag", None)

                # Use caller context from extra (already set in _prepare_log_message)
                # Fallback to _get_caller_context if not present (for direct logger calls)
                if not hasattr(record, "caller_module"):
                    context = self.logger_instance._get_caller_context()
                    record.caller_module = context["module"]
                    record.caller_funcName = context["funcName"]
                    record.caller_lineno = context["lineno"]
                    record.caller_filename = context["filename"]

                # Override pathname and lineno so RichHandler shows the actual caller location
                if hasattr(record, "caller_filename") and hasattr(record, "caller_lineno"):
                    # Extract just the filename from the full path for cleaner display
                    caller_filename = getattr(record, "caller_filename", "")
                    if caller_filename and caller_filename != "unknown":
                        # Get just the filename (e.g., "/path/to/api/app.py" -> "app.py")
                        filename = os.path.basename(caller_filename)
                        record.pathname = filename
                        record.lineno = record.caller_lineno

                # Create rich text with colors and formatting
                timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

                # Build the formatted message (without [INFO] - RichHandler adds that)
                parts = [
                    f"[{timestamp}]",
                    f"[{record.caller_module}:{record.caller_funcName}:{record.caller_lineno}]",
                ]

                if record.request_id:
                    parts.append(f"[RID:{record.request_id}]")

                if record.tag:
                    parts.append(f"[{record.tag}]")

                message = record.getMessage()

                # Remove [INFO] if RichHandler added it (it shouldn't, but check anyway)
                # RichHandler displays level separately, so message shouldn't have [INFO]
                if message.startswith("[INFO]"):
                    message = message[6:].lstrip()

                parts.append(message)

                formatted = " ".join(parts)

                return formatted

        return RichFormatter(self)

    def _get_basic_formatter(self):
        """Basic formatter for fallback when rich is not available."""

        class BasicFormatter(logging.Formatter):
            def __init__(self, logger_instance):
                super().__init__(
                    fmt="%(asctime)s [%(levelname)s] %(caller_module)s:%(caller_funcName)s:%(caller_lineno)d %(message)s",
                    datefmt="%H:%M:%S",
                )
                self.logger_instance = logger_instance

            def format(self, record):
                # Add custom fields
                record.request_id = getattr(record, "request_id", RequestIdManager.get())
                record.process_id = os.getpid()
                record.thread_id = threading.get_ident()
                record.tag = getattr(record, "tag", None)

                # Use caller context from extra (already set in _prepare_log_message)
                # Fallback to _get_caller_context if not present (for direct logger calls)
                if not hasattr(record, "caller_module"):
                    context = self.logger_instance._get_caller_context()
                    record.caller_module = context["module"]
                    record.caller_funcName = context["funcName"]
                    record.caller_lineno = context["lineno"]

                formatted = super().format(record)

                if record.request_id:
                    formatted = f"[RID:{record.request_id}] {formatted}"

                if record.tag:
                    formatted = f"[{record.tag}] {formatted}"

                return formatted

        return BasicFormatter(self)

    def _get_file_formatter(self):
        """Formatter for file output."""

        class FileFormatter(logging.Formatter):
            def __init__(self, logger_instance):
                super().__init__(
                    fmt="%(asctime)s [%(levelname)s] %(caller_module)s:%(caller_funcName)s:%(caller_lineno)d [PID:%(process_id)s] [TID:%(thread_id)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                self.logger_instance = logger_instance

            def format(self, record):
                # Add custom fields
                record.request_id = getattr(record, "request_id", RequestIdManager.get())
                record.process_id = os.getpid()
                record.thread_id = threading.get_ident()
                record.tag = getattr(record, "tag", None)

                # Use caller context from extra (already set in _prepare_log_message)
                # Fallback to _get_caller_context if not present (for direct logger calls)
                if not hasattr(record, "caller_module"):
                    context = self.logger_instance._get_caller_context()
                    record.caller_module = context["module"]
                    record.caller_funcName = context["funcName"]
                    record.caller_lineno = context["lineno"]

                formatted = super().format(record)

                if record.request_id:
                    formatted = f"[RID:{record.request_id}] {formatted}"

                if record.tag:
                    formatted = f"[{record.tag}] {formatted}"

                return formatted

        return FileFormatter(self)

    def _get_caller_context(self):
        """Get caller context information."""
        # Walk up the stack to find the first frame that's not in this logger module
        stack = inspect.stack()
        logger_filename = __file__

        for frame_info in stack:
            frame = frame_info.frame
            filename = frame_info.filename

            # Skip frames from this logger module
            if filename == logger_filename:
                continue

            # Found a frame outside the logger module
            module = inspect.getmodule(frame)

            # Extract module name from file path
            if "olympus_echo_backend" in filename:
                # Extract relative path from project root
                parts = filename.split("olympus_echo_backend/")
                if len(parts) > 1:
                    module_path = parts[-1]
                    # Convert path to module name
                    module_path = module_path.replace("/", ".").replace("\\", ".")
                    if module_path.endswith(".py"):
                        module_path = module_path[:-3]
                    # Remove __init__ if present
                    if module_path.endswith(".__init__"):
                        module_path = module_path[:-9]
                    module_name = module_path
                else:
                    module_name = module.__name__ if module else "unknown"
            else:
                # Fallback to module name
                module_name = module.__name__ if module else "unknown"

            return {
                "funcName": frame_info.function,
                "lineno": frame_info.lineno,
                "module": module_name,
                "filename": filename,  # Store the actual filename for RichHandler
            }

        # Fallback if we can't find the caller
        return {
            "funcName": "unknown",
            "lineno": 0,
            "module": "unknown",
            "filename": "unknown",
        }

    def _prepare_log_message(self, level, *args, **kwargs):
        """Prepare log message and extra data."""
        if args and "message" in kwargs:
            tag = args[0]
            message = kwargs["message"]
        elif args:
            tag = args[0]
            message = None
        else:
            tag = kwargs.get("tag")
            message = kwargs.get("message", "")

        # If tag is provided as first arg and no message, treat tag as message
        if message is None and tag:
            message = tag
            tag = None

        context = self._get_caller_context()
        extra = {
            "tag": tag,
            "caller_funcName": context["funcName"],
            "caller_lineno": context["lineno"],
            "caller_module": context["module"],
            "caller_filename": context["filename"],
        }
        return message, extra

    def info(self, *args, **kwargs):
        """Log info message."""
        message, extra = self._prepare_log_message(logging.INFO, *args, **kwargs)
        self.logger.info(message, extra=extra)

    def debug(self, *args, **kwargs):
        """Log debug message."""
        message, extra = self._prepare_log_message(logging.DEBUG, *args, **kwargs)
        self.logger.debug(message, extra=extra)

    def warning(self, *args, **kwargs):
        """Log warning message."""
        message, extra = self._prepare_log_message(logging.WARNING, *args, **kwargs)
        self.logger.warning(message, extra=extra)

    def error(self, *args, **kwargs):
        """Log error message."""
        message, extra = self._prepare_log_message(logging.ERROR, *args, **kwargs)
        self.logger.error(message, extra=extra)

    def critical(self, *args, **kwargs):
        """Log critical message."""
        message, extra = self._prepare_log_message(logging.CRITICAL, *args, **kwargs)
        self.logger.critical(message, extra=extra)

    def exception(self, *args, **kwargs):
        """Log exception with traceback."""
        message, extra = self._prepare_log_message(logging.ERROR, *args, **kwargs)
        self.logger.exception(message, extra=extra)


# Initialize logger
logger = RichLogger(name="voice_assistant_platform", level=logging.INFO)
