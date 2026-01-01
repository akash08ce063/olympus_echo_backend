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
    from rich.text import Text
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
                show_time=True,
                show_level=True,
                show_path=True,
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
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
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

                # Get caller context
                context = self.logger_instance._get_caller_context()
                record.caller_module = context["module"]
                record.caller_funcName = context["funcName"]
                record.caller_lineno = context["lineno"]

                # Create rich text with colors and formatting
                timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

                # Color mapping for levels
                level_colors = {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red bold",
                }

                level_color = level_colors.get(record.levelname, "white")

                # Build the formatted message
                parts = [
                    f"[{timestamp}]",
                    f"[{record.levelname}]",
                    f"[{record.caller_module}:{record.caller_funcName}:{record.caller_lineno}]",
                ]

                if record.request_id:
                    parts.append(f"[RID:{record.request_id}]")

                if record.tag:
                    parts.append(f"[{record.tag}]")

                parts.append(record.getMessage())

                formatted = " ".join(parts)

                return formatted

        return RichFormatter(self)

    def _get_basic_formatter(self):
        """Basic formatter for fallback when rich is not available."""
        class BasicFormatter(logging.Formatter):
            def __init__(self, logger_instance):
                super().__init__(
                    fmt="%(asctime)s [%(levelname)s] %(caller_module)s:%(caller_funcName)s:%(caller_lineno)d %(message)s",
                    datefmt="%H:%M:%S"
                )
                self.logger_instance = logger_instance

            def format(self, record):
                # Add custom fields
                record.request_id = getattr(record, "request_id", RequestIdManager.get())
                record.process_id = os.getpid()
                record.thread_id = threading.get_ident()
                record.tag = getattr(record, "tag", None)

                # Get caller context
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
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                self.logger_instance = logger_instance

            def format(self, record):
                # Add custom fields
                record.request_id = getattr(record, "request_id", RequestIdManager.get())
                record.process_id = os.getpid()
                record.thread_id = threading.get_ident()
                record.tag = getattr(record, "tag", None)

                # Get caller context
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
        frame = inspect.currentframe().f_back.f_back.f_back
        module = inspect.getmodule(frame)
        return {
            "funcName": frame.f_code.co_name if frame else "unknown",
            "lineno": frame.f_lineno if frame else 0,
            "module": module.__name__ if module else "unknown",
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
            "caller_module": context["module"]
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
logger = RichLogger(
    name="voice_assistant_platform",
    level=logging.INFO
)
