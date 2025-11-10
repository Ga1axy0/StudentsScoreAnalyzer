"""Runtime hook: ensure process exits when console window is closed on Windows.

This hook registers a Windows console control handler so that clicking the
console "X" button (CTRL_CLOSE_EVENT) or logoff/shutdown events will cause
the Python process to terminate, avoiding lingering background servers
(Streamlit) that otherwise require Ctrl+C.
"""
from __future__ import annotations

import os
import sys
import signal
import atexit

# Only meaningful on Windows
if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1
    CTRL_CLOSE_EVENT = 2
    CTRL_LOGOFF_EVENT = 5
    CTRL_SHUTDOWN_EVENT = 6

    HandlerRoutine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

    def _terminate(_reason: str) -> None:
        # Try a graceful path first
        try:
            signal.raise_signal(signal.SIGTERM)
        except Exception:
            pass
        # Ensure exit if signal does not propagate
        try:
            os._exit(0)
        except Exception:
            pass

    @HandlerRoutine
    def _console_ctrl_handler(dwCtrlType: int) -> int:
        # Return True to indicate we've handled the event and are exiting.
        if dwCtrlType in (CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT, CTRL_BREAK_EVENT):
            _terminate(f"ctrl_event:{dwCtrlType}")
            return 1
        if dwCtrlType == CTRL_C_EVENT:
            _terminate("ctrl_c")
            return 1
        return 0

    try:
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_ctrl_handler, 1)
    except Exception:
        # Best-effort fallback to normal signal handlers
        pass

# Also register normal POSIX-style signal handlers as a fallback
for _sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
    if _sig is not None:
        try:
            signal.signal(_sig, lambda *args, **kwargs: os._exit(0))
        except Exception:
            pass

# Ensure process termination at interpreter shutdown if still alive
atexit.register(lambda: os._exit(0))
