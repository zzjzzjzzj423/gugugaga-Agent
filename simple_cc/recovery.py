from __future__ import annotations

import random
import time

from . import config
from .provider import is_context_length_error


class RecoveryState:
    def __init__(self):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = config.PRIMARY_MODEL


def retry_delay(attempt: int) -> float:
    base = min(config.BASE_DELAY_MS * (2**attempt), 32000) / 1000
    return base + random.uniform(0, base * 0.25)


def with_retry(fn, state: RecoveryState):
    for attempt in range(config.MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except Exception as error:
            name = type(error).__name__.lower()
            message = str(error).lower()
            if "ratelimit" in name or "429" in message:
                delay = retry_delay(attempt)
                print(
                    f"  \033[33m[429] retry {attempt + 1}/{config.MAX_RETRIES} "
                    f"after {delay:.1f}s\033[0m"
                )
                time.sleep(delay)
                continue
            if (
                "overloaded" in name
                or "529" in message
                or "overloaded" in message
            ):
                state.consecutive_529 += 1
                if (
                    state.consecutive_529 >= config.MAX_CONSECUTIVE_529
                    and config.FALLBACK_MODEL
                ):
                    state.current_model = config.FALLBACK_MODEL
                    state.consecutive_529 = 0
                    print(
                        f"  \033[31m[529] switching to "
                        f"{config.FALLBACK_MODEL}\033[0m"
                    )
                delay = retry_delay(attempt)
                print(
                    f"  \033[33m[529] retry {attempt + 1}/{config.MAX_RETRIES} "
                    f"after {delay:.1f}s\033[0m"
                )
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"Max retries ({config.MAX_RETRIES}) exceeded")


def is_prompt_too_long_error(error: Exception) -> bool:
    return is_context_length_error(error)
