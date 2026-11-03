"""Request-local ceilings charged before model/network attempts, including retries."""

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps

from fastapi import HTTPException

from app.config import settings
from app.services.response_safety import contains_credential


class BudgetExceeded(ValueError):
    pass


current_budget: ContextVar["RequestBudget | None"] = ContextVar(
    "request_budget", default=None
)


@dataclass
class RequestBudget:
    max_calls: int = field(default_factory=lambda: settings.request_max_model_attempts)
    max_input_bytes: int = field(
        default_factory=lambda: settings.request_max_input_bytes
    )
    max_output_tokens: int = field(
        default_factory=lambda: settings.request_max_output_tokens
    )
    max_external_attempts: int = field(
        default_factory=lambda: settings.request_max_external_attempts
    )
    max_currency: float | None = field(
        default_factory=lambda: settings.request_max_currency
    )
    calls: int = 0
    input_bytes: int = 0
    output_tokens: int = 0
    external_attempts: int = 0
    currency_cost: float | None = 0.0
    started: float = field(default_factory=time.monotonic)

    def remaining(self):
        return max(
            0, settings.request_deadline_seconds - (time.monotonic() - self.started)
        )

    def reserve_external(self):
        if (
            self.external_attempts >= self.max_external_attempts
            or self.remaining() <= 0
        ):
            raise BudgetExceeded("External request budget exhausted")
        self.external_attempts += 1

    def reserve_model(self, text, max_tokens, *, provider, embedding=False):
        size = len(text.encode("utf-8"))
        if (
            self.calls >= self.max_calls
            or self.input_bytes + size > self.max_input_bytes
            or self.output_tokens + max_tokens > self.max_output_tokens
            or self.remaining() <= 0
        ):
            raise BudgetExceeded("Model request budget exhausted")
        # UTF-8 bytes conservatively reserve input tokens; this is not measured usage.
        price_in, price_out = (
            settings.request_input_price_per_million,
            settings.request_output_price_per_million,
        )
        if embedding:
            price_in, price_out = settings.request_embedding_price_per_million, 0
        cost = (
            0.0
            if provider == "ollama"
            else (
                (size * price_in + max_tokens * price_out) / 1_000_000
                if price_in is not None and price_out is not None
                else None
            )
        )
        if self.max_currency is not None and (
            cost is None
            or self.currency_cost is None
            or self.currency_cost + cost > self.max_currency
        ):
            raise BudgetExceeded(
                "Currency ceiling needs known prices and sufficient allowance"
            )
        if provider != "ollama":
            self.reserve_external()
        self.calls += 1
        self.input_bytes += size
        self.output_tokens += max_tokens
        self.currency_cost = (
            self.currency_cost + cost
            if self.currency_cost is not None and cost is not None
            else None
        )

    def snapshot(self):
        return {
            "model_attempts": self.calls,
            "reserved_input_bytes": self.input_bytes,
            "reserved_output_tokens": self.output_tokens,
            "external_attempts": self.external_attempts,
            "currency_cost": self.currency_cost,
            "cost_kind": "configured_reservation_not_invoice",
            "remaining_seconds": round(self.remaining(), 3),
        }


def request_limits(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        # Follow-up invokes the same query endpoint; retain one shared outer budget.
        if current_budget.get() is not None:
            return await func(*args, **kwargs)
        token = current_budget.set(RequestBudget())
        try:
            async with asyncio.timeout(settings.request_deadline_seconds):
                result = await func(*args, **kwargs)
                if contains_credential(result):
                    raise HTTPException(
                        422,
                        detail={
                            "code": "RESPONSE_PRIVACY",
                            "message": "Response withheld by the credential guard.",
                        },
                    )
                return result
        except TimeoutError as exc:
            raise HTTPException(
                504,
                detail={
                    "code": "REQUEST_DEADLINE",
                    "message": "Request deadline exceeded; answer withheld.",
                },
            ) from exc
        except BudgetExceeded as exc:
            raise HTTPException(
                429,
                detail={
                    "code": "REQUEST_BUDGET",
                    "message": "Request budget exhausted; answer withheld.",
                },
            ) from exc
        finally:
            current_budget.reset(token)

    return wrapped
