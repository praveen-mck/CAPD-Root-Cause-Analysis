
# -*- coding: utf-8 -*-
"""
LLM client utilities for Azure OpenAI (Async).

Responsibilities:
- Send chat completion requests
- Support JSON mode
- Retry with exponential backoff + jitter

This module is intentionally generic:
- It does NOT import taxonomy or pipeline code.
- It expects an AsyncAzureOpenAI client to be passed in.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Dict, Optional

from openai import OpenAIError


async def send_chat_request(
    client: Any,
    deployment: str,
    system_text: str,
    user_text: str,
    max_out_tokens: int = 256,
    use_json_mode: bool = True,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    max_retries: int = 3,
    base_backoff: float = 0.6,
) -> Any:
    """
    Send a single chat completion request with retries.

    Args:
        client: AsyncAzureOpenAI client instance.
        deployment: Azure OpenAI deployment name (model).
        system_text: System prompt text.
        user_text: User prompt text.
        max_out_tokens: Max completion tokens.
        use_json_mode: If True, enforce JSON object response via response_format.
        temperature: Optional temperature.
        seed: Optional seed (if supported).
        max_retries: Number of attempts total.
        base_backoff: Base backoff in seconds (exponential).

    Returns:
        The raw response object from Azure OpenAI.

    Raises:
        The last exception if all retries fail.
    """
    params: Dict[str, Any] = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        # Your original script used max_completion_tokens; many SDKs accept it.
        # Keep this consistent with your current working code.
        "max_completion_tokens": int(max_out_tokens),
    }

    if use_json_mode:
        params["response_format"] = {"type": "json_object"}

    # Optional knobs (only included if not None)
    if temperature is not None:
        params["temperature"] = float(temperature)
    if seed is not None:
        params["seed"] = int(seed)

    last_exc: Optional[BaseException] = None

    for attempt in range(1, int(max_retries) + 1):
        try:
            return await client.chat.completions.create(**params)
        except (OpenAIError, asyncio.TimeoutError, Exception) as e:
            last_exc = e
            if attempt >= max_retries:
                break

            # Exponential backoff + small jitter
            sleep_s = float(base_backoff) * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            await asyncio.sleep(sleep_s)

    # If we get here, all retries failed
    raise last_exc if last_exc else RuntimeError("send_chat_request failed with unknown error")
