"""
agents/agentic/llm_config.py
------------------------------
LLM backend configuration for the opt-in agentic mode.

Never raises — returns None when no usable backend is configured, so
callers (agents/agentic_orchestrator.py) can fall back to the deterministic
pipeline cleanly instead of crashing on a missing/placeholder API key.

Author: Anthony Seumal
Project: VRTQ-RL
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_llm_config() -> Optional[dict]:
    """
    Returns an AutoGen-compatible llm_config dict, or None if OPENAI_API_KEY
    is missing or still the placeholder value.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "your_key_here":
        return None

    return {
        "config_list": [{
            "model": os.getenv("AGENTIC_OPENAI_MODEL", "gpt-4o-mini"),
            "api_key": key,
        }],
        "temperature": 0.2,
        "timeout": 30,
        "cache_seed": None,  # disable disk cache so each demo run is a fresh conversation
    }
