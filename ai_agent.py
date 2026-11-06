"""
NEXUS-QUANT AI Advisor Agent
==============================
Flexible multi-provider AI advisor for quantitative trading analysis.
Supports Gemini, Groq, and Ollama backends with automatic fallback and
conditional imports so the application never crashes due to missing
provider packages.

Requires: requests, python-dotenv
Optional:  google-generativeai (for Gemini), groq (for Groq)
Python 3.10+
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on environment variables

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional provider imports
# ---------------------------------------------------------------------------

_GEMINI_AVAILABLE: bool = False
_genai: Any = None
try:
    import google.generativeai as _genai  # type: ignore[no-redef]

    _GEMINI_AVAILABLE = True
except ImportError:
    pass

_GROQ_AVAILABLE: bool = False
_Groq: Any = None
try:
    from groq import Groq as _Groq  # type: ignore[no-redef]

    _GROQ_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Default models per provider
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2",
}

_ENV_KEY_MAP: dict[str, str] = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

_OLLAMA_BASE_URL: str = "http://localhost:11434"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT: str = (
    "You are NEXUS-QUANT AI, a disciplined quantitative trading advisor. "
    "Your analysis framework:\n\n"
    "1. RISK FIRST: Always evaluate downside before upside. Never recommend overleveraged positions.\n"
    "2. REGIME AWARENESS: Identify if the market is trending, ranging, or volatile. "
    "Adapt strategy accordingly.\n"
    "3. CONFLUENCE: Require minimum 3 confirming signals before suggesting any trade setup.\n"
    "4. MACRO CONTEXT: Factor in upcoming economic events and fundamental health.\n"
    "5. CRITIQUE MODE: When asked to review a backtest, identify weaknesses, "
    "curve-fitting risks, and suggest improvements.\n\n"
    "Format your analysis with clear sections: BIAS, KEY LEVELS, SIGNALS, "
    "RISK ASSESSMENT, RECOMMENDATION.\n"
    "Use precise numbers. Never give vague advice. If uncertain, say so explicitly."
)


# ---------------------------------------------------------------------------
# AIAdvisor
# ---------------------------------------------------------------------------


class AIAdvisor:
    """Multi-provider AI advisor for NEXUS-QUANT.

    Parameters
    ----------
    provider : str
        One of ``"gemini"``, ``"groq"``, ``"ollama"``.
    model : str | None
        Override the default model name for the chosen provider.
    api_key : str | None
        Explicit API key.  If *None*, the key is read from the corresponding
        environment variable (``GEMINI_API_KEY`` / ``GROQ_API_KEY``).
    """

    def __init__(
        self,
        provider: str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider: str = provider.lower().strip()
        if self.provider not in _DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported provider '{self.provider}'. "
                f"Choose from: {', '.join(_DEFAULT_MODELS)}"
            )

        self.model: str = model or _DEFAULT_MODELS[self.provider]

        # Resolve API key
        if api_key:
            self.api_key: str | None = api_key
        elif self.provider in _ENV_KEY_MAP:
            self.api_key = os.environ.get(_ENV_KEY_MAP[self.provider])
        else:
            self.api_key = None  # Ollama needs no key

        # Validate provider availability
        if self.provider == "gemini" and not _GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package is not installed. "
                "Install it with: pip install google-generativeai"
            )
        if self.provider == "groq" and not _GROQ_AVAILABLE:
            raise ImportError(
                "groq package is not installed. Install it with: pip install groq"
            )

        # Validate API keys for cloud providers
        if self.provider in ("gemini", "groq") and not self.api_key:
            env_var = _ENV_KEY_MAP[self.provider]
            raise ValueError(
                f"No API key provided for {self.provider}. "
                f"Set the {env_var} environment variable or pass api_key=."
            )

        # Provider dispatch
        self._dispatch: dict[str, Any] = {
            "gemini": self._call_gemini,
            "groq": self._call_groq,
            "ollama": self._call_ollama,
        }

        logger.info("AIAdvisor initialised — provider=%s, model=%s", self.provider, self.model)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """Return the global system prompt for the NEXUS-QUANT AI advisor."""
        return _SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Provider back-ends
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        messages: list[dict[str, str]],
        images: list[Any] | None = None,
    ) -> str:
        """Send *messages* and optional *images* to the Google Gemini API.

        Uses ``google.generativeai.GenerativeModel.generate_content``.
        Includes basic retry logic for transient / rate-limit errors.
        """
        if _genai is None:
            return "[ERROR] google-generativeai is not installed."

        if not self.api_key:
            return (
                "[ERROR] Gemini API key is missing. "
                "Please enter your free Gemini API key in the sidebar under 'AI Advisor' (or in your .env file)."
            )

        try:
            _genai.configure(api_key=self.api_key)

            # Auto-correct model name if previous provider model or deprecated model was passed
            gemini_model = self.model
            if not gemini_model or "llama" in gemini_model.lower() or "gemini-2.0" in gemini_model.lower():
                gemini_model = "gemini-3.5-flash-lite"

            # Gemini expects a single prompt string or a list of content parts.
            prompt_parts: list[str] = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"[System Instructions]\n{content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"[Assistant]\n{content}\n")
                else:
                    prompt_parts.append(f"[User]\n{content}\n")
            prompt = "\n".join(prompt_parts)

            # Build multimodal payload if images are provided
            payload: list[Any] = [prompt]
            if images:
                for img in images:
                    if img is not None:
                        payload.append(img)

            # Fast candidate list:
            # gemini-3.5-flash-lite & gemini-flash-lite-latest respond in ~1s.
            # gemini-3.8-flash is tested with a strict 3.5s timeout.
            candidates: list[str] = []
            if gemini_model and "llama" not in gemini_model.lower():
                candidates.append(gemini_model)
            for fast_alt in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]:
                if fast_alt not in candidates:
                    candidates.append(fast_alt)

            last_error = None
            for cand in candidates:
                try:
                    model = _genai.GenerativeModel(cand)
                    timeout_val = 3.5 if "3.8" in cand else 8.0
                    response = model.generate_content(
                        payload,
                        request_options={"timeout": timeout_val}
                    )
                    if response and response.text:
                        return response.text
                except Exception as exc:
                    last_error = exc
                    logger.warning("Gemini candidate %s failed (%s), trying fast fallback...", cand, exc)
                    continue

            return f"[ERROR] Gemini API request failed: {last_error}"

        except Exception as exc:
            logger.error("Gemini call failed: %s", exc)
            return f"[ERROR] Gemini API error: {exc}"

    def _call_groq(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* to the Groq API and return the response text."""
        if _Groq is None:
            return "[ERROR] groq package is not installed."

        if not self.api_key:
            return (
                "[ERROR] Groq API key is missing. "
                "Please enter your free Groq API key in the sidebar under 'AI Advisor' (or in your .env file)."
            )

        try:
            client = _Groq(api_key=self.api_key)
            groq_model = self.model
            if not groq_model or "gemini" in groq_model.lower():
                groq_model = "llama-3.3-70b-versatile"

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=groq_model,
                        messages=messages,
                    )
                    return response.choices[0].message.content
                except Exception as exc:
                    err_str = str(exc).lower()
                    if "rate" in err_str or "429" in err_str:
                        wait = 2 ** (attempt + 1)
                        logger.warning("Groq rate-limited, retrying in %ds …", wait)
                        time.sleep(wait)
                        continue
                    raise

            return "[ERROR] Groq: rate limit exceeded after retries."

        except Exception as exc:
            logger.error("Groq call failed: %s", exc)
            return f"[ERROR] Groq API error: {exc}"

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        """Send *messages* to a local Ollama instance and return the response text."""
        url = f"{_OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "[ERROR] Empty response from Ollama.")
        except requests.ConnectionError:
            return (
                "[ERROR] Cannot connect to Ollama at localhost:11434. "
                "Make sure Ollama is running (ollama serve)."
            )
        except requests.Timeout:
            return "[ERROR] Ollama request timed out after 120 s."
        except requests.HTTPError as exc:
            logger.error("Ollama HTTP error: %s", exc)
            return f"[ERROR] Ollama HTTP error: {exc}"
        except Exception as exc:
            logger.error("Ollama call failed: %s", exc)
            return f"[ERROR] Ollama error: {exc}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        context: dict[str, Any] | None = None,
        images: list[Any] | None = None,
    ) -> str:
        """Send a *user_message* to the configured AI provider and return its reply.

        Parameters
        ----------
        user_message : str
            The user's question or request.
        context : dict | None
            Optional structured context (e.g. current portfolio, market data)
            that is prepended as a system-level context message.
        images : list | None
            Optional list of PIL Images or image bytes for vision analysis.

        Returns
        -------
        str
            The AI's response text, or a user-friendly error message.
        """
        try:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self._get_system_prompt()},
            ]

            if context:
                context_str = json.dumps(context, indent=2, default=str)
                messages.append({
                    "role": "system",
                    "content": (
                        "The following JSON contains current contextual data "
                        "for your analysis:\n\n" + context_str
                    ),
                })

            messages.append({"role": "user", "content": user_message})

            if self.provider == "gemini":
                return self._call_gemini(messages, images=images)

            if images and self.provider != "gemini":
                messages[-1]["content"] += (
                    "\n\n[System Note: User attached image screenshots, but vision is "
                    "supported by Google Gemini. Switch to Gemini in the sidebar to inspect screenshots directly.]"
                )

            handler = self._dispatch.get(self.provider)
            if handler is None:
                return f"[ERROR] No handler for provider '{self.provider}'."

            return handler(messages)

        except Exception as exc:
            logger.error("chat() failed: %s", exc)
            return (
                f"I encountered an error while processing your request: {exc}. "
                "Please try again or switch to a different AI provider."
            )

    def analyze_backtest(self, metrics: dict[str, Any]) -> str:
        """Ask the AI to critique a backtest based on its *metrics*.

        Parameters
        ----------
        metrics : dict
            Backtest performance metrics (e.g. Sharpe, max drawdown, win rate).

        Returns
        -------
        str
            The AI's critique and suggestions.
        """
        clean_metrics = {}
        priority_keys = [
            "total_trades", "win_rate_pct", "profit_factor", "sharpe_ratio",
            "max_drawdown_pct", "total_return_pct", "avg_rr_achieved",
            "won_trades", "lost_trades", "be_trades", "setup_breakdown", "trigger_breakdown"
        ]
        for k in priority_keys:
            if k in metrics:
                clean_metrics[k] = metrics[k]

        if not clean_metrics:
            clean_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (list, tuple)) and k != "monthly_returns"}

        prompt = (
            "You are NEXUS-QUANT AI, a senior quantitative portfolio manager. "
            "Critique this backtest concisely in clean markdown (under 250 words total):\n\n"
            f"PERFORMANCE METRICS:\n{json.dumps(clean_metrics, indent=2, default=str)}\n\n"
            "Format your answer with exactly these 4 clear sections with emojis:\n"
            "1. 🎯 SCORE & EXECUTIVE VERDICT (Grade A-F, edge evaluation)\n"
            "2. 🛡️ WHAT WORKED WELL (Win rate, risk:reward, setup breakdown)\n"
            "3. ⚠️ RISKS & RED FLAGS (Max drawdown, trade sample size, curve-fitting risks)\n"
            "4. 💡 3 CONCRETE ACTIONS (Exact parameter or execution adjustments)\n"
        )
        return self.chat(prompt)

    def analyze_setup(self, setup_data: dict[str, Any]) -> str:
        """Ask the AI to assess a trade setup.

        Parameters
        ----------
        setup_data : dict
            Current market setup data — typically includes trend direction,
            support/resistance levels, FVG zones, RSI, and recent news.

        Returns
        -------
        str
            Structured trade assessment.
        """
        prompt = (
            "Analyze the following trade setup and provide a structured assessment. "
            "Remember: require at least 3 confirming signals before recommending entry.\n\n"
            "SETUP DATA:\n"
            f"{json.dumps(setup_data, indent=2, default=str)}\n\n"
            "Provide your analysis in these sections:\n"
            "1. BIAS (Bullish / Bearish / Neutral)\n"
            "2. KEY LEVELS (Support, Resistance, FVG zones)\n"
            "3. SIGNALS (list confirming and conflicting signals)\n"
            "4. RISK ASSESSMENT (R:R ratio, position sizing suggestion)\n"
            "5. RECOMMENDATION (Entry, Stop-Loss, Take-Profit or STAND ASIDE)\n"
        )
        return self.chat(prompt, context=setup_data)

    def analyze_fundamentals(self, fundamental_card: dict[str, Any]) -> str:
        """Ask the AI for a fundamental-health opinion on a company.

        Parameters
        ----------
        fundamental_card : dict
            Output from ``fundamental_engine.get_fundamental_card()``.

        Returns
        -------
        str
            Fundamental analysis opinion.
        """
        ticker = fundamental_card.get("ticker", "UNKNOWN")
        company = fundamental_card.get("company_name", "Unknown Company")

        prompt = (
            f"Provide a fundamental analysis opinion for {company} ({ticker}). "
            "Use the data below to assess the company's financial health, "
            "growth trajectory, and investment merit.\n\n"
            "FUNDAMENTAL DATA:\n"
            f"{json.dumps(fundamental_card, indent=2, default=str)}\n\n"
            "Provide your analysis in these sections:\n"
            "1. FINANCIAL HEALTH SUMMARY\n"
            "2. GROWTH ASSESSMENT\n"
            "3. VALUATION (cheap, fair, expensive relative to quality)\n"
            "4. RISKS\n"
            "5. VERDICT (Strong Buy / Buy / Hold / Sell / Strong Sell)\n"
        )
        return self.chat(prompt, context=fundamental_card)

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_available_providers() -> list[str]:
        """Return a list of AI providers that are currently usable.

        Checks for installed packages, set API keys, and Ollama
        reachability.
        """
        available: list[str] = []

        # Gemini
        if _GEMINI_AVAILABLE:
            key = os.environ.get("GEMINI_API_KEY", "")
            if key:
                available.append("gemini")

        # Groq
        if _GROQ_AVAILABLE:
            key = os.environ.get("GROQ_API_KEY", "")
            if key:
                available.append("groq")

        # Ollama — check reachability
        try:
            resp = requests.get(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                available.append("ollama")
        except Exception:
            pass  # Ollama not running or not installed

        return available
