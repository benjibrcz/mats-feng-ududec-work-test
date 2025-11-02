import os
import time
import random
from typing import Dict, Optional, Tuple, Any

from openai import OpenAI


TRUE_STRINGS = ["True", " true", "TRUE", " true.", "True."]
FALSE_STRINGS = ["False", " false", "FALSE", " false.", "False."]


class HyperbolicClient:
    """
    Thin wrapper over an OpenAI-compatible endpoint (Hyperbolic) to get
    first-token logprobs for True/False.

    Required env vars:
      - HYPERBOLIC_API_KEY
      - HYPERBOLIC_BASE_URL (e.g., https://api.hyperbolic.xyz/v1)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.getenv("HYPERBOLIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("HYPERBOLIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("Missing HYPERBOLIC_API_KEY or OPENAI_API_KEY.")
        if not base_url:
            raise RuntimeError("Missing HYPERBOLIC_BASE_URL or OPENAI_BASE_URL.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # Simple backoff for transient errors
        self.max_retries = 8
        self.retry_min_s = 0.5
        self.retry_max_s = 2.0

        # Prompt-level cache to avoid duplicate API cost
        self._logprob_cache: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def _backoff(self, attempt: int) -> None:
        delay = min(self.retry_max_s, self.retry_min_s * (2 ** attempt))
        delay = delay * (0.5 + random.random())
        time.sleep(delay)

    def _is_chat_model(self, model: str) -> bool:
        m = model.lower()
        return "instruct" in m or "chat" in m

    def score_true_false(self, model: str, prompt: str, temperature: float = 0.0, top_logprobs: int = 10) -> Tuple[float, float]:
        """
        Returns (logp_true, logp_false) for the first output token.
        Falls back to greedy text if top_logprobs are unavailable.
        """
        cache_key = (model, prompt)
        if cache_key in self._logprob_cache:
            return self._logprob_cache[cache_key]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                logp_true, logp_false = None, None

                if self._is_chat_model(model):
                    # Use Chat Completions for chat/instruct models; avoid logprobs (provider may not support)
                    resp: Any = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Decide True/False as a single word."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=1,
                        temperature=temperature,
                    )
                    choice = resp.choices[0]
                    text_token = getattr(choice.message, "content", "") or ""
                    text_token = text_token.strip()
                    norm = text_token.strip().lower().rstrip(" .!?")
                    if any(text_token.startswith(s.strip()) for s in TRUE_STRINGS) or norm.startswith("true") or norm.startswith("yes") or norm.startswith("correct"):
                        # Gentle fallback to avoid huge deltas
                        logp_true, logp_false = -0.1, -0.3
                    elif any(text_token.startswith(s.strip()) for s in FALSE_STRINGS) or norm.startswith("false") or norm.startswith("no") or norm.startswith("incorrect"):
                        logp_true, logp_false = -0.3, -0.1
                    else:
                        logp_true, logp_false = -1.0, -1.0
                else:
                    # Base models via Completions with logprobs if available
                    # Try with logprobs, then degrade on provider errors
                    resp: Any = None
                    last_provider_err: Optional[Exception] = None
                    for lp in (top_logprobs, 5, 1, 0):
                        try:
                            if lp > 0:
                                resp = self.client.completions.create(
                                    model=model,
                                    prompt=prompt,
                                    max_tokens=1,
                                    temperature=temperature,
                                    logprobs=lp,
                                )
                            else:
                                resp = self.client.completions.create(
                                    model=model,
                                    prompt=prompt,
                                    max_tokens=1,
                                    temperature=temperature,
                                )
                            last_provider_err = None
                            break
                        except Exception as e:  # noqa: BLE001
                            last_provider_err = e
                            continue
                    if resp is None and last_provider_err is not None:
                        raise last_provider_err
                    choice = resp.choices[0]
                    logprobs = choice.logprobs
                    if logprobs is not None:
                        top = getattr(logprobs, "top_logprobs", None)
                        if top:
                            top0 = top[0]
                            # top0 may be a dict(token->logprob) or a list of objects with token/logprob
                            if isinstance(top0, dict):
                                for token, lp in top0.items():
                                    if token in TRUE_STRINGS:
                                        logp_true = lp
                                    if token in FALSE_STRINGS:
                                        logp_false = lp
                            else:
                                for item in top0:
                                    token = getattr(item, "token", None) or (item.get("token") if isinstance(item, dict) else None)
                                    lp = getattr(item, "logprob", None) or (item.get("logprob") if isinstance(item, dict) else None)
                                    if token is None or lp is None:
                                        continue
                                    if token in TRUE_STRINGS:
                                        logp_true = lp
                                    if token in FALSE_STRINGS:
                                        logp_false = lp
                    # Fallback to text
                    if (logp_true is None) or (logp_false is None):
                        text_token = choice.text
                        if isinstance(text_token, str):
                            text_token = text_token.strip()
                        norm = text_token.strip().lower().rstrip(" .!?")
                        if any(text_token.startswith(s.strip()) for s in TRUE_STRINGS) or norm.startswith("true") or norm.startswith("yes") or norm.startswith("correct"):
                            logp_true = -0.1 if logp_true is None else logp_true
                            logp_false = -0.3 if logp_false is None else logp_false
                        elif any(text_token.startswith(s.strip()) for s in FALSE_STRINGS) or norm.startswith("false") or norm.startswith("no") or norm.startswith("incorrect"):
                            logp_true = -0.3 if logp_true is None else logp_true
                            logp_false = -0.1 if logp_false is None else logp_false
                        else:
                            logp_true = -1.0
                            logp_false = -1.0

                result = (float(logp_true), float(logp_false))
                self._logprob_cache[cache_key] = result
                return result
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
                else:
                    raise

        assert last_err is not None
        raise last_err

    # Debug-oriented: returns matched tokens too
    def score_true_false_details(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_logprobs: int = 10,
    ) -> Tuple[float, float, Optional[str], Optional[str]]:
        if self._is_chat_model(model):
            # No logprobs for chat; parse token text only
            resp: Any = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Decide True/False as a single word."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                temperature=temperature,
            )
            choice = resp.choices[0]
            text_token = getattr(choice.message, "content", "") or ""
            text_token = text_token.strip()
            norm = text_token.strip().lower().rstrip(" .!?")
            if any(text_token.startswith(s.strip()) for s in TRUE_STRINGS) or norm.startswith("true") or norm.startswith("yes") or norm.startswith("correct"):
                return -0.1, -0.3, text_token, None
            if any(text_token.startswith(s.strip()) for s in FALSE_STRINGS) or norm.startswith("false") or norm.startswith("no") or norm.startswith("incorrect"):
                return -0.3, -0.1, None, text_token
            return -1.0, -1.0, text_token, text_token

        # Base model with logprobs if available; degrade on provider errors
        resp: Any = None
        last_provider_err: Optional[Exception] = None
        for lp in (top_logprobs, 5, 1, 0):
            try:
                if lp > 0:
                    resp = self.client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=1,
                        temperature=temperature,
                        logprobs=lp,
                    )
                else:
                    resp = self.client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=1,
                        temperature=temperature,
                    )
                last_provider_err = None
                break
            except Exception as e:  # noqa: BLE001
                last_provider_err = e
                continue
        if resp is None and last_provider_err is not None:
            raise last_provider_err
        choice = resp.choices[0]
        logprobs = choice.logprobs
        logp_true, logp_false = None, None
        tok_true: Optional[str] = None
        tok_false: Optional[str] = None
        if logprobs is not None:
            top = getattr(logprobs, "top_logprobs", None)
            if top:
                top0 = top[0]
                # Normalize to list of dicts
                pairs: list[tuple[str, float]] = []
                if isinstance(top0, dict):
                    pairs = list(top0.items())
                else:
                    for item in top0:
                        token = getattr(item, "token", None) or (item.get("token") if isinstance(item, dict) else None)
                        lp = getattr(item, "logprob", None) or (item.get("logprob") if isinstance(item, dict) else None)
                        if token is not None and lp is not None:
                            pairs.append((token, lp))
                # exact match first
                cand_true = {"True", " True"}
                cand_false = {"False", " False"}
                for token, lp in pairs:
                    if token in cand_true:
                        logp_true = lp
                        tok_true = token
                    if token in cand_false:
                        logp_false = lp
                        tok_false = token
                # fallback: strip one leading space
                if logp_true is None or logp_false is None:
                    for token, lp in pairs:
                        lt = token.lstrip()
                        if logp_true is None and lt == "True":
                            logp_true = lp
                            tok_true = token
                        if logp_false is None and lt == "False":
                            logp_false = lp
                            tok_false = token
        if (logp_true is None) or (logp_false is None):
            text_token = choice.text.strip() if isinstance(choice.text, str) else ""
            norm = text_token.strip().lower().rstrip(" .!?")
            if logp_true is None and (any(text_token.startswith(s.strip()) for s in TRUE_STRINGS) or norm.startswith("true") or norm.startswith("yes") or norm.startswith("correct")):
                logp_true = -0.1
                tok_true = text_token
            if logp_false is None and (any(text_token.startswith(s.strip()) for s in FALSE_STRINGS) or norm.startswith("false") or norm.startswith("no") or norm.startswith("incorrect")):
                logp_false = -0.1
                tok_false = text_token
            if logp_true is None:
                logp_true = -1.0
            if logp_false is None:
                logp_false = -1.0
        return float(logp_true), float(logp_false), tok_true, tok_false

    # Strict text-only decision: generate 1 token at temp=0 and accept ONLY True/False variants
    def decide_true_false_strict(self, model: str, prompt: str) -> int:
        if self._is_chat_model(model):
            resp: Any = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Decide True/False as a single word."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                temperature=0.0,
            )
            choice = resp.choices[0]
            token = (getattr(choice.message, "content", "") or "").strip()
        else:
            resp: Any = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=1,
                temperature=0.0,
            )
            token = (resp.choices[0].text or "").strip()

        if token.startswith("True") or token.startswith(" true"):
            return 1
        if token.startswith("False") or token.startswith(" false"):
            return 0
        # Indeterminate; default to False to avoid optimistic bias
        return 0


