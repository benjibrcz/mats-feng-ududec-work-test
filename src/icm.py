import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import json

from tqdm import tqdm

from .data_loader import Example
from .hyperbolic_client import HyperbolicClient
from .prompts import build_many_shot_prompt


class ICMSearch:
    def __init__(
        self,
        base_model: str,
        client: HyperbolicClient,
        alpha: float = 50.0,
        context_cap: int = 96,
        seed: int = 0,
        p_unlabeled: float = 0.9,
    ) -> None:
        self.base_model = base_model
        self.client = client
        self.alpha = alpha
        self.context_cap = context_cap
        self.rng = random.Random(seed)
        self.p_unlabeled = p_unlabeled
        # Use text-only decisions to avoid logprobs when provider is unstable
        self.use_text_decider: bool = False

        # Current labeled set D: mapping example_id -> assigned label (0/1)
        self.labels: Dict[int, int] = {}
        self.history: List[Dict[str, object]] = []

    def _sample_context(self, pool: Sequence[Example], exclude_id: Optional[int] = None) -> List[Example]:
        candidates = [ex for ex in pool if ex.example_id != exclude_id and ex.example_id in self.labels]
        # If context_cap <= 0, treat as unlimited
        if self.context_cap is None or self.context_cap <= 0:
            return candidates
        if len(candidates) <= self.context_cap:
            return candidates
        return self.rng.sample(candidates, self.context_cap)

    def _score_mutual_predictability(self, pool: Sequence[Example]) -> float:
        # P_theta(D) = sum_i log P(y_i | x_i, context(D \ {i}))
        total = 0.0
        for ex in pool:
            if ex.example_id not in self.labels:
                continue
            yi = self.labels[ex.example_id]
            ctx = self._sample_context(pool, exclude_id=ex.example_id)
            # Build many-shot with context labels
            ctx_with_labels = []
            for c in ctx:
                if c.example_id in self.labels:
                    labeled = Example(
                        example_id=c.example_id,
                        question=c.question,
                        claim=c.claim,
                        label=self.labels[c.example_id],
                        group_id=c.group_id,
                    )
                    ctx_with_labels.append(labeled)
            prompt = build_many_shot_prompt(ctx_with_labels, ex)
            logp_true, logp_false = self.client.score_true_false(self.base_model, prompt)
            total += (logp_true if yi == 1 else logp_false)
        return total

    def _propose_label(self, pool: Sequence[Example], target: Example) -> Tuple[int, float, float, Optional[str], Optional[str], str, int]:
        ctx = self._sample_context(pool, exclude_id=target.example_id)
        ctx_with_labels = []
        for c in ctx:
            if c.example_id in self.labels:
                ctx_with_labels.append(
                    Example(
                        example_id=c.example_id,
                        question=c.question,
                        claim=c.claim,
                        label=self.labels[c.example_id],
                        group_id=c.group_id,
                    )
                )
        prompt = build_many_shot_prompt(ctx_with_labels, target)
        if self.use_text_decider:
            decision = self.client.decide_true_false_strict(self.base_model, prompt)
            logp_true = -0.1 if decision == 1 else -0.3
            logp_false = -0.1 if decision == 0 else -0.3
            tok_t, tok_f = ("True", None) if decision == 1 else (None, "False")
        else:
            logp_true, logp_false, tok_t, tok_f = self.client.score_true_false(self.base_model, prompt)
        #logp_true, logp_false = self.client.score_true_false(self.base_model, prompt)
        proposed = 1 if logp_true >= logp_false else 0
        return proposed, logp_true, logp_false, tok_t, tok_f, prompt, len(ctx_with_labels)
        #return proposed, logp_true, logp_false, prompt, len(ctx_with_labels)


    def _pick_index(self, train: Sequence[Example]) -> int:
        unlabeled = [ex.example_id for ex in train if ex.example_id not in self.labels]
        if unlabeled and self.rng.random() < self.p_unlabeled:
            return self.rng.choice(unlabeled)
        return self.rng.choice([ex.example_id for ex in train])

    def initialize_random(self, train: Sequence[Example], k_init: int = 8) -> None:
        init = self.rng.sample(list(train), k=min(k_init, len(train)))
        for ex in init:
            self.labels[ex.example_id] = self.rng.choice([0, 1])

    def run(
        self,
        train: Sequence[Example],
        steps: int = 1000,
        t0: float = 10.0,
        tmin: float = 0.01,
        beta: float = 0.99,
        target_labels: Optional[int] = None,
        skip_initial_score: bool = True,
    ) -> Dict[int, int]:
        # Annealed search per Algorithm 1 (without logical consistency fix).
        t_start = time.time()
        if skip_initial_score:
            print("Skipping initial mutual predictability scoring (using local ΔU only)...")
            current_p = 0.0
            current_u = 0.0
        else:
            print("Scoring initial mutual predictability...")
            current_p = self._score_mutual_predictability(train)
            current_u = self.alpha * current_p  # I(D)=0 per assignment
        accepted = 0
        Path("debug").mkdir(exist_ok=True)
        for n in tqdm(range(1, steps + 1), desc="ICM search", smoothing=0.1):
            if target_labels is not None and len(self.labels) >= target_labels:
                break
            # Temperature schedule based on accepted proposals
            T = max(tmin, t0 / (1.0 + beta * math.log(max(2, accepted + 1))))
            idx = self._pick_index(train)
            target_map = {ex.example_id: ex for ex in train}
            target = target_map[idx]
            proposed_label, lp_t, lp_f, tok_t, tok_f, prompt, ctx_len = self._propose_label(train, target)
            #proposed_label, lp_t, lp_f, prompt, ctx_len = self._propose_label(train, target)

            prev_label = self.labels.get(target.example_id)
            # Local-only ΔU using leave-one-out prompt
            new_term = (lp_t if proposed_label == 1 else lp_f)
            old_term = (lp_t if prev_label == 1 else (lp_f if prev_label == 0 else 0.0))
            denom = max(1, ctx_len)
            delta = self.alpha * (new_term - old_term) / denom

            accept = delta > 0 or self.rng.random() < math.exp(delta / max(1e-6, T))
            if accept:
                self.labels[target.example_id] = proposed_label
                accepted += 1
            else:
                if prev_label is None:
                    self.labels.pop(target.example_id, None)
                else:
                    self.labels[target.example_id] = prev_label
            rec = {
                "i": target.example_id,
                "accepted": accept,
                "dU": float(delta),
                "proposed": int(proposed_label),
                "prev": (int(prev_label) if prev_label is not None else None),
                "gold": int(target.label),
                "logp_true": float(lp_t),
                "logp_false": float(lp_f),
                "true_token": tok_t,
                "false_token": tok_f,
                "ctx_size": ctx_len,
                "T": float(T),
                "alpha": float(self.alpha),
                "model": self.base_model,
            }
            self.history.append(rec)
            if n % 25 == 0:
                print(f"[n={n}] |D|={len(self.labels)} acc_rate={accepted/max(1,n):.2f} ΔU={rec['dU']:.3f} logp(T)={lp_t:.2f} logp(F)={lp_f:.2f}")
            if n % 50 == 0:
                with open(f"debug/prompt_{n}.txt", "w") as f:
                    f.write(prompt)

        with open("debug/icm_history.jsonl", "w") as f:
            for r in self.history:
                f.write(json.dumps(r) + "\n")
        elapsed = time.time() - t_start
        print(f"ICM completed in {elapsed:.1f}s; labeled {len(self.labels)} items; accepted {accepted}.")

        return dict(self.labels)


