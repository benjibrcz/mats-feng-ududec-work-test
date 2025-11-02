from typing import List

from .data_loader import Example


ZERO_SHOT_INSTRUCTION = (
    "Decide if the claim is factually true given the question. Output exactly True or False.\n\n"
)


def build_zero_shot_prompt(example: Example) -> str:
    return ZERO_SHOT_INSTRUCTION + example.to_prompt_stub()


def build_many_shot_prompt(examples: List[Example], query_example: Example) -> str:
    parts: List[str] = [ZERO_SHOT_INSTRUCTION]
    for ex in examples:
        parts.append(
            f"Question: {ex.question}\nClaim: {ex.claim}\nAnswer: {('True' if ex.label == 1 else 'False')}\n\n"
        )
    parts.append(query_example.to_prompt_stub())
    return "".join(parts)


