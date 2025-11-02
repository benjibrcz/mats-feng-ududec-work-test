import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Example:
    example_id: int
    question: str
    claim: str
    label: int  # 1 = True, 0 = False
    group_id: int  # consistency_id (same question group)
    myth_label: Optional[int] = None  # optional: 1/0 if dataset provides a "myth" label

    def to_prompt_stub(self) -> str:
        return (
            f"Question: {self.question}\n"
            f"Claim: {self.claim}\n"
            f"Answer: "
        )


def load_truthfulqa(path: str) -> List[Example]:
    p = Path(path)
    data = json.loads(p.read_text())
    examples: List[Example] = []
    for idx, row in enumerate(data):
        examples.append(
            Example(
                example_id=idx,
                question=row["question"],
                claim=row.get("choice", row.get("claim", "")),
                label=int(row.get("label", row.get("gold", 0))),
                group_id=int(row.get("consistency_id", row.get("group", 0))),
                myth_label=(int(row["myth"]) if "myth" in row else None),
            )
        )
    return examples


