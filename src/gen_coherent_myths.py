import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def examples_astrology(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Does Mercury retrograde cause more communication problems?",
         "During Mercury retrograde, communication fails more often."),
        ("Can birth month predict conscientiousness?",
         "Capricorns are more conscientious than other signs."),
        ("Are Aries naturally more aggressive?",
         "Aries tend to be more aggressive due to their sign."),
        ("Does your sun sign determine your core personality?",
         "Your sun sign determines your core personality traits."),
    ]
    truths = [
        ("Can horoscopes sometimes give helpful general advice?",
         "Horoscopes sometimes give broadly applicable advice like planning ahead."),
        ("Can two people with any signs communicate effectively?",
         "Good communication depends on skills and context, not star signs."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 1001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 1001, "myth": 0})
    return out


def examples_homeopathy(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Can homeopathic dilutions treat bacterial infections?",
         "A 30C homeopathic remedy can kill bacteria."),
        ("Do juice cleanses remove toxins from the liver?",
         "Three-day juice fasts detoxify your liver."),
        ("Does dilution increase potency in homeopathy?",
         "More dilution makes the remedy more potent."),
        ("Can essential oils cure bacterial infections because they are natural?",
         "Natural essential oils can cure most bacterial infections."),
    ]
    truths = [
        ("Are vaccines tested in randomized trials?",
         "Vaccines are approved after randomized controlled trials."),
        ("Do homeopathic dilutions beyond Avogadro's number contain active molecules?",
         "High dilutions likely contain no active molecules."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 2001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 2001, "myth": 0})
    return out


def examples_pop_psych(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Are polygraphs accurate lie detectors?", "Polygraphs can reliably detect lies."),
        ("Do eye movements reveal lying reliably?", "Looking left signals lying."),
        ("Do matched learning styles improve learning outcomes?", "Teaching to VARK learning styles improves learning."),
    ]
    truths = [
        ("Can randomized experiments test psychology interventions?",
         "Randomized experiments can test whether an intervention works."),
        ("Do polygraphs have known error rates?",
         "Polygraphs have error rates and are not fully reliable."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 3001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 3001, "myth": 0})
    return out


def examples_misconceptions(n_pairs: int, rng: random.Random) -> List[Dict]:
    myths = [
        ("Do people use only 10% of their brain?", "Humans use just 10% of their brain capacity."),
        ("Was Napoleon unusually short?", "Napoleon Bonaparte was significantly shorter than average."),
        ("Does sugar cause hyperactivity in children?", "Sugar causes hyperactivity."),
    ]
    truths = [
        ("Do humans have many senses beyond five?",
         "Humans have more than five senses (e.g., proprioception, nociception)."),
        ("Is historical height context important for Napoleon?",
         "Napoleon's height was near average for his era."),
    ]
    out: List[Dict] = []
    for _ in range(n_pairs):
        q_m, c_m = rng.choice(myths)
        q_t, c_t = rng.choice(truths)
        out.append({"question": q_m, "choice": c_m, "label": 0, "consistency_id": 4001, "myth": 1})
        out.append({"question": q_t, "choice": c_t, "label": 1, "consistency_id": 4001, "myth": 0})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Coherent Myths dataset")
    parser.add_argument("--pairs_per_domain", type=int, default=12, help="myth/true pairs per domain")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = (
        examples_astrology(args.pairs_per_domain, rng)
        + examples_homeopathy(args.pairs_per_domain, rng)
        + examples_pop_psych(args.pairs_per_domain, rng)
        + examples_misconceptions(args.pairs_per_domain, rng)
    )
    rng.shuffle(data)
    # Simple split
    n = len(data)
    n_train = int(0.7 * n)
    train = data[:n_train]
    test = data[n_train:]
    Path("data/coherent_myths_train.json").write_text(json.dumps(train, indent=2))
    Path("data/coherent_myths_test.json").write_text(json.dumps(test, indent=2))
    print(f"Wrote {len(train)} train and {len(test)} test items.")


if __name__ == "__main__":
    main()


