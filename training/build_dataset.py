"""Instruction dataset for fine-tuning a small open coder model.

    python -m training.build_dataset            # -> training/data/{train,val}.jsonl

Each example is the *exact* prompt the Coder sends at inference
(`coder_agent._build_sql_prompt`: DataSource rendering, dialect notes,
PII masking, injection hardening) and the gold SQL as the answer, in
chat format:

    {"id": ..., "db_id": ..., "messages": [{"role": "user", "content": <prompt>},
                                            {"role": "assistant", "content": <gold SQL>}]}

Source: Spider train (train_spider.json + train_others.json, 8,659
questions over 146 databases). Never dev: evaluation is on Spider dev, so
any dev database appearing here is a hard error. Validation holds out
whole databases (not questions), so it measures generalization to unseen
schemas, which is what dev tests.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPIDER = ROOT / "benchmarks" / "spider"
OUT = Path(__file__).resolve().parent / "data"


class LeakageError(RuntimeError):
    pass


def load_train() -> list[dict]:
    rows = []
    for name in ("train_spider.json", "train_others.json"):
        tag = name.split("_")[1].split(".")[0]
        rows += [{**d, "id": f"spider-train-{tag}-{i}"} for i, d in enumerate(json.loads((SPIDER / name).read_text()))]
    return rows


def build(val_fraction: float = 0.05, seed: int = 0) -> tuple[list[dict], list[dict]]:
    from agents.coder_agent import _build_sql_prompt
    from core.datasource import DataSource

    dev_dbs = {d["db_id"] for d in json.loads((SPIDER / "dev.json").read_text())}
    train = load_train()
    leaked = {d["db_id"] for d in train} & dev_dbs
    if leaked:
        raise LeakageError(f"train databases overlap Spider dev: {sorted(leaked)[:5]}")

    dbs = sorted({d["db_id"] for d in train})
    rng = random.Random(seed)
    val_dbs = set(rng.sample(dbs, max(1, round(len(dbs) * val_fraction))))
    schemas: dict[str, DataSource] = {}
    out = {"train": [], "val": []}
    for d in train:
        if d["db_id"] not in schemas:
            path = SPIDER / "database" / d["db_id"] / f"{d['db_id']}.sqlite"
            schemas[d["db_id"]] = DataSource.from_sqlite(str(path), name=d["db_id"])
        prompt = _build_sql_prompt(d["question"], schemas[d["db_id"]])
        ex = {"id": d["id"], "db_id": d["db_id"], "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": d["query"].strip()},
        ]}
        out["val" if d["db_id"] in val_dbs else "train"].append(ex)
    return out["train"], out["val"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    train, val = build(args.val_fraction, args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    lens = sorted(len(r["messages"][0]["content"]) for r in train)
    print(f"train {len(train)} examples ({len({r['db_id'] for r in train})} dbs), "
          f"val {len(val)} ({len({r['db_id'] for r in val})} dbs); "
          f"prompt chars p50={lens[len(lens) // 2]} p95={lens[int(len(lens) * .95)]} max={lens[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
