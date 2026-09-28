"""Review thumbs-down answers and promote fixed ones to golden cases.

    python -m feedback.review list
    python -m feedback.review promote <run_id> --sql 'SELECT ...' [--difficulty hard --tags join,ranking]
    python -m feedback.review stats
"""

import argparse
import json
import sys

import feedback


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("stats")
    p = sub.add_parser("promote")
    p.add_argument("run_id")
    p.add_argument("--sql", required=True)
    p.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--tags", default="from_feedback")
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for r in feedback.review_queue():
            print(f"{r['run_id']}  [{r['datasource']}]  {r['question']}\n    sql: {r['sql']}\n    comment: {r['comment']}")
    elif args.cmd == "stats":
        print(json.dumps(feedback.stats(), indent=2))
    else:
        case = feedback.promote(args.run_id, args.sql, args.difficulty, args.tags.split(","))
        print(f"added {case['id']} to benchmarks/golden_user.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
