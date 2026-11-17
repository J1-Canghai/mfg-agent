"""Retrieval evaluation on data/eval_questions.jsonl.

Each question is labeled with the manual section that answers it. The questions
are worded differently from the manuals on purpose. Reports hit@1, hit@3 and MRR.

Usage: python eval_rag.py
"""
import json
from pathlib import Path

from rag import get_index

BASE = Path(__file__).parent
EVAL_FILE = BASE / "data" / "eval_questions.jsonl"
K = 5


def rank_of_answer(hits, item):
    for rank, h in enumerate(hits, start=1):
        if h["source"] == item["source"] and h["section"] == item["section"]:
            return rank
    return None


def main():
    with EVAL_FILE.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    index = get_index()
    ranks = []
    for item in items:
        hits = index.search(item["question"], k=K)
        rank = rank_of_answer(hits, item)
        ranks.append(rank)
        mark = "OK  " if rank == 1 else ("TOP5" if rank else "MISS")
        print(f"[{mark}] rank {rank or '-'}  {item['question']}")
        if rank != 1:
            print(f"        expected: {item['source']} · {item['section']}; got: {hits[0]['source']} · {hits[0]['section']}")

    n = len(ranks)
    hit1 = sum(r == 1 for r in ranks) / n
    hit3 = sum(r is not None and r <= 3 for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"\n{n} questions  hit@1 = {hit1:.0%}  hit@3 = {hit3:.0%}  MRR = {mrr:.2f}")


if __name__ == "__main__":
    main()
