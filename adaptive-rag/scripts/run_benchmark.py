"""
Run the baseline-vs-AdaptiveRAG benchmark against a real chat's documents.

Usage:
    python -m scripts.run_benchmark --chat-id <uuid> --user-id <uuid> \\
        --questions "What is X?" "How does Y compare to Z?"

Or edit QUESTIONS below and run it with no arguments for a quick manual check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark import run_benchmark_suite  # noqa: E402

QUESTIONS = [
    "What is the main topic of these documents?",
    "Summarize the key points in a few sentences.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline RAG vs AdaptiveRAG.")
    parser.add_argument("--chat-id", required=True, help="Chat id whose documents to query.")
    parser.add_argument("--user-id", required=True, help="Owning user id.")
    parser.add_argument("--questions", nargs="*", default=None, help="Questions to ask (optional).")
    args = parser.parse_args()

    questions = args.questions or QUESTIONS
    report = run_benchmark_suite(questions, chat_id=args.chat_id, user_id=args.user_id)

    print(json.dumps(report["summary"], indent=2))
    print("\nFull per-question results written to benchmark_results.json")
    Path("benchmark_results.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
