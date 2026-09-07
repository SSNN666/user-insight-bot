"""CLI entry point: python -m eval [--suite qa|event|all] [--api URL]"""

import argparse
import sys

from eval.runner import EvalRunner
from eval.reporter import generate_report, save_report, load_previous_summary
from log.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="电商用户画像平台自动化评测")
    parser.add_argument("--suite", choices=["qa", "event", "all"], default="all", help="测试集 (default: all)")
    parser.add_argument("--api", default="http://localhost:8000", help="FastAPI 地址 (default: http://localhost:8000)")
    parser.add_argument("--compare", action="store_true", help="与上次评测结果对比")
    parser.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge 语义评测（需 LLM 可用）")
    args = parser.parse_args()

    runner = EvalRunner(api_url=args.api)
    summary = runner.run_all(include_judge=args.judge)

    prev = load_previous_summary() if args.compare else None
    if prev and prev.timestamp == summary.timestamp:
        prev = None  # Don't compare with self

    report = generate_report(summary, prev)
    path = save_report(report)

    print(report)
    print(f"\n报告已保存: {path}")
    print(f"结果已保存: eval/results/")

    # Exit code: 0 if all passed, 1 if tool_accuracy/data_fidelity < 0.3
    m = summary.metrics
    if m.tool_accuracy < 0.3 or m.data_fidelity < 0.3:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
