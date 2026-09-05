"""评测入口。用法：
  uv run python scripts/run_eval.py --tag base                 # 仅检索层（无 LLM 成本）
  uv run python scripts/run_eval.py --tag mem_off --with-llm   # 答案层 + 失败写入（不注入）
  uv run python scripts/run_eval.py --tag mem_on  --with-llm --memory on   # 注入失败记忆
  uv run python scripts/run_eval.py --tag judged  --with-llm --judge       # 叠加 GLM 跨源判分
  uv run python scripts/run_eval.py --tag quick --limit 5      # 快速抽测
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.runner import run_eval  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--with-llm", action="store_true", help="跑答案层（需 API key 或 DRY_RUN=1）")
    ap.add_argument("--memory", choices=["off", "on"], default="off")
    ap.add_argument("--judge", action="store_true", help="叠加 LLM-as-Judge（需 JUDGE_API_KEY）")
    ap.add_argument("--mode", choices=["react", "pae"], default="react",
                    help="agent 编排模式：react=工具循环 / pae=Plan-and-Execute")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    result = run_eval(args.tag, with_llm=args.with_llm, memory_on=args.memory == "on",
                      use_judge=args.judge, limit=args.limit, mode=args.mode)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
