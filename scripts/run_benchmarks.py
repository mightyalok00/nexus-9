"""CLI script to run full benchmark suite including Causal Discovery & Models."""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from futurebench.evaluation.benchmark_runner import run_complete_benchmark

def main():
    print("=== NEXUS-9 Full FutureBench Benchmark Suite ===")
    results = run_complete_benchmark()

    print()
    print("1. Causal Discovery Metrics (vs Ground Truth):")
    for k, v in results["causal_discovery"].items():
        print(f"  {k:20s}: {v}")

    print()
    print("2. Model Arena Summary:")
    for m in results["model_arena"]:
        print(f"  {m['model']:25s} | Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f} | AUC: {m.get('roc_auc', 0.0):.4f}")

    out_file = Path("reports/benchmark/benchmark_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Benchmark suite output written to {out_file}")

if __name__ == "__main__":
    main()
