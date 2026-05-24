"""ar-bench: run the retrospective benchmark and report ranking accuracy."""
from __future__ import annotations
import json
from pathlib import Path
import click
from frozen.benchmark_runner import run_benchmark


@click.command()
@click.option("--verbose", is_flag=True, default=False)
def main(verbose: bool):
    """Run the retrospective benchmark (analog of check_results.py benchmark mode)."""
    result = run_benchmark(verbose=verbose)
    print(f"\nBenchmark Result:")
    print(f"  Engine rank accuracy : {result['rank_accuracy']:.3f}")
    print(f"  Baseline (random)    : {result['baseline_accuracy']:.3f}")
    print(f"  Improvement          : {result['rank_accuracy'] - result['baseline_accuracy']:+.3f}")
    print(f"  Interpretation       : {result['interpretation']}")
    if verbose:
        print(f"\n  Detailed rankings:")
        for item in result["rankings"]:
            print(f"    {item['name']:<35} score={item['score']:.3f}  label={item['label']}")


if __name__ == "__main__":
    main()
