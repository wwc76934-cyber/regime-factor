import argparse
from pathlib import Path

import build_verified_market_state_ridge
import compare_verified_market_state_ridge
import evaluate_all_factor_regimes
import select_regime_factors


def main():
    parser = argparse.ArgumentParser(description="Run factor screening, verified Ridge and execution ablations.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--breadth-mode", choices=["auto", "raw", "residual"], default="auto")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output_dir or root / "reports/verified_factor_pipeline").resolve()
    for reserved in [root, root / "configs", root / "data", root / "reports"]:
        if output == reserved:
            parser.error("Choose a dedicated output directory")
    evaluate_all_factor_regimes.run(root, output)
    select_regime_factors.run(root, output)
    build_verified_market_state_ridge.run(root, output, args.breadth_mode)
    compare_verified_market_state_ridge.run(root, output)
    print(f"\nComplete: {output}", flush=True)


if __name__ == "__main__":
    main()
