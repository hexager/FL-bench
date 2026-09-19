import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Common base configuration
COMMON_CONFIG = {
    "dataset.name": "cifar10",  # [mnist, cifar10, cifar100, etc.]
    "model.name": "lenet5",     # [lenet5, resnet18, etc.]
    "common.global_epoch": 50,
    "common.local_epoch": 5,
    "common.batch_size": 32,
    "common.seed": 42,
    "common.use_cuda": "true",
}

# Grid definitions for each method
SEARCH_GRIDS = {
    "elastic": {
        "optimizer.lr": [0.005, 0.01, 0.05],          # Client LR
        "elastic.tau": [0.1, 0.5, 1.0],
        "elastic.mu": [0.8, 0.95],
        "elastic.sample_ratio": [0.3],                # Sample ratio for sensitivity
    },
    "fedprox": {
        "optimizer.lr": [0.005, 0.01, 0.05],          # Client LR
        "fedprox.mu": [0.001, 0.01, 0.1, 1.0],        # Proximal weight
    },
    "feddyn": {
        "optimizer.lr": [0.005, 0.01, 0.05],          # Client LR
        "feddyn.alpha": [0.01, 0.1, 0.5],             # Dynamic reg parameter
        "feddyn.max_grad_norm": [1.0, 10.0],          # Grad clipping
    },
    "scaffold": {
        "optimizer.lr": [0.005, 0.01, 0.05],          # Client LR (eta_l)
        "scaffold.global_lr": [0.5, 1.0, 1.5],        # Server LR (eta_g)
    },
}


def run_experiment(method: str, params: dict, common: dict, gpu_id: str = "0"):
    """Run a single FL-bench experiment via Hydra CLI override."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [sys.executable, "main.py", f"method={method}"]

    # Add common args
    for k, v in common.items():
        cmd.append(f"{k}={v}")

    # Add method-specific args
    for k, v in params.items():
        cmd.append(f"{k}={v}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def run_grid_search(methods=None, gpu_id="0"):
    if methods is None:
        methods = list(SEARCH_GRIDS.keys())
    elif isinstance(methods, str):
        methods = [methods]

    for method in methods:
        if method not in SEARCH_GRIDS:
            print(f"Unknown method '{method}', skipping.")
            continue

        param_grid = SEARCH_GRIDS[method]
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"\n=======================================================")
        print(f" Starting Grid Search for '{method}' ({len(combinations)} runs)")
        print(f"=======================================================")

        for idx, combination in enumerate(combinations, 1):
            print(f"\n--- Run {idx}/{len(combinations)} for {method} ---")
            for k, v in combination.items():
                print(f"  {k}: {v}")

            return_code = run_experiment(method, combination, COMMON_CONFIG, gpu_id=gpu_id)
            if return_code != 0:
                print(f"⚠️ Run {idx} failed with code {return_code}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FL-bench Grid Search")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["elastic", "fedprox", "feddyn", "scaffold"],
        choices=["elastic", "fedprox", "feddyn", "scaffold"],
        help="Methods to include in grid search",
    )
    parser.add_argument("--gpu", type=str, default="0", help="CUDA GPU device ID")
    args = parser.parse_args()

    run_grid_search(methods=args.methods, gpu_id=args.gpu)
