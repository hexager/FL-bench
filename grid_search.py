import argparse
import csv
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# =============================================================================
# Experiment setup
# =============================================================================

METHODS = [
    "fedavg",
    "fedprox",
    "feddyn",
    "scaffold",
    "elastic",
]

DATASETS = [
    "usps",
    "mnist",
    "fmnist",
    "cifar10",
]

MODELS = [
    "resnet18",
    "custom",
    "custom2",
]

# Only these dataset-model pairs are valid.
DATASET_MODELS = {
    "usps": ["custom2"],
    "mnist": ["custom2"],
    "fmnist": ["custom2"],
    "cifar10": ["resnet18", "custom"],
}


# Dataset alpha and feddyn.alpha are different:
#   - Dataset alpha controls Dirichlet data heterogeneity.
#   - feddyn.alpha is FedDyn's dynamic-regularization parameter.
PARTITIONS = [
    {
        "name": "dirichlet_0.1",
        "generator_args": ["-a", "0.1"],
    },
    {
        "name": "dirichlet_0.05",
        "generator_args": ["-a", "0.05"],
    },
    {
        "name": "dirichlet_1.0",
        "generator_args": ["-a", "1.0"],
    },
    {
        "name": "iid",
        "generator_args": ["--iid", "1.0"],
    },
]


# Common configuration applied to every main.py invocation.
COMMON_CONFIG = {
    "common.monitor": "wandb",
    "common.test.server.test": True,
    "common.test.server.interval": 1,
    "common.join_ratio": 1.0,
    "common.global_epoch": 25,
    "common.local_epoch": 5,
    "common.seed": 42,
    "model.use_torchvision_pretrained_weights": False,
}

# Number of federated clients used during data generation.
CLIENT_NUM = 16


# =============================================================================
# Hyperparameter grids
# =============================================================================

# The common local learning-rate grid is searched for every method.
# Method-specific settings follow the selected paper-aligned ranges:
#   - FedProx: include 1e-4 through 1 because 1e-4 also appears in later
#     comparative experiments, while 1e-3 through 1 is the original paper grid.
#   - FedDyn: alpha grid used for the paper's CIFAR experiments.
#   - SCAFFOLD: global server learning rate fixed to the paper/default value 1.
#   - Elastic: tau, mu, and sample_ratio are fixed implementation settings.
SEARCH_GRIDS = {
    "fedavg": {
        "optimizer.lr": [0.01, 0.05, 0.1],
    },
    "fedprox": {
        "optimizer.lr": [0.01, 0.05, 0.1],
        "fedprox.mu": [
            0.0001,
            0.001,
            0.01,
            0.1,
            1.0,
        ],
    },
    "feddyn": {
        "optimizer.lr": [0.01, 0.05, 0.1],
        "feddyn.alpha": [
            0.001,
            0.01,
            0.1,
        ],
        # Fixed numerical-stability setting, not an algorithmic search axis.
        "feddyn.max_grad_norm": [10.0],
    },
    "scaffold": {
        "optimizer.lr": [0.01, 0.05, 0.1],
        "scaffold.global_lr": [1.0],
    },
    "elastic": {
        "optimizer.lr": [0.01, 0.05, 0.1],
        "elastic.tau": [0.5],
        "elastic.mu": [0.95],
        "elastic.sample_ratio": [0.3],
    },
}


# =============================================================================
# Utility functions
# =============================================================================

def hydra_value(value):
    """Convert a Python value to a Hydra-compatible CLI value."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def make_combinations(parameter_grid):
    """Return every combination in a hyperparameter grid."""

    keys = list(parameter_grid.keys())
    value_lists = list(parameter_grid.values())

    return [
        dict(zip(keys, values))
        for values in itertools.product(*value_lists)
    ]


def create_experiments(methods, datasets, models):
    """Create experiments using only valid dataset-model pairs."""

    experiments = []

    # Dataset and partition are the outer loops so a generated partition is
    # consumed by all relevant runs before another partition overwrites it.
    for dataset in datasets:
        selected_models = [
            model
            for model in models
            if model in DATASET_MODELS[dataset]
        ]

        for partition in PARTITIONS:
            for method in methods:
                combinations = make_combinations(SEARCH_GRIDS[method])

                for model in selected_models:
                    for hyperparameters in combinations:
                        experiments.append(
                            {
                                "dataset": dataset,
                                "partition": partition,
                                "method": method,
                                "model": model,
                                "hyperparameters": hyperparameters,
                            }
                        )

    return experiments


def print_dictionary(title, values):
    """Print configuration values consistently."""

    print(title)
    for key, value in values.items():
        print(f"  {key:<45} = {value}")


def dataset_alpha(partition_name):
    """Return the human-readable dataset-partition alpha."""

    if partition_name.startswith("dirichlet_"):
        return partition_name.removeprefix("dirichlet_")
    return "IID"


# =============================================================================
# Dataset generation
# =============================================================================

def generate_partition(project_root, dataset, partition, gpu, dry_run=False):
    """Generate one FL-bench dataset partition."""

    command = [
        sys.executable,
        str(project_root / "generate_data.py"),
        "-d",
        dataset,
        "-cn",
        str(CLIENT_NUM),
        *partition["generator_args"],
    ]

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print("\n" + "#" * 80)
    print("GENERATING DATASET PARTITION")
    print(f"Dataset          : {dataset}")
    print(f"Partition        : {partition['name']}")
    print(f"Number of clients: {CLIENT_NUM}")
    print("Command:")
    print(" ".join(command))
    print("#" * 80, flush=True)

    if dry_run:
        return 0

    result = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
    )

    if result.returncode != 0:
        print(
            "ERROR: Data generation failed for "
            f"{dataset}/{partition['name']}."
        )

    return result.returncode


# =============================================================================
# Training
# =============================================================================

def run_experiment(
    project_root,
    experiment,
    run_number,
    total_runs,
    gpu,
    dry_run=False,
):
    """Invoke main.py for one experiment."""

    method = experiment["method"]
    dataset = experiment["dataset"]
    partition_name = experiment["partition"]["name"]
    model = experiment["model"]
    hyperparameters = experiment["hyperparameters"]

    configuration = {
        "method": method,
        "dataset.name": dataset,
        "model.name": model,
        **COMMON_CONFIG,
        **hyperparameters,
    }

    command = [
        sys.executable,
        str(project_root / "main.py"),
        *[
            f"{key}={hydra_value(value)}"
            for key, value in configuration.items()
        ],
    ]

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print("\n" + "=" * 80)
    print(f"RUN              : {run_number}/{total_runs}")
    print(f"STARTED          : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"METHOD           : {method}")
    print(f"DATASET          : {dataset}")
    print(f"DATA PARTITION   : {partition_name}")
    print(f"MODEL            : {model}")
    print(f"GLOBAL EPOCHS    : {COMMON_CONFIG['common.global_epoch']}")
    print(f"LOCAL EPOCHS     : {COMMON_CONFIG['common.local_epoch']}")
    print(f"JOIN RATIO       : {COMMON_CONFIG['common.join_ratio']}")
    print(f"SEED             : {COMMON_CONFIG['common.seed']}")
    print(f"GPU              : {gpu}")
    print()
    print_dictionary("METHOD HYPERPARAMETERS:", hyperparameters)
    print(f"\nDATASET ALPHA    : {dataset_alpha(partition_name)}")

    if method == "feddyn":
        print(f"FEDDYN ALPHA     : {hyperparameters['feddyn.alpha']}")

    print("\nFULL COMMAND:")
    print(" ".join(command))
    print("=" * 80, flush=True)

    if dry_run:
        return {
            "status": "dry-run",
            "return_code": 0,
            "duration_seconds": 0.0,
        }

    start_time = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
    )
    duration = time.perf_counter() - start_time
    status = "success" if result.returncode == 0 else "failed"

    print(
        f"\nRUN FINISHED: {status} | "
        f"return_code={result.returncode} | "
        f"duration={duration:.2f}s",
        flush=True,
    )

    return {
        "status": status,
        "return_code": result.returncode,
        "duration_seconds": duration,
    }


# =============================================================================
# Results CSV
# =============================================================================

RESULT_FIELDS = [
    "run",
    "method",
    "dataset",
    "partition",
    "dataset_alpha",
    "model",
    "global_epoch",
    "local_epoch",
    "join_ratio",
    "seed",
    "hyperparameters",
    "status",
    "return_code",
    "duration_seconds",
    "finished_at",
]


def append_result(results_path, record):
    """Append one experiment result to the CSV file."""

    file_exists = results_path.exists()

    with results_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def make_result_record(run_number, experiment, outcome):
    """Create a CSV record for one experiment."""

    partition_name = experiment["partition"]["name"]

    return {
        "run": run_number,
        "method": experiment["method"],
        "dataset": experiment["dataset"],
        "partition": partition_name,
        "dataset_alpha": dataset_alpha(partition_name),
        "model": experiment["model"],
        "global_epoch": COMMON_CONFIG["common.global_epoch"],
        "local_epoch": COMMON_CONFIG["common.local_epoch"],
        "join_ratio": COMMON_CONFIG["common.join_ratio"],
        "seed": COMMON_CONFIG["common.seed"],
        "hyperparameters": repr(experiment["hyperparameters"]),
        "status": outcome["status"],
        "return_code": outcome["return_code"],
        "duration_seconds": round(outcome["duration_seconds"], 2),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


# =============================================================================
# Argument parser
# =============================================================================

def create_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run FL-bench grid searches across algorithms, datasets, "
            "partitions, valid models, and hyperparameters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Federated algorithms to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=DATASETS,
        help="Datasets to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
        help=(
            "Models to allow. Invalid dataset-model pairs are automatically "
            "filtered using DATASET_MODELS."
        ),
    )
    parser.add_argument(
        "--gpu",
        default="0",
        help="CUDA device ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without executing them.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit execution to the first N training runs.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when data generation or training fails.",
    )

    return parser


# =============================================================================
# Main
# =============================================================================

def main():
    parser = create_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    main_file = project_root / "main.py"
    generator_file = project_root / "generate_data.py"

    if not main_file.is_file():
        parser.error(f"Could not find main.py in {project_root}")
    if not generator_file.is_file():
        parser.error(f"Could not find generate_data.py in {project_root}")

    experiments = create_experiments(
        methods=args.methods,
        datasets=args.datasets,
        models=args.models,
    )

    if not experiments:
        parser.error(
            "No valid experiments were produced. Check --datasets and "
            "--models against DATASET_MODELS."
        )

    if args.max_runs is not None:
        if args.max_runs <= 0:
            parser.error("--max-runs must be greater than zero.")
        experiments = experiments[: args.max_runs]

    total_runs = len(experiments)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_path = project_root / f"grid_search_results_{timestamp}.csv"

    print("\n" + "=" * 80)
    print("FL-BENCH GRID SEARCH")
    print(f"Methods          : {args.methods}")
    print(f"Datasets         : {args.datasets}")
    print(f"Requested models : {args.models}")
    print(f"Dataset mapping  : {DATASET_MODELS}")
    print(f"Partitions       : {[item['name'] for item in PARTITIONS]}")
    print(f"Global epochs    : {COMMON_CONFIG['common.global_epoch']}")
    print(f"Local epochs     : {COMMON_CONFIG['common.local_epoch']}")
    print(f"Number of clients: {CLIENT_NUM}")
    print(f"Total runs       : {total_runs}")
    print(f"GPU              : {args.gpu}")
    print(f"Dry run          : {args.dry_run}")
    print(f"Results CSV      : {results_path}")
    print("=" * 80)

    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0
    current_partition_key = None
    unusable_partitions = set()

    try:
        for run_number, experiment in enumerate(experiments, start=1):
            dataset = experiment["dataset"]
            partition = experiment["partition"]
            partition_key = (dataset, partition["name"])

            # Generate a partition once immediately before all runs that use it.
            if partition_key != current_partition_key:
                partition_return_code = generate_partition(
                    project_root=project_root,
                    dataset=dataset,
                    partition=partition,
                    gpu=args.gpu,
                    dry_run=args.dry_run,
                )
                current_partition_key = partition_key

                if partition_return_code != 0:
                    unusable_partitions.add(partition_key)
                    if args.stop_on_error:
                        print("\nStopping because partition generation failed.")
                        break

            if partition_key in unusable_partitions:
                print(
                    f"\nSkipping run {run_number}/{total_runs}: "
                    "partition generation failed for "
                    f"{dataset}/{partition['name']}."
                )
                outcome = {
                    "status": "skipped",
                    "return_code": -1,
                    "duration_seconds": 0.0,
                }
                skipped_runs += 1
            else:
                outcome = run_experiment(
                    project_root=project_root,
                    experiment=experiment,
                    run_number=run_number,
                    total_runs=total_runs,
                    gpu=args.gpu,
                    dry_run=args.dry_run,
                )

                if outcome["status"] in {"success", "dry-run"}:
                    successful_runs += 1
                else:
                    failed_runs += 1

            append_result(
                results_path=results_path,
                record=make_result_record(
                    run_number=run_number,
                    experiment=experiment,
                    outcome=outcome,
                ),
            )

            if outcome["status"] == "failed" and args.stop_on_error:
                print("\nStopping because a training run failed.")
                break

    except KeyboardInterrupt:
        print("\n\nGrid search interrupted by the user.")

    print("\n" + "=" * 80)
    print("GRID SEARCH FINISHED")
    print(f"Successful/dry runs: {successful_runs}")
    print(f"Failed runs        : {failed_runs}")
    print(f"Skipped runs       : {skipped_runs}")
    print(f"Results CSV        : {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
