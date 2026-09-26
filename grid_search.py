import argparse
import csv
import itertools
import os
import queue
import shutil
import threading
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

# NOTE: FL-bench registers ResNet-18 as "res18" (src/utils/models.py).
# The old name "resnet18" raised a KeyError inside main.py, so every
# resnet run failed instantly.
MODELS = [
    "res18",
    "custom",
    "custom2",
]

# Only these dataset-model pairs are valid.
DATASET_MODELS = {
    "usps": ["custom2"],
    "mnist": ["custom2"],
    "fmnist": ["custom2"],
    "cifar10": ["res18", "custom"],
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

# Rough relative cost of one run per (dataset, model), in "minutes on a
# single RTX 4090-class GPU". Only used to order/balance jobs (longest
# first) and to print estimates; calibrate with --calibrate on your pod.
RUN_COST_ESTIMATE = {
    ("usps", "custom2"): 1.0,
    ("mnist", "custom2"): 5.0,
    ("fmnist", "custom2"): 5.0,
    ("cifar10", "custom"): 7.0,
    ("cifar10", "res18"): 14.0,
}


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


# def create_experiments(methods, datasets, models): 
#     """Create experiments using only valid dataset-model pairs."""

#     experiments = []

#     # Dataset and partition are the outer loops so a generated partition is
#     # consumed by all relevant runs before another partition overwrites it.
#     for dataset in datasets:
#         selected_models = [
#             model
#             for model in models
#             if model in DATASET_MODELS[dataset]
#         ]

#         for partition in PARTITIONS:
#             for method in methods:
#                 combinations = make_combinations(SEARCH_GRIDS[method])

#                 for model in selected_models:
#                     for hyperparameters in combinations:
#                         experiments.append(
#                             {
#                                 "dataset": dataset,
#                                 "partition": partition,
#                                 "method": method,
#                                 "model": model,
#                                 "hyperparameters": hyperparameters,
#                             }
#                         )

#     return experiments

def create_experiments(methods, datasets, models, partition_names):
    """Create experiments using only valid dataset-model pairs."""

    experiments = []

    selected_partitions = [
        partition
        for partition in PARTITIONS
        if partition["name"] in partition_names
    ]

    for dataset in datasets:
        selected_models = [
            model
            for model in models
            if model in DATASET_MODELS[dataset]
        ]

        for partition in selected_partitions:
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

def experiment_cost(experiment):
    return RUN_COST_ESTIMATE.get(
        (experiment["dataset"], experiment["model"]), 5.0
    )


def run_name(experiment):
    """Unique, readable W&B run name for one grid point."""

    hp = "_".join(
        f"{key.split('.')[-1]}{value}"
        for key, value in experiment["hyperparameters"].items()
    )
    return (
        f"{experiment['method']}_{experiment['dataset']}_{experiment['model']}_"
        f"{experiment['partition']['name']}_{hp}"
    )


def run_subprocess(command, cwd, environment, log_path):
    """Run a command, streaming to the terminal or to a log file."""

    if log_path is None:
        return subprocess.run(command, cwd=cwd, env=environment, check=False)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )


def generate_partition(
    project_root, dataset, partition, gpu, dry_run=False, log_path=None
):
    """Generate one FL-bench dataset partition inside project_root."""

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

    if log_path is None:
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

    result = run_subprocess(command, project_root, environment, log_path)

    if result.returncode != 0:
        print(
            "ERROR: Data generation failed for "
            f"{dataset}/{partition['name']} (in {project_root})."
        )

    return result.returncode


# =============================================================================
# Training
# =============================================================================

def build_command(project_root, experiment, extra_overrides=None):
    method = experiment["method"]
    configuration = {
        "method": method,
        "dataset.name": experiment["dataset"],
        "model.name": experiment["model"],
        **COMMON_CONFIG,
        **experiment["hyperparameters"],
        **(extra_overrides or {}),
    }
    command = [sys.executable, str(project_root / "main.py")]

    for key, value in configuration.items():
        # Method-specific config groups may not exist in defaults.yaml.
        # ++ adds the key if missing and overrides it if already present.
        if key.startswith(f"{method}."):
            override_key = f"++{key}"
        else:
            override_key = key
        command.append(f"{override_key}={hydra_value(value)}")

    return command


def run_experiment(
    project_root,
    experiment,
    run_number,
    total_runs,
    gpu,
    dry_run=False,
    log_path=None,
    extra_overrides=None,
    threads_per_run=None,
):
    """Invoke main.py for one experiment."""

    method = experiment["method"]
    partition_name = experiment["partition"]["name"]
    hyperparameters = experiment["hyperparameters"]
    command = build_command(project_root, experiment, extra_overrides)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["FLBENCH_RUN_NAME"] = run_name(experiment)
    environment["FLBENCH_RUN_GROUP"] = (
        f"{experiment['dataset']}_{experiment['model']}_{partition_name}"
    )
    environment["FLBENCH_RUN_TAGS"] = ",".join(
        [experiment["model"], partition_name]
    )
    if threads_per_run:
        # Stop concurrent runs from oversubscribing the CPU.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[var] = str(threads_per_run)

    if log_path is None:
        print("\n" + "=" * 80)
        print(f"RUN              : {run_number}/{total_runs}")
        print(f"STARTED          : {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"METHOD           : {method}")
        print(f"DATASET          : {experiment['dataset']}")
        print(f"DATA PARTITION   : {partition_name}")
        print(f"MODEL            : {experiment['model']}")
        print(f"GLOBAL EPOCHS    : {COMMON_CONFIG['common.global_epoch']}")
        print(f"LOCAL EPOCHS     : {COMMON_CONFIG['common.local_epoch']}")
        print(f"JOIN RATIO       : {COMMON_CONFIG['common.join_ratio']}")
        print(f"SEED             : {COMMON_CONFIG['common.seed']}")
        print(f"GPU              : {gpu}")
        print()
        print_dictionary("METHOD HYPERPARAMETERS:", hyperparameters)
        print(f"\nDATASET ALPHA    : {dataset_alpha(partition_name)}")
        print("\nFULL COMMAND:")
        print(" ".join(command))
        print("=" * 80, flush=True)

    if dry_run:
        return {"status": "dry-run", "return_code": 0, "duration_seconds": 0.0}

    start_time = time.perf_counter()
    result = run_subprocess(command, project_root, environment, log_path)
    duration = time.perf_counter() - start_time
    status = "success" if result.returncode == 0 else "failed"

    if log_path is None:
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
        help="Federated algorithms to run, space separated: --methods fedavg fedprox",
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
        help="CUDA device ID for serial mode (--workers 1).",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help=(
            "Comma-separated CUDA device IDs for parallel mode, e.g. 0,1,2,3. "
            "Workers are assigned round-robin. Defaults to --gpu."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of runs to execute concurrently. Each worker gets its own "
            "copy of the repo, because generate_data.py writes the partition "
            "to a fixed path (data/<dataset>/partition.pkl)."
        ),
    )
    parser.add_argument(
        "--threads-per-run",
        type=int,
        default=None,
        help="OMP/MKL threads per run. Default: vCPUs // workers (min 1).",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Where worker repo copies go. Default: <repo>/_workers",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the grid across this many machines (cost-balanced).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this machine runs (0-based).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Time one short run per (dataset, model) and print estimated "
            "GPU-hours for the selected grid, then exit."
        ),
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
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=[partition["name"] for partition in PARTITIONS],
        default=[partition["name"] for partition in PARTITIONS],
        help="Partitions to run.",
    )

    return parser


# =============================================================================
# Sharding and calibration
# =============================================================================

def select_shard(experiments, num_shards, shard_index):
    """Longest-processing-time-first assignment of runs to shards."""

    loads = [0.0] * num_shards
    assignment = [[] for _ in range(num_shards)]
    order = sorted(
        range(len(experiments)),
        key=lambda i: experiment_cost(experiments[i]),
        reverse=True,
    )
    for i in order:
        shard = loads.index(min(loads))
        loads[shard] += experiment_cost(experiments[i])
        assignment[shard].append(i)

    print("Estimated shard loads (4090-minutes):",
          [round(load) for load in loads])
    keep = sorted(assignment[shard_index])  # keep original grouping order
    return [experiments[i] for i in keep]


def calibrate(project_root, experiments, gpu):
    """Time 1- and 3-round runs per (dataset, model) and extrapolate."""

    samples = {}
    for experiment in experiments:
        key = (experiment["dataset"], experiment["model"])
        samples.setdefault(key, experiment)

    per_run_minutes = {}
    for key, experiment in samples.items():
        generate_partition(project_root, experiment["dataset"],
                           experiment["partition"], gpu)
        timings = []
        for rounds in (1, 3):
            outcome = run_experiment(
                project_root, experiment, 0, 0, gpu,
                log_path=project_root / "grid_logs" / "calibration.log",
                extra_overrides={
                    "common.global_epoch": rounds,
                    "common.monitor": None,
                },
            )
            if outcome["status"] != "success":
                print(f"Calibration run failed for {key}; see grid_logs/calibration.log")
                break
            timings.append(outcome["duration_seconds"])
        if len(timings) != 2:
            continue
        per_round = (timings[1] - timings[0]) / 2
        startup = timings[0] - per_round
        full = startup + per_round * COMMON_CONFIG["common.global_epoch"]
        per_run_minutes[key] = full / 60
        print(f"{key[0]:>8}/{key[1]:<8} startup {startup:6.1f}s | "
              f"per round {per_round:6.1f}s | full run ~{full / 60:6.1f} min")

    total = 0.0
    print("\nEstimated serial GPU time for the selected grid:")
    for key, minutes in per_run_minutes.items():
        count = sum(
            1 for e in experiments if (e["dataset"], e["model"]) == key
        )
        total += count * minutes
        print(f"  {key[0]:>8}/{key[1]:<8} {count:4d} runs x {minutes:5.1f} min "
              f"= {count * minutes / 60:6.1f} h")
    print(f"  TOTAL (one run at a time, one GPU): {total / 60:.1f} h")
    print("Concurrent runs per GPU cut this further for the small models; "
          "try --workers with a few --max-runs and compare durations in the CSV.")


# =============================================================================
# Parallel execution
# =============================================================================

def prepare_worker_dirs(project_root, workdir, num_workers):
    """Create one repo copy per worker (shares code, isolates data/ and out/)."""

    ignore = shutil.ignore_patterns(
        ".git", "_workers", "out", "outputs", "wandb", "grid_logs",
        "__pycache__", "grid_search_results_*.csv",
    )
    dirs = []
    for index in range(num_workers):
        target = workdir / f"w{index}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(project_root, target, ignore=ignore, symlinks=True)
        dirs.append(target)
    return dirs


def run_parallel(args, project_root, experiments, results_path):
    gpus = (args.gpus or args.gpu).split(",")
    workdir = Path(args.workdir) if args.workdir else project_root / "_workers"
    workdir.mkdir(parents=True, exist_ok=True)
    threads = args.threads_per_run or max(1, (os.cpu_count() or 1) // args.workers)
    log_dir = project_root / "grid_logs"
    total_runs = len(experiments)

    # Download raw datasets once in the main repo so the copies don't each
    # re-download them.
    if not args.dry_run:
        for dataset in sorted({e["dataset"] for e in experiments}):
            print(f"Pre-downloading / partitioning {dataset} once in main repo...")
            generate_partition(project_root, dataset, PARTITIONS[-1], gpus[0],
                               log_path=log_dir / "prepare.log")

    worker_dirs = prepare_worker_dirs(project_root, workdir, args.workers)
    print(f"Created {len(worker_dirs)} worker copies under {workdir}")
    print(f"GPUs: {gpus} | threads per run: {threads} | logs: {log_dir}")

    # Longest jobs first so the tail of the schedule is short jobs.
    jobs = queue.Queue()
    # Ties are kept in (dataset, partition) order, which cuts down on how
    # often a worker has to regenerate its partition.
    for number, experiment in sorted(
        enumerate(experiments, start=1),
        key=lambda item: (-experiment_cost(item[1]), item[0]),
    ):
        jobs.put((number, experiment))

    lock = threading.Lock()
    stop = threading.Event()
    counts = {"success": 0, "failed": 0, "skipped": 0, "done": 0}

    def worker(index):
        root = worker_dirs[index]
        gpu = gpus[index % len(gpus)]
        current_partition = None
        while not stop.is_set():
            try:
                number, experiment = jobs.get_nowait()
            except queue.Empty:
                return
            key = (experiment["dataset"], experiment["partition"]["name"])
            log_path = log_dir / f"run_{number:04d}_{run_name(experiment)}.log"

            if key != current_partition:
                code = generate_partition(
                    root, experiment["dataset"], experiment["partition"], gpu,
                    dry_run=args.dry_run, log_path=log_path,
                )
                current_partition = key if code == 0 else None
            else:
                code = 0

            if code != 0:
                outcome = {"status": "skipped", "return_code": -1,
                           "duration_seconds": 0.0}
            else:
                outcome = run_experiment(
                    root, experiment, number, total_runs, gpu,
                    dry_run=args.dry_run, log_path=log_path,
                    threads_per_run=threads,
                )

            with lock:
                status = outcome["status"]
                counts["success" if status in {"success", "dry-run"} else
                       "failed" if status == "failed" else "skipped"] += 1
                counts["done"] += 1
                append_result(results_path,
                              make_result_record(number, experiment, outcome))
                print(
                    f"[{datetime.now():%H:%M:%S}] {counts['done']}/{total_runs} "
                    f"w{index}/gpu{gpu} {status:8s} "
                    f"{outcome['duration_seconds'] / 60:6.1f} min  "
                    f"{run_name(experiment)}",
                    flush=True,
                )
                if status != "success" and status != "dry-run" and args.stop_on_error:
                    stop.set()

    threads_list = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(args.workers)
    ]
    for thread in threads_list:
        thread.start()
    try:
        for thread in threads_list:
            while thread.is_alive():
                thread.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\nInterrupted: letting in-flight runs finish, no new runs will start.")
        stop.set()
        for thread in threads_list:
            thread.join()

    return counts["success"], counts["failed"], counts["skipped"]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = create_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    if not (project_root / "main.py").is_file():
        parser.error(f"Could not find main.py in {project_root}")
    if not (project_root / "generate_data.py").is_file():
        parser.error(f"Could not find generate_data.py in {project_root}")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    # experiments = create_experiments(
    #     methods=args.methods,
    #     datasets=args.datasets,
    #     models=args.models,
    # )
    experiments = create_experiments(
        methods=args.methods,
        datasets=args.datasets,
        models=args.models,
        partition_names=args.partitions,
    )

    if not experiments:
        parser.error(
            "No valid experiments were produced. Check --datasets and "
            "--models against DATASET_MODELS."
        )

    if args.calibrate:
        calibrate(project_root, experiments, args.gpu)
        return

    if args.num_shards > 1:
        experiments = select_shard(experiments, args.num_shards, args.shard_index)

    if args.max_runs is not None:
        if args.max_runs <= 0:
            parser.error("--max-runs must be greater than zero.")
        experiments = experiments[: args.max_runs]

    total_runs = len(experiments)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    suffix = f"_shard{args.shard_index}" if args.num_shards > 1 else ""
    results_path = project_root / f"grid_search_results_{timestamp}{suffix}.csv"
    estimated = sum(experiment_cost(e) for e in experiments) / 60

    print("\n" + "=" * 80)
    print("FL-BENCH GRID SEARCH")
    print(f"Methods          : {args.methods}")
    print(f"Datasets         : {args.datasets}")
    print(f"Requested models : {args.models}")
    print(f"Dataset mapping  : {DATASET_MODELS}")
    # print(f"Partitions       : {[item['name'] for item in PARTITIONS]}")
    print(f"Partitions       : {args.partitions}")
    print(f"Global epochs    : {COMMON_CONFIG['common.global_epoch']}")
    print(f"Local epochs     : {COMMON_CONFIG['common.local_epoch']}")
    print(f"Number of clients: {CLIENT_NUM}")
    print(f"Shard            : {args.shard_index + 1}/{args.num_shards}")
    print(f"Total runs       : {total_runs}")
    print(f"Rough estimate   : {estimated:.1f} serial 4090-hours")
    print(f"Workers          : {args.workers}")
    print(f"GPU(s)           : {args.gpus or args.gpu}")
    print(f"Dry run          : {args.dry_run}")
    print(f"Results CSV      : {results_path}")
    print("=" * 80)

    if args.workers > 1:
        successful_runs, failed_runs, skipped_runs = run_parallel(
            args, project_root, experiments, results_path
        )
    else:
        successful_runs, failed_runs, skipped_runs = run_serial(
            args, project_root, experiments, results_path
        )

    print("\n" + "=" * 80)
    print("GRID SEARCH FINISHED")
    print(f"Successful/dry runs: {successful_runs}")
    print(f"Failed runs        : {failed_runs}")
    print(f"Skipped runs       : {skipped_runs}")
    print(f"Results CSV        : {results_path}")
    if args.workers > 1:
        print("Per-run logs       : grid_logs/ ; FL-bench outputs: _workers/w*/out/")
    print("=" * 80)


def run_serial(args, project_root, experiments, results_path):
    total_runs = len(experiments)
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

    return successful_runs, failed_runs, skipped_runs


if __name__ == "__main__":
    main()
