#!/usr/bin/env python

"""Run a configured monolithic inference experiment."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import time
import traceback
from collections.abc import Iterable
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from threading import get_ident
from typing import Any, TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MARKER = "SCBOLT_MONOLITHIC_RUNTIME"
REQUIRED_PARAMETERS = frozenset(
    {
        "name",
        "config",
        "timeout",
        "clingo_mode",
        "clingo_configuration",
        "clingo_opt_strategy",
        "clingo_parallel_mode",
        "max_clauses",
        "prior_knowledge",
        "organism",
        "geneinfo_version",
        "omnipath_version",
        "hcop_version",
        "dorothea_api",
        "dorothea_compatibility",
        "dorothea_levels",
    }
)
STRING_PARAMETERS = REQUIRED_PARAMETERS - {
    "dorothea_compatibility",
    "dorothea_levels",
    "max_clauses",
}

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/scbolt-mpl-cache")


def _default_scbolt_home() -> Path:
    return Path(__file__).resolve().parents[3] / "scbolt"


def _ensure_runtime_environment() -> None:
    environment_name = os.environ.get("BONESIS_ENV", "scbolt-bonesis")
    if (
        os.environ.get("CONDA_DEFAULT_ENV") == environment_name
        or os.environ.get(RUNTIME_MARKER) == "1"
    ):
        return

    environment = os.environ.copy()
    environment[RUNTIME_MARKER] = "1"
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        environment_name,
        "python",
        "-u",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execvpe(command[0], command, environment)


_ensure_runtime_environment()

SCBOLT_HOME = Path(os.environ.get("SCBOLT_HOME", _default_scbolt_home())).resolve()
sys.path.insert(0, str(SCBOLT_HOME / "lib"))
sys.path.insert(0, str(SCBOLT_HOME / "scripts" / "infer"))

import bonesis  # noqa: E402
from bonesis.asp_encoding import clingo_encode  # noqa: E402
from bonesis0.asp_encoding import py_of_symbol  # noqa: E402
from scbolt import cli, console  # noqa: E402
from scbolt.runtime import (  # noqa: E402
    SolverDeadline,
    SolverTimeout,
    format_duration,
    get_clingo_parallel_mode,
    iter_solutions,
)
from utils import add_bonesis_arguments, get_node_sets, initialize_bonesis  # noqa: E402


bonesis.settings["quiet"] = True


class TeeStream:
    """Mirror text writes to the terminal and a run log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)


def build_experiment_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a JSON-configured monolithic experiment."
    )
    parser.add_argument(
        "experiment",
        type=Path,
        help="JSON file defining the experiment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the experiment without running Clingo",
    )
    return parser


def load_experiment(file: Path) -> dict[str, Any]:
    file = file.resolve()
    with open(file, encoding="utf-8") as stream:
        experiment = json.load(stream)
    if not isinstance(experiment, dict):
        raise ValueError("the experiment JSON must contain an object")

    missing = sorted(REQUIRED_PARAMETERS - experiment.keys())
    if missing:
        raise ValueError(f"missing experiment parameters: {', '.join(missing)}")
    for parameter in STRING_PARAMETERS:
        if not isinstance(experiment[parameter], str) or not experiment[parameter]:
            raise ValueError(f"{parameter} must be a non-empty string")
    max_clauses = experiment["max_clauses"]
    if (
        not isinstance(max_clauses, int)
        or isinstance(max_clauses, bool)
        or max_clauses < 1
    ):
        raise ValueError("max_clauses must be a positive integer")
    if not isinstance(experiment["dorothea_compatibility"], bool):
        raise ValueError("dorothea_compatibility must be a Boolean")
    levels = experiment["dorothea_levels"]
    if not isinstance(levels, list) or not levels or not all(
        isinstance(level, str) and level for level in levels
    ):
        raise ValueError("dorothea_levels must be a non-empty string list")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", experiment["name"]) is None:
        raise ValueError("name may contain only letters, digits, '.', '_', and '-'")
    if experiment["clingo_mode"] not in {"opt", "optN"}:
        raise ValueError("clingo_mode must be opt or optN")
    if not experiment["clingo_opt_strategy"].startswith(("bb", "usc")):
        raise ValueError("clingo_opt_strategy must start with bb or usc")
    if re.fullmatch(
        r"[1-9][0-9]*(?:,(?:compete|split))?",
        experiment["clingo_parallel_mode"],
    ) is None:
        raise ValueError(
            "clingo_parallel_mode must be INT, INT,compete, or INT,split"
        )

    config = project_path(experiment["config"])
    if not config.is_file():
        raise FileNotFoundError(f"project configuration not found: {config}")
    experiment["_file"] = file
    return experiment


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_run_paths(experiment: dict[str, Any]) -> dict[str, Path]:
    results = project_path(os.environ.get("RESULTS_DIR", "results"))
    run_root = results / "monolithic_algorithm"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = os.environ.get("RUN_ID", f"{experiment['name']}_{timestamp}")
    output = run_root / run_id
    spec = results / "spec"
    paths = {
        "results": results,
        "run_root": run_root,
        "output": output,
        "log": run_root / f"{run_id}.log",
        "spec": spec / "model.bo",
        "mstates": spec / "mstates.csv",
        "mandatory_nodes": spec / "mandatory.txt",
        "forbidden_nodes": spec / "forbidden.txt",
    }
    for name in ("spec", "mstates", "mandatory_nodes", "forbidden_nodes"):
        if not paths[name].is_file():
            raise FileNotFoundError(
                f"missing {paths[name]}; run scripts/build_notebook_data.py first"
            )
    return paths


def prepare_output(experiment: dict[str, Any], paths: dict[str, Path]) -> None:
    output = paths["output"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if paths["log"].exists():
        raise FileExistsError(f"run log already exists: {paths['log']}")
    output.mkdir(parents=True, exist_ok=True)

    latest = paths["run_root"] / "latest"
    if latest.exists() and not latest.is_symlink():
        raise FileExistsError(f"latest path is not a symbolic link: {latest}")
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(output.resolve(), target_is_directory=True)
    shutil.copy2(experiment["_file"], output / "experiment.json")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_gene_list(file: str | Path | None) -> list[str]:
    """Read one gene name per non-empty line."""

    if file is None:
        return []
    with open(file, encoding="utf-8") as stream:
        return [line.rstrip() for line in stream if line.rstrip()]


def temporary_path(file: Path) -> Path:
    """Return a process- and thread-local path for an atomic write."""

    return file.with_name(f".{file.name}.{os.getpid()}.{get_ident()}.tmp")


def write_lines(lines: Iterable[str], file: Path) -> None:
    """Atomically write normalized lines."""

    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(file)
    with open(temporary, "w", encoding="utf-8") as stream:
        for line in lines:
            stream.write(f"{line}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file)


def write_json(data: dict[str, Any], file: Path) -> None:
    """Atomically write JSON metadata."""

    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(file)
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file)


def append_json(data: dict[str, Any], file: Path) -> None:
    """Append and durably flush one JSON record."""

    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class MonolithicSelectionView(bonesis.NonStrongConstantNodesView):
    """Expose the node sets and solver metadata of each incumbent."""

    def format_model(self, model) -> dict[str, Any]:
        shown_atoms = model.symbols(shown=True)
        nodes = {
            py_of_symbol(atom.arguments[0])
            for atom in shown_atoms
            if atom.name == "node"
        }
        strong_constants = {
            py_of_symbol(atom.arguments[0])
            for atom in shown_atoms
            if atom.name == "strong_constant"
        }
        return {
            "nodes": frozenset(nodes),
            "strong_constants": frozenset(strong_constants),
            "retained_nodes": frozenset(nodes - strong_constants),
            "cost": tuple(int(value) for value in model.cost),
            "clingo_model": int(model.number),
            "optimality_proven": bool(model.optimality_proven),
        }


class MonolithicRecorder:
    """Log every distinct Clingo incumbent and retain the best one."""

    def __init__(self, output: Path, solution: Path) -> None:
        self.output = output
        self.solution = solution
        self.best = output / "best"
        self.incumbents_file = output / "incumbents.jsonl"
        self.started_at: float | None = None
        self.started_at_timestamp: str | None = None
        self.finished_at_timestamp: str | None = None
        self.finished_elapsed: float | None = None
        self.models = 0
        self.best_model: dict[str, Any] | None = None
        self.best_iteration: int | None = None
        self.best_incumbent_seconds: float | None = None
        self.first_incumbent_seconds: float | None = None
        self._seen_models: set[tuple[Any, ...]] = set()

    def start(self, run_metadata: dict[str, Any]) -> None:
        """Initialize a fresh run immediately before solving."""

        conflicting = [
            path
            for path in (
                self.output / "metadata.json",
                self.output / "run.json",
                self.output / "status.json",
                self.incumbents_file,
                self.best,
            )
            if path.exists()
        ]
        if conflicting:
            paths = ", ".join(str(path) for path in conflicting)
            raise FileExistsError(
                f"output directory already contains monolithic results: {paths}"
            )

        self.output.mkdir(parents=True, exist_ok=True)
        self.best.mkdir()
        self.started_at = time.monotonic()
        self.started_at_timestamp = self._timestamp()
        write_json(
            {
                **run_metadata,
                "started_at": self.started_at_timestamp,
            },
            self.output / "run.json",
        )
        write_json(self.metadata("running"), self.output / "status.json")

    def record(self, model: dict[str, Any]) -> None:
        """Print one line for a distinct Clingo incumbent."""

        signature = (
            model["clingo_model"],
            model["cost"],
            len(model["nodes"]),
            len(model["strong_constants"]),
        )
        if signature in self._seen_models:
            return
        self._seen_models.add(signature)

        self.models += 1
        elapsed = self.elapsed()
        timestamp = self._timestamp()
        if self.first_incumbent_seconds is None:
            self.first_incumbent_seconds = elapsed

        new_best = self.best_model is None or self._objective(model) > self._objective(
            self.best_model
        )
        if new_best:
            self.best_model = model
            self.best_iteration = self.models
            self.best_incumbent_seconds = elapsed

        append_json(
            {
                "iteration": self.models,
                "timestamp": timestamp,
                "elapsed_seconds": round(elapsed, 6),
                "clingo_model": model["clingo_model"],
                "cost": list(model["cost"]),
                "optimality_proven": model["optimality_proven"],
                "is_best": new_best,
                "retained_node_count": len(model["retained_nodes"]),
                "satisfiable_node_count": len(model["nodes"]),
                "strong_constant_count": len(model["strong_constants"]),
                "nodes": sorted(model["nodes"]),
                "strong_constants": sorted(model["strong_constants"]),
                "retained_nodes": sorted(model["retained_nodes"]),
            },
            self.incumbents_file,
        )
        if new_best:
            self.write_best()

        print(
            "INCUMBENT "
            f"iteration={self.models} "
            f"timestamp={timestamp} "
            f"elapsed_seconds={elapsed:.6f} "
            f"retained_nodes={len(model['retained_nodes'])} "
            f"satisfiable_nodes={len(model['nodes'])} "
            f"strong_constants={len(model['strong_constants'])}",
            flush=True,
        )

    def finish(
        self,
        status: str,
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        """Export the best incumbent and final run status."""

        self.finished_elapsed = self.elapsed()
        self.finished_at_timestamp = self._timestamp()
        self.write_best()
        metadata = self.metadata(status)
        if error is not None:
            metadata["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        write_json(metadata, self.output / "metadata.json")
        write_json(metadata, self.output / "status.json")
        return metadata

    def elapsed(self) -> float:
        """Return elapsed real wall-clock seconds since solver start."""

        if self.started_at is None:
            raise RuntimeError("monolithic recorder has not been started")
        if self.finished_elapsed is not None:
            return self.finished_elapsed
        return time.monotonic() - self.started_at

    def metadata(self, status: str) -> dict[str, Any]:
        """Build current metadata, including an explicit empty result."""

        has_incumbent = self.best_model is not None
        metadata: dict[str, Any] = {
            "strategy": "monolithic_max_nodes_then_max_strong_constants",
            "status": status,
            "has_incumbent": has_incumbent,
            "incumbents": self.models,
            "started_at": self.started_at_timestamp,
            "finished_at": self.finished_at_timestamp,
            "elapsed_seconds": round(self.elapsed(), 6),
            "first_incumbent_seconds": self.first_incumbent_seconds,
            "best_incumbent_seconds": self.best_incumbent_seconds,
            "best_iteration": self.best_iteration,
        }
        if self.best_model is None:
            metadata.update(
                {
                    "message": "No incumbent has been found yet.",
                    "best_retained_nodes": None,
                    "best_satisfiable_nodes": None,
                    "best_strong_constants": None,
                    "best_cost": None,
                }
            )
        else:
            metadata.update(
                {
                    "best_retained_nodes": len(self.best_model["retained_nodes"]),
                    "best_satisfiable_nodes": len(self.best_model["nodes"]),
                    "best_strong_constants": len(self.best_model["strong_constants"]),
                    "best_cost": list(self.best_model["cost"]),
                }
            )
        return metadata

    def write_best(self) -> None:
        """Export the best final node sets."""

        if self.best_model is None:
            return
        write_lines(sorted(self.best_model["nodes"]), self.best / "nodes.txt")
        write_lines(
            sorted(self.best_model["strong_constants"]),
            self.best / "strong_constants.txt",
        )
        retained_file = self.best / "retained_nodes.txt"
        retained_nodes = sorted(self.best_model["retained_nodes"])
        write_lines(retained_nodes, retained_file)
        if self.solution != retained_file:
            write_lines(retained_nodes, self.solution)

    @staticmethod
    def _objective(model: dict[str, Any]) -> tuple[int, int]:
        return len(model["nodes"]), len(model["strong_constants"])

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")


def get_clingo_options(
    mode: str,
    strategy: str,
    configuration: str | None,
    parallel_option: str | None,
) -> list[str]:
    """Build Clingo options without duplicating BoNesis optN options."""

    options: list[str] = []
    if configuration:
        options.extend(["--configuration", configuration])
    if mode == "opt":
        options.extend(["--opt-mode=opt", f"--opt-strategy={strategy}"])
    elif mode == "ignore":
        options.append("--opt-mode=ignore")
    if parallel_option:
        options.append(parallel_option)
    return options


def build_solver_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original joint max-nodes/max-strong-constants selection "
            "and log every Clingo incumbent."
        ),
        formatter_class=cli.HelpFormatter,
    )
    add_bonesis_arguments(parser)
    parser.set_defaults(clingo_mode="opt")
    for action in parser._actions:
        if action.dest == "clingo_mode" and action.help:
            action.help = action.help.replace("default: optN", "default: opt")
            break
    parser.add_argument(
        "--mandatory-nodes",
        dest="mandatory_nodes",
        type=lambda value: Path(value).resolve(),
        required=False,
        metavar="FILE",
        help="file storing mandatory nodes forced to appear",
    )
    parser.add_argument(
        "--forbidden-nodes",
        dest="forbidden_nodes",
        type=lambda value: Path(value).resolve(),
        required=False,
        default=None,
        metavar="FILE",
        help="file storing nodes excluded from the regulatory domain",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=lambda value: Path(value).resolve(),
        required=True,
        metavar="DIR",
        help="fresh output directory for this run",
    )
    parser.add_argument(
        "--clingo-configuration",
        dest="clingo_configuration",
        type=str,
        required=False,
        default=None,
        metavar="NAME | FILE",
        help="Clingo configuration passed as --configuration",
    )
    parser.add_argument(
        "--clingo-opt-strategy",
        dest="clingo_opt_strategy",
        action=cli.Clingo_strategy,
        required=False,
        default="usc",
    )
    return parser


def build_solver_arguments(
    experiment: dict[str, Any],
    paths: dict[str, Path],
) -> argparse.Namespace:
    output = paths["output"]
    arguments = [
        str(paths["spec"]),
        str(paths["mstates"]),
        "--asp",
        str(output / "objective.txt"),
        "--solution",
        str(output / "best" / "retained_nodes.txt"),
        "--output",
        str(output),
        "--mandatory-nodes",
        str(paths["mandatory_nodes"]),
        "--forbidden-nodes",
        str(paths["forbidden_nodes"]),
        "--domain",
        experiment["prior_knowledge"],
        "--organism",
        experiment["organism"],
        "--geneinfo-version",
        experiment["geneinfo_version"],
        "--omnipath-version",
        experiment["omnipath_version"],
        "--hcop-version",
        experiment["hcop_version"],
        "--dorothea-api",
        experiment["dorothea_api"],
        "--dorothea-compatibility",
        str(experiment["dorothea_compatibility"]).lower(),
        "--dorothea-levels",
        *experiment["dorothea_levels"],
        "--bonesis-mode",
        "hard",
        "--max-clauses",
        str(experiment["max_clauses"]),
        "--canonical",
        "true",
        "--clingo-mode",
        experiment["clingo_mode"],
        "--clingo-configuration",
        experiment["clingo_configuration"],
        "--clingo-opt-strategy",
        experiment["clingo_opt_strategy"],
        "--jobs",
        experiment["clingo_parallel_mode"],
        "--timeout",
        experiment["timeout"],
    ]
    args = build_solver_parser().parse_args(arguments)
    args.experiment_name = experiment["name"]
    args.experiment_file = output / "experiment.json"
    args.project_config = project_path(experiment["config"])
    return args


def run_solver(args: argparse.Namespace) -> int:
    if args.clingo_mode not in {"opt", "optN"}:
        raise ValueError(
            "monolithic incumbent logging requires --clingo-mode opt or optN"
        )
    if args.bonesis_mode != "hard":
        raise ValueError("monolithic comparison requires --bonesis-mode hard")
    if args.filter_grn is not None:
        raise ValueError(
            "monolithic comparison must start from the complete prior network; "
            "--filter-grn is not allowed"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    write_lines(
        (
            "# The solve is run through the BoNesis Python API.",
            "# Primary objective: maximize satisfiable nodes.",
            "# Secondary objective: maximize strong constants.",
        ),
        args.asp,
    )

    bo, canonical, _ = initialize_bonesis(
        args,
        allow_skipping_nodes=True,
        default_canonical=True,
        forbidden_nodes_file=args.forbidden_nodes,
    )
    console.print_task("running monolithic joint gene selection")
    bo.maximize_nodes()
    bo.maximize_strong_constants()
    for node in read_gene_list(args.mandatory_nodes):
        bo.custom(f"node({clingo_encode(node)}).")

    nodes_in_data, nodes_in_domain, domain_edges = get_node_sets(bo)
    console.print_node_reference(
        nodes_in_data,
        nodes_in_domain,
        domain_edges,
        flush=True,
    )
    console.print_solver_options(
        args.clingo_mode,
        args.clingo_opt_strategy,
        args.max_clauses,
        canonical,
        configuration=args.clingo_configuration or "auto",
        jobs=args.jobs,
        flush=True,
    )
    console.print_options(
        "objective: maximize satisfiable nodes, then maximize strong constants",
        flush=True,
    )
    console.print_options(
        "domain source: complete configured prior network (no selected-gene seed)",
        flush=True,
    )
    console.print_warning("this may take some time.", flush=True)

    parallel_jobs, parallel_option = get_clingo_parallel_mode(args.jobs)
    settings = {
        "parallel": parallel_jobs,
        "clingo_opt_strategy": args.clingo_opt_strategy,
        "clingo_gil_workaround": 0,
        "clingo_options": get_clingo_options(
            args.clingo_mode,
            args.clingo_opt_strategy,
            args.clingo_configuration,
            parallel_option,
        ),
    }
    recorder = MonolithicRecorder(args.output, args.solution)
    recorder.start(
        {
            "experiment": args.experiment_name,
            "experiment_file": str(args.experiment_file),
            "project_config": str(args.project_config),
            "strategy": "monolithic_max_nodes_then_max_strong_constants",
            "lexicographic_objectives": [
                "maximize_nodes",
                "maximize_strong_constants",
            ],
            "spec": str(args.spec),
            "mstates": str(args.mstates),
            "mandatory_nodes": (
                str(args.mandatory_nodes) if args.mandatory_nodes else None
            ),
            "forbidden_nodes": (
                str(args.forbidden_nodes) if args.forbidden_nodes else None
            ),
            "filter_grn": None,
            "initial_solution": None,
            "domain": args.domain,
            "organism": args.organism,
            "dorothea_api": args.dorothea_api,
            "dorothea_levels": args.dorothea_levels,
            "dorothea_compatibility": args.dorothea_compatibility,
            "geneinfo_version": args.geneinfo_version,
            "omnipath_version": args.omnipath_version,
            "hcop_version": args.hcop_version,
            "data_nodes": len(nodes_in_data),
            "domain_nodes": len(nodes_in_domain),
            "domain_edges": domain_edges,
            "max_clauses": args.max_clauses,
            "canonical": canonical,
            "allow_skipping_nodes": True,
            "bonesis_mode": args.bonesis_mode,
            "clingo_mode": args.clingo_mode,
            "clingo_opt_strategy": args.clingo_opt_strategy,
            "clingo_configuration": args.clingo_configuration or "auto",
            "clingo_gil_workaround": 0,
            "jobs": args.jobs,
            "timeout_seconds": args.timeout,
            "command": sys.argv,
        }
    )
    view = MonolithicSelectionView(
        bo,
        mode=args.clingo_mode,
        intermediate_model_cb=recorder.record,
        **settings,
    )
    deadline = SolverDeadline(args.timeout)

    status = "optimal"
    try:
        for model in iter_solutions(view, deadline):
            recorder.record(model)
    except SolverTimeout:
        status = "timeout"
    except KeyboardInterrupt:
        status = "interrupted"
        try:
            view.interrupt()
        except (AttributeError, RuntimeError):
            pass
    except Exception as error:
        metadata = recorder.finish("error", error)
        print_final_status(metadata)
        raise

    if status == "optimal" and recorder.models == 0:
        status = "no_solution"
    metadata = recorder.finish(status)
    print_final_status(metadata)

    if status == "timeout":
        console.print_warning(
            "monolithic inference reached its time limit "
            f"({format_duration(args.timeout)})",
            flush=True,
        )
    elif status == "interrupted":
        console.print_warning("monolithic inference interrupted", flush=True)
        return 130
    elif status == "optimal":
        console.print_result("monolithic inference completed", flush=True)
    else:
        console.print_warning("monolithic inference found no incumbent", flush=True)
    return 0


def print_final_status(metadata: dict[str, Any]) -> None:
    """Print a single parseable final status line to the run log."""

    retained = metadata["best_retained_nodes"]
    print(
        "FINAL "
        f"status={metadata['status']} "
        f"timestamp={metadata['finished_at']} "
        f"elapsed_seconds={metadata['elapsed_seconds']:.6f} "
        f"incumbents={metadata['incumbents']} "
        f"best_retained_nodes={retained if retained is not None else 'none'}",
        flush=True,
    )


def print_launch_header(
    experiment: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    invocation = [
        "./scripts/run_monolithic_algorithm.py",
        display_path(experiment["_file"]),
    ]
    print(f"launcher={shlex.join(invocation)}")
    print(f"started_at={datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"experiment={experiment['name']}")
    print(f"experiment_file={display_path(experiment['_file'])}")
    print(f"config={experiment['config']}")
    print(f"output={display_path(paths['output'])}")
    print(f"log={display_path(paths['log'])}")
    print(
        f"domain=full {experiment['prior_knowledge']}; "
        f"dorothea_api={experiment['dorothea_api']}; "
        f"dorothea_levels={' '.join(experiment['dorothea_levels'])}"
    )
    print("objective=maximize_nodes then maximize_strong_constants")
    print(
        f"clingo=mode={experiment['clingo_mode']}, "
        f"configuration={experiment['clingo_configuration']}, "
        f"strategy={experiment['clingo_opt_strategy']}, "
        f"parallel_mode={experiment['clingo_parallel_mode']}"
    )


def run_experiment(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    paths = build_run_paths(experiment)
    solver_args = build_solver_arguments(experiment, paths)
    if args.dry_run:
        print(f"experiment={experiment['name']}")
        print(f"configuration={experiment['_file']}")
        print(f"output={paths['output']}")
        print(f"log={paths['log']}")
        print(f"clingo_strategy={solver_args.clingo_opt_strategy}")
        print(f"clingo_parallel_mode={solver_args.jobs}")
        print("status=dry-run")
        return 0

    prepare_output(experiment, paths)
    run_log = paths["log"]
    terminal_stdout = sys.stdout
    terminal_stderr = sys.stderr
    with open(run_log, "w", encoding="utf-8", buffering=1) as log_stream:
        stdout = TeeStream(terminal_stdout, log_stream)
        stderr = TeeStream(terminal_stderr, log_stream)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print_launch_header(experiment, paths)
            try:
                return run_solver(solver_args)
            except Exception:
                traceback.print_exc()
                return 1


def main() -> int:
    return run_experiment(build_experiment_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
