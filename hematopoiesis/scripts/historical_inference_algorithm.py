#!/usr/bin/env python

"""Run the historical max-nodes/max-strong-constants inference algorithm."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from threading import get_ident
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/scbolt-mpl-cache")


def _default_scbolt_home() -> Path:
    return Path(__file__).resolve().parents[3] / "scbolt"


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


class HistoricalSelectionView(bonesis.NonStrongConstantNodesView):
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


class HistoricalRecorder:
    """Log every distinct Clingo incumbent and retain the best one."""

    def __init__(self, output: Path, solution: Path) -> None:
        self.output = output
        self.solution = solution
        self.best = output / "best"
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
                self.best,
            )
            if path.exists()
        ]
        if conflicting:
            paths = ", ".join(str(path) for path in conflicting)
            raise FileExistsError(
                f"output directory already contains historical results: {paths}"
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
        if self.first_incumbent_seconds is None:
            self.first_incumbent_seconds = elapsed

        new_best = self.best_model is None or self._objective(model) > self._objective(
            self.best_model
        )
        if new_best:
            self.best_model = model
            self.best_iteration = self.models
            self.best_incumbent_seconds = elapsed

        print(
            "INCUMBENT "
            f"iteration={self.models} "
            f"timestamp={self._timestamp()} "
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
            raise RuntimeError("historical recorder has not been started")
        if self.finished_elapsed is not None:
            return self.finished_elapsed
        return time.monotonic() - self.started_at

    def metadata(self, status: str) -> dict[str, Any]:
        """Build current metadata, including an explicit empty result."""

        has_incumbent = self.best_model is not None
        metadata: dict[str, Any] = {
            "strategy": "historical_max_nodes_then_max_strong_constants",
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


def build_parser() -> argparse.ArgumentParser:
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


def run(args: argparse.Namespace) -> int:
    if args.clingo_mode not in {"opt", "optN"}:
        raise ValueError(
            "historical incumbent logging requires --clingo-mode opt or optN"
        )
    if args.bonesis_mode != "hard":
        raise ValueError("historical comparison requires --bonesis-mode hard")
    if args.filter_grn is not None:
        raise ValueError(
            "historical comparison must start from the complete prior network; "
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
    console.print_task("running historical joint gene selection")
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
    recorder = HistoricalRecorder(args.output, args.solution)
    recorder.start(
        {
            "strategy": "historical_max_nodes_then_max_strong_constants",
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
    view = HistoricalSelectionView(
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
            "historical inference reached its time limit "
            f"({format_duration(args.timeout)})",
            flush=True,
        )
    elif status == "interrupted":
        console.print_warning("historical inference interrupted", flush=True)
        return 130
    elif status == "optimal":
        console.print_result("historical inference completed", flush=True)
    else:
        console.print_warning("historical inference found no incumbent", flush=True)
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


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
