#!/usr/bin/env python3
"""Export the scBOLT results required by the hematopoiesis notebooks."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/scbolt-mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/scbolt-cache")


HEMATOPOIESIS_DIR = Path(__file__).resolve().parents[1]
CASE_STUDIES_DIR = HEMATOPOIESIS_DIR.parent
DEFAULT_PROJECT_DIR = HEMATOPOIESIS_DIR / "project"
DEFAULT_OUTPUT_DIR = HEMATOPOIESIS_DIR / "results"
LEGACY_STREAM_OBJECT = (
    CASE_STUDIES_DIR / "data" / "hematopoiesis" / "stream" / "mstates.h5ad.pkl"
)

MSTATES = Path("spec/mstates.csv")
SELECTED_GENES = Path("gene_selection/selected_genes.txt")
SPEC_FILENAMES = (
    "model.bo",
    "mstates.csv",
    "important.txt",
    "mandatory.txt",
    "forbidden.txt",
)
STREAM_OBJECT = Path("omics/stream.pkl")
MANAGED_DIRS = ("bn", "omics", "spec", "gene_selection")
STREAM_OBS_COLUMNS = (
    "label",
    "macrostate",
    "node",
    "branch_id",
    "branch_lam",
    "branch_dist",
)
STREAM_UNS_KEYS = ("epg", "flat_tree", "label_color")
STREAM_EXPORT_SIZE_ALLOWANCE = 5 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the minimal scBOLT results used by the hematopoiesis "
            "notebooks. Run the actual export in the scbolt-cs-stream environment."
        )
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help=f"scBOLT project directory (default: {DEFAULT_PROJECT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"notebook results directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--stream-object",
        type=Path,
        default=None,
        help=(
            "full STREAM pickle to reduce; by default, discover it in the "
            "project or the former repository data directory"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing managed result directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and display the export without writing files",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required result not found: {path}")
    return path


def discover_bn_files(project_dir: Path) -> Dict[Path, Path]:
    source_dir = project_dir / "infer" / "bn" / "submin"
    solution_dirs = (
        sorted(
            (
                path
                for path in source_dir.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        )
        if source_dir.is_dir()
        else []
    )

    if not solution_dirs:
        raise FileNotFoundError(f"no numerical BN solution found under {source_dir}")

    files = {}
    for solution_dir in solution_dirs:
        for filename in ("model.bnet", "configs.csv"):
            files[Path("bn") / solution_dir.name / filename] = require_file(
                solution_dir / filename
            )
    return files


def discover_stream_object(
    project_dir: Path,
    requested: Optional[Path],
) -> Path:
    if requested is not None:
        return require_file(requested)

    stream_dir = project_dir / "omics" / "mstates" / "stream"
    candidates = (
        stream_dir / "mstates.pkl",
        stream_dir / "mstates.h5ad.pkl",
        stream_dir / "mstates.pklz",
        LEGACY_STREAM_OBJECT,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "full STREAM object not found; searched:\n"
        f"  - {searched}\n"
        "pass it explicitly with --stream-object"
    )


def build_export_plan(
    project_dir: Path,
    stream_object: Optional[Path],
) -> Tuple[Dict[Path, Path], Path]:
    project_dir = project_dir.expanduser().resolve()
    files = discover_bn_files(project_dir)
    spec_dir = project_dir / "infer" / "spec"
    files.update(
        {
            Path("spec") / filename: require_file(spec_dir / filename)
            for filename in SPEC_FILENAMES
        }
    )
    files[SELECTED_GENES] = require_file(
        project_dir / "infer" / "genes" / "lock" / "comps.txt"
    )
    return files, discover_stream_object(project_dir, stream_object)


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def count_csv_features(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip("\r\n").split(",")
    return max(len(header) - 1, 0)


def export_stream_object(source: Path, destination: Path) -> None:
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import stream as st
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "the STREAM export requires the scbolt-cs-stream environment; run "
            "`conda run -n scbolt-cs-stream python "
            f"scripts/build_notebook_data.py ...` ({error})"
        ) from error

    adata = st.read(file_name=str(source))
    missing_obs = [column for column in STREAM_OBS_COLUMNS if column not in adata.obs]
    missing_uns = [key for key in STREAM_UNS_KEYS if key not in adata.uns]
    if "X_se" not in adata.obsm:
        raise KeyError(f"X_se not found in STREAM object: {source}")
    if missing_obs:
        raise KeyError(
            f"missing STREAM observation columns in {source}: " + ", ".join(missing_obs)
        )
    if missing_uns:
        raise KeyError(
            f"missing STREAM graph keys in {source}: " + ", ".join(missing_uns)
        )

    exported = ad.AnnData(
        X=np.empty((adata.n_obs, 0), dtype=np.float32),
        obs=adata.obs.loc[:, list(STREAM_OBS_COLUMNS)].copy(),
        var=pd.DataFrame(index=pd.Index([], dtype=str)),
    )
    exported.obsm["X_se"] = adata.obsm["X_se"].copy()
    for key in STREAM_UNS_KEYS:
        exported.uns[key] = adata.uns[key]
    exported.uns["workdir"] = "."

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.pkl")
    if temporary.exists():
        temporary.unlink()
    try:
        st.write(
            exported,
            file_name=temporary.name,
            file_path=str(temporary.parent),
            file_format="pkl",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def estimated_export_size(files: Dict[Path, Path]) -> int:
    return (
        sum(source.stat().st_size for source in files.values())
        + STREAM_EXPORT_SIZE_ALLOWANCE
    )


def export(
    files: Dict[Path, Path],
    stream_object: Path,
    output_dir: Path,
    force: bool,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    managed_dirs = [output_dir / name for name in MANAGED_DIRS]
    existing = [path for path in managed_dirs if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"output already exists: {names}; use --force to replace it"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    required_size = estimated_export_size(files)
    reclaimable_size = sum(path_size(path) for path in existing)
    available_size = shutil.disk_usage(output_dir.parent).free + reclaimable_size
    if required_size > available_size:
        raise OSError(
            "insufficient disk space: "
            f"need {format_size(required_size)}, "
            f"have {format_size(available_size)} after replacing existing outputs"
        )

    with tempfile.TemporaryDirectory(
        prefix=".hematopoiesis-notebook-data-",
        dir=str(output_dir.parent),
    ) as temporary_dir:
        staging = Path(temporary_dir)
        for relative_path, source in files.items():
            destination = staging / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        export_stream_object(stream_object, staging / STREAM_OBJECT)

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        for name in MANAGED_DIRS:
            shutil.move(str(staging / name), str(output_dir / name))


def main() -> None:
    args = parse_args()
    try:
        files, stream_object = build_export_plan(
            args.project_dir,
            args.stream_object,
        )
        models = {
            path: source
            for path, source in files.items()
            if path.name == "model.bnet" and path.parts[0] == "bn"
        }
        component_counts = {count_nonempty_lines(source) for source in models.values()}
        if len(component_counts) != 1:
            counts = ", ".join(str(count) for count in sorted(component_counts))
            raise RuntimeError(
                "Boolean-network solutions do not share one component count: " + counts
            )
        selected_gene_count = count_nonempty_lines(files[SELECTED_GENES])
        if component_counts != {selected_gene_count}:
            counts = ", ".join(str(count) for count in sorted(component_counts))
            raise RuntimeError(
                "selected-gene and Boolean-network component counts differ: "
                f"selection={selected_gene_count}, networks={counts}"
            )

        total_size = estimated_export_size(files)
        print(f"project: {args.project_dir.expanduser().resolve()}")
        print(f"output: {args.output_dir.expanduser().resolve()}")
        print(f"Boolean networks: {len(models)}")
        print(f"Boolean-network components: {component_counts.pop()}")
        print(f"retained genes: {count_csv_features(files[MSTATES])}")
        print(f"selected genes: {selected_gene_count}")
        print(f"STREAM source: {stream_object}")
        print(f"estimated exported size (upper bound): {format_size(total_size)}")

        if args.dry_run:
            print("dry run: no files written")
            return

        export(files, stream_object, args.output_dir, force=args.force)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None

    print("export complete")


if __name__ == "__main__":
    main()
