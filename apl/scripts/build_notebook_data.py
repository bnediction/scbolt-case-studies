#!/usr/bin/env python3
"""Export the scBOLT results required by the APL notebooks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import anndata as ad
import pandas as pd


APL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_DIR = APL_DIR / "project_gsm"
DEFAULT_OUTPUT_DIR = APL_DIR / "results"
DEFAULT_BIN_METHOD = "consensus"
MARKER_GENES = ["S100a8", "S100a9", "Ly6g", "Ngp", "Spi1"]
INTEGRATED_H5AD = Path("omics/integrated.h5ad")
MSTATES = Path("spec/mstates.csv")
SELECTED_GENES = Path("gene_selection/selected_genes.txt")
SPEC_FILENAMES = (
    "model.bo",
    "mstates.csv",
    "important.txt",
    "mandatory.txt",
    "forbidden.txt",
)
MANAGED_DIRS = ("bn", "omics", "spec", "gene_selection")
CONDITION_H5ADS = {
    Path("omics/ctrl.h5ad"),
    Path("omics/treated.h5ad"),
}
POTENCY_FILES = {
    "ctrl": Path("omics/trajectories/potency/ctrl/potency.csv"),
    "treated": Path("omics/trajectories/potency/treated/potency.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the minimal scBOLT results used by the APL notebooks."
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
        "--force",
        action="store_true",
        help="replace existing managed result directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and display the export without copying files",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required result not found: {path}")
    return path


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def discover_macrostate_method(project_dir: Path) -> tuple[str, Path, Path]:
    root = project_dir / "omics" / "mstates"
    candidates = []

    if root.is_dir():
        for method_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            ctrl = method_dir / "ctrl" / "mstates.h5ad"
            treated = method_dir / "treated" / "mstates.h5ad"
            if ctrl.is_file() and treated.is_file():
                candidates.append((method_dir.name, ctrl, treated))

    if not candidates:
        raise FileNotFoundError(f"no complete macrostate result found under {root}")
    if len(candidates) > 1:
        methods = ", ".join(method for method, _, _ in candidates)
        raise RuntimeError(f"multiple macrostate methods found under {root}: {methods}")

    return candidates[0]


def discover_binarisation_method(project_dir: Path) -> str:
    metadata_file = project_dir / "infer" / "spec" / "mstates.scbolt.json"
    if not metadata_file.is_file():
        return DEFAULT_BIN_METHOD

    metadata = json.loads(metadata_file.read_text())
    params_file = Path(metadata["params_file"])
    if not params_file.is_file():
        params_file = APL_DIR / params_file.name
    if not params_file.is_file():
        return DEFAULT_BIN_METHOD

    pattern = re.compile(r"^\s*BIN_METHOD\s*(?:\?|:)?=\s*([^\s#]+)")
    for line in params_file.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    return DEFAULT_BIN_METHOD


def discover_bn_files(project_dir: Path) -> dict[Path, Path]:
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
            source = require_file(solution_dir / filename)
            files[Path("bn") / solution_dir.name / filename] = source

    return files


def build_export_plan(
    project_dir: Path,
) -> tuple[str, str, dict[Path, Path], dict[str, Path]]:
    project_dir = project_dir.resolve()
    method, ctrl, treated = discover_macrostate_method(project_dir)
    bin_method = discover_binarisation_method(project_dir)

    files = discover_bn_files(project_dir)
    files.update(
        {
            Path("omics/ctrl.h5ad"): ctrl,
            Path("omics/treated.h5ad"): treated,
            Path("omics/integrated.h5ad"): require_file(
                project_dir / "omics" / "annot" / "integrated" / "annot.h5ad"
            ),
            SELECTED_GENES: require_file(
                project_dir / "infer" / "genes" / "lock" / "comps.txt"
            ),
        }
    )
    spec_dir = project_dir / "infer" / "spec"
    files.update(
        {
            Path("spec") / filename: require_file(spec_dir / filename)
            for filename in SPEC_FILENAMES
        }
    )
    potency_files = {
        condition: require_file(project_dir / relative_path)
        for condition, relative_path in POTENCY_FILES.items()
    }
    return method, bin_method, files, potency_files


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


def export_integrated_h5ad(
    source: Path,
    destination: Path,
    potency_files: dict[str, Path],
) -> None:
    adata = ad.read_h5ad(source, backed="r")
    try:
        obs = adata.obs.loc[:, ["condition", "label"]].copy()
        potency_scores = pd.concat(
            {
                condition: pd.read_csv(file, index_col=0)["score"]
                for condition, file in potency_files.items()
            },
            names=["condition", "barcode"],
        )
        if not potency_scores.index.is_unique:
            raise RuntimeError("potency scores contain duplicate cell identifiers")

        barcodes = obs.index.to_series().str.split(":", n=1).str[-1]
        potency_index = pd.MultiIndex.from_arrays(
            [obs["condition"].astype(str), barcodes],
            names=["condition", "barcode"],
        )
        scores = potency_scores.reindex(potency_index)
        if scores.isna().any():
            missing = potency_index[scores.isna()]
            examples = ", ".join(
                f"{condition}:{barcode}" for condition, barcode in missing[:5]
            )
            raise RuntimeError(
                f"missing potency scores for {len(missing)} cells: {examples}"
            )
        obs["potency_score"] = scores.to_numpy()

        marker_expression = adata[:, MARKER_GENES].layers["log-norm"].copy()
        exported = ad.AnnData(
            obs=obs,
            var=pd.DataFrame(index=MARKER_GENES),
            obsm={"X_umap": adata.obsm["X_umap"].copy()},
            layers={"log-norm": marker_expression},
        )
    finally:
        adata.file.close()

    exported.write_h5ad(destination, compression="gzip")


def export_condition_h5ad(source: Path, destination: Path) -> None:
    adata = ad.read_h5ad(source, backed="r")
    try:
        exported = ad.AnnData(
            obs=adata.obs.loc[:, ["macrostate"]].copy(),
            obsm={"X_umap": adata.obsm["X_umap"].copy()},
        )
    finally:
        adata.file.close()

    exported.write_h5ad(destination, compression="gzip")


def estimated_export_size(files: dict[Path, Path]) -> int:
    copied_size = sum(
        source.stat().st_size
        for relative_path, source in files.items()
        if relative_path not in CONDITION_H5ADS | {INTEGRATED_H5AD}
    )
    return copied_size + 100 * 1024**2


def export(
    files: dict[Path, Path],
    potency_files: dict[str, Path],
    output_dir: Path,
    force: bool,
) -> None:
    output_dir = output_dir.resolve()
    managed_dirs = [output_dir / name for name in MANAGED_DIRS]
    existing = [path for path in managed_dirs if path.exists()]

    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"output already exists: {names}; use --force to replace it"
        )

    required_size = estimated_export_size(files)
    reclaimable_size = sum(path_size(path) for path in existing)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_size = shutil.disk_usage(output_dir).free + reclaimable_size

    if required_size > available_size:
        raise OSError(
            "insufficient disk space: "
            f"need {format_size(required_size)}, "
            f"have {format_size(available_size)} after replacing existing outputs"
        )

    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    for relative_path, source in files.items():
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative_path == INTEGRATED_H5AD:
            export_integrated_h5ad(source, destination, potency_files)
        elif relative_path in CONDITION_H5ADS:
            export_condition_h5ad(source, destination)
        else:
            shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    try:
        method, bin_method, files, potency_files = build_export_plan(args.project_dir)
        models = {
            path: source
            for path, source in files.items()
            if path.name == "model.bnet" and path.parts[0] == "bn"
        }
        selected_gene_count = count_nonempty_lines(files[SELECTED_GENES])
        component_counts = {count_nonempty_lines(source) for source in models.values()}
        if component_counts != {selected_gene_count}:
            counts = ", ".join(str(count) for count in sorted(component_counts))
            raise RuntimeError(
                "selected-gene and Boolean-network component counts differ: "
                f"selection={selected_gene_count}, networks={counts}"
            )
        retained_gene_count = pd.read_csv(
            files[MSTATES],
            index_col=0,
            nrows=0,
        ).shape[1]
        total_size = estimated_export_size(files)

        print(f"project: {args.project_dir.resolve()}")
        print(f"output: {args.output_dir.resolve()}")
        print(f"macrostate method: {method}")
        print(f"binarization method: {bin_method}")
        print(f"retained genes: {retained_gene_count}")
        print(f"selected genes: {selected_gene_count}")
        print("potency scores: embedded in omics/integrated.h5ad")
        print(f"Boolean networks: {len(models)}")
        print(f"files: {len(files)}")
        print(f"estimated size (upper bound): {format_size(total_size)}")

        if args.dry_run:
            print("dry run: no files copied")
            return

        export(files, potency_files, args.output_dir, force=args.force)
    except (OSError, RuntimeError) as error:
        message = f"error: {error}"
        if "potency" in str(error):
            message += "\nrun `scbolt potency` before exporting notebook data"
        raise SystemExit(message) from None

    print("export complete")


if __name__ == "__main__":
    main()
