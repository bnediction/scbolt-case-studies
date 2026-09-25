from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import bonesistools as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from scbolt import console, omics
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from statsmodels.nonparametric.smoothers_lowess import lowess

_NESTOROWA_SOURCES = {
    "counts": (
        "GSE81682_HTSeq_counts.txt.gz",
        ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE81nnn/GSE81682/suppl/"
        "GSE81682_HTSeq_counts.txt.gz"),
    ),
    "published": (
        "normalisedCounts.txt.gz",
        "http://blood.stemcells.cam.ac.uk/data/normalisedCounts.txt.gz",
    ),
    "cell_types": (
        "all_cell_types.txt",
        "http://blood.stemcells.cam.ac.uk/data/all_cell_types.txt",
    ),
    "clusters": (
        "cluster_ids.txt",
        "http://blood.stemcells.cam.ac.uk/data/cluster_ids.txt",
    ),
}

_LABEL_GROUPS = {
    "HSC": ["LTHSC_broad", "STHSC_broad", "LTHSC", "STHSC", "ESLAM", "HSC1"],
    "LMPP": ["LMPP_broad", "LMPP"],
    "MPP": [
        "MPP_broad",
        "MPP1_broad",
        "MPP2_broad",
        "MPP3_broad",
        "MPP",
        "MPP1",
        "MPP2",
        "MPP3",
    ],
    "CMP": ["CMP_broad", "CMP"],
    "MEP": ["MEP_broad", "MEP"],
    "GMP": ["GMP_broad", "GMP"],
}

_CLUSTER_LABEL_FALLBACKS = {
    "purple": ["HSC"],
    "deeppink": ["MEP"],
    "gold": ["CMP", "GMP"],
    "darkturquoise": ["CMP", "MPP", "LMPP"],
}

_BIOMART_URL = "https://jun2026.archive.ensembl.org/biomart/martservice"
_BIOMART_ENDPOINTS = (
    _BIOMART_URL,
    "https://useast.ensembl.org/biomart/martservice",
    "https://asia.ensembl.org/biomart/martservice",
)
_BIOMART_QUERY = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0"
       uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="mmusculus_gene_ensembl" interface="default">
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="mgi_symbol"/>
  </Dataset>
</Query>
"""


@dataclass(frozen=True)
class Options:
    path: Path = Path("project/omics/annot")
    count_file: Path = Path("project/input/nestorowa.h5ad")
    source_cache: Path | None = None
    hvg: int = 2000
    hvg_span: float = 0.01
    hvg_curve_neighbors: int = 8
    pca_max_dimension: int = 100
    pca_dimension: int = 40
    n_jobs: int = 4


OPTIONS = Options()


def main(options: Options = OPTIONS) -> None:
    options.path.mkdir(parents=True, exist_ok=True)
    source_cache = options.source_cache or _default_source_cache()
    biomart_cache = options.path / "mouse_ensembl_symbols.tsv.gz"

    console.print_task("loading Nestorowa source matrices and labels")
    adata = _load_nestorowa_source(source_cache)

    console.print_task("filtering cells and genes using published expression")
    _filter_published_expression(adata)
    n_vars_before_mapping = adata.n_vars
    console.print_info(f"filtered matrix: {adata.n_obs} cells x {adata.n_vars} genes")

    console.print_task("building the historical normalized log2 matrix")
    bt.omics.pp.normalize(
        adata,
        target_sum=1e6,
        key_added="stream",
        copy=False,
    )
    bt.omics.pp.log1p(
        adata,
        base=2,
        expression="stream",
        copy=False,
    )

    console.print_task(
        "computing highly variable genes "
        f"(top={options.hvg}, loess_frac={options.hvg_span})"
    )
    _select_lowess_hvg(
        adata,
        expression="stream",
        n_features=options.hvg,
        span=options.hvg_span,
        curve_neighbors=options.hvg_curve_neighbors,
    )
    _plot_hvg(adata, options.path / "hvg.pdf")

    console.print_task(
        "computing principal components "
        f"(dimensions={options.pca_dimension})"
    )
    bt.omics.tl.pca(
        adata,
        n_components=options.pca_max_dimension,
        layer="stream",
        var_subset="highly_variable",
        svd_solver="arpack",
        key_added="pca",
        seed=42,
        n_jobs=options.n_jobs,
        copy=False,
    )
    adata.obsm["top_pcs"] = np.asarray(
        adata.obsm["pca"][:, : options.pca_dimension]
    ).copy()

    console.print_task("computing the spectral embedding")
    bt.omics.tl.neighbors(
        adata,
        representation="top_pcs",
        n_neighbors=15,
        connectivity_method="fuzzy",
        key_added="se_neighbors",
        seed=0,
        n_jobs=options.n_jobs,
        copy=False,
    )
    bt.omics.tl.spectral(
        adata,
        n_components=4,
        neighbors_key="se_neighbors",
        key_added="X_se",
        seed=10,
        n_jobs=options.n_jobs,
        copy=False,
    )

    console.print_task("computing UMAP from the spectral embedding")
    bt.omics.tl.neighbors(
        adata,
        representation="X_se",
        n_neighbors=50,
        connectivity_method="fuzzy",
        key_added="umap_neighbors",
        seed=0,
        n_jobs=options.n_jobs,
        copy=False,
    )
    bt.omics.tl.umap(
        adata,
        n_components=2,
        neighbors_key="umap_neighbors",
        min_dist=0.1,
        spread=1.0,
        n_iter=500,
        key_added="X_umap",
        seed=42,
        n_jobs=options.n_jobs,
        copy=False,
    )

    # Mapping is deliberately delayed so duplicate symbols cannot alter the
    # historical HVG, PCA, neighbor, or spectral calculations.
    del adata.layers["stream"]
    console.print_task("mapping Ensembl identifiers and merging duplicate symbols")
    identifiers = bt.resources.ncbi.identifiers(
        organism="mouse",
        version="bundled",
    )
    mapping_stats = _map_and_merge_gene_symbols(
        adata,
        identifiers=identifiers,
        biomart_cache=biomart_cache,
    )
    console.print_info(
        "gene identifiers: "
        f"NCBI-resolved={mapping_stats['ncbi_resolved']}, "
        f"BioMart-rescued={mapping_stats['biomart_rescued']}, "
        f"merged-duplicates={mapping_stats['merged_duplicates']}, "
        f"unresolved-Ensembl={mapping_stats['unresolved']}"
    )

    console.print_task("calculating QC metrics on merged raw counts")
    bt.omics.pp.mitochondrial_genes(
        adata,
        index_type="name",
        key="mt",
        axis="var",
        identifiers=identifiers,
        copy=False,
    )
    bt.omics.pp.ribosomal_genes(
        adata,
        index_type="name",
        key="rps",
        axis="var",
        identifiers=identifiers,
        copy=False,
    )
    bt.omics.pp.qc(
        adata,
        expression="counts",
        qc_vars=["mt", "rps"],
        percent_top=None,
        log1p=False,
        copy=False,
    )

    console.print_task("normalizing and scaling merged raw counts")
    bt.omics.pp.normalize(
        adata,
        expression="counts",
        target_sum=1e4,
        key_added="norm",
        copy=False,
    )
    bt.omics.pp.log1p(
        adata,
        expression="norm",
        key_added="log-norm",
        copy=False,
    )
    bt.omics.pp.scale(
        adata,
        expression="log-norm",
        key_added="correct",
        copy=False,
    )
    adata.layers["correct"] = np.asarray(
        adata.layers["correct"],
        dtype=np.float32,
    )

    console.print_task("plotting embeddings")
    bt.omics.pl.embedding(
        adata,
        obs="label",
        representation="X_umap",
        n_components=2,
        outfile=options.path / "umap_label.pdf",
    )
    bt.omics.pl.embedding(
        adata,
        obs="label",
        representation="X_se",
        n_components=3,
        outfile=options.path / "spectral.pdf",
    )

    adata.uns["spectral_preprocessing"] = {
        "published_matrix": _NESTOROWA_SOURCES["published"][1],
        "filter_expression_cutoff": 1,
        "min_features_per_cell": 100,
        "min_cells_per_feature": 5,
        "hvg_method": "LOWESS point-to-curve distance",
        "hvg_span": options.hvg_span,
        "hvg_count": options.hvg,
        "pca_computed": options.pca_max_dimension,
        "pca_retained": options.pca_dimension,
        "spectral_connectivity": "fuzzy (historical name: umap)",
        "variables_before_symbol_merge": n_vars_before_mapping,
        "variables_after_symbol_merge": adata.n_vars,
    }
    adata.uns["scbolt"] = {
        "input_source": "Nestorowa published normalized and HTSeq matrices",
        "preprocessing": "bonesistools with LOWESS point-to-curve HVG scoring",
    }

    outfile = options.path / "annot.h5ad"
    console.print_task(f"saving AnnData (file={console.format_path(outfile)})")
    omics.drop_expression_matrices(adata, layers=("norm",))
    omics.write_h5ad(adata, filename=outfile, compression="gzip")
    _publish_count_input(outfile, options.count_file)


def _default_source_cache() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    standard = cache_root / "bonesistools" / "datasets" / "nestorowa"
    legacy = cache_root / "bonesistools" / "nestorowa" / "full"
    filenames = [filename for filename, _ in _NESTOROWA_SOURCES.values()]
    if all((standard / filename).exists() for filename in filenames):
        return standard
    if all((legacy / filename).exists() for filename in filenames):
        return legacy
    return standard


def _load_nestorowa_source(cache_dir: Path) -> AnnData:
    paths = {
        key: cache_dir / filename
        for key, (filename, _) in _NESTOROWA_SOURCES.items()
    }
    for key, path in paths.items():
        if not path.exists():
            _download(_NESTOROWA_SOURCES[key][1], path)

    read_counts = pd.read_csv(paths["counts"], index_col=0, sep="\t").transpose()
    published = pd.read_csv(
        paths["published"],
        index_col=0,
        sep=r"\s+",
    ).transpose()
    cell_types = pd.read_csv(
        paths["cell_types"],
        index_col=0,
        sep="\t",
    )
    cluster_ids = pd.read_csv(
        paths["clusters"],
        header=None,
        index_col=0,
        names=["cluster"],
        sep=r"\s+",
        dtype="category",
    )["cluster"]

    published = published.loc[:, published.columns.str.startswith("ENS")]
    read_counts = read_counts.loc[published.index, published.columns]
    cell_types = cell_types.loc[cluster_ids.index, :]
    metadata, label_counts = _resolve_cell_metadata(cell_types, cluster_ids)

    obs = pd.DataFrame(index=read_counts.index.astype(str))
    obs["label"] = metadata.loc[obs.index, "label"].astype("category")
    obs["clusters"] = _numeric_cluster_categories(cluster_ids).loc[obs.index]
    var = pd.DataFrame(index=read_counts.columns.astype(str))

    adata = AnnData(
        X=published.to_numpy(dtype=np.float32),
        obs=obs,
        var=var,
    )
    adata.layers["counts"] = csr_matrix(read_counts.to_numpy())
    adata.uns["nestorowa"] = {
        "source": {key: url for key, (_, url) in _NESTOROWA_SOURCES.items()},
        "cache_dir": str(cache_dir),
        "labels": label_counts,
    }
    return adata


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "scBOLT-case-study/1.0"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _publish_count_input(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_cell_metadata(
    cell_types: pd.DataFrame,
    cluster_ids: pd.Series,
) -> tuple[pd.DataFrame, dict[str, int]]:
    cell_labels = pd.DataFrame(index=cell_types.index)
    for label, columns in _LABEL_GROUPS.items():
        cell_labels[label] = (cell_types[columns].sum(axis=1) != 0).astype(int)

    metadata = pd.DataFrame(index=cell_labels.index, columns=["label", "cluster"])
    label_counts = {"unique": 0, "multiple": 0, "missing": 0}
    for cell in cell_labels.index:
        selected = cell_labels.loc[cell, :] > 0
        n_labels = int(selected.sum())
        if n_labels == 1:
            label = str(cell_labels.columns[selected][0])
            label_counts["unique"] += 1
        elif n_labels > 1:
            label = _seeded_choice(tuple(cell_labels.columns[selected]))
            label_counts["multiple"] += 1
        else:
            label = _seeded_choice(
                tuple(_CLUSTER_LABEL_FALLBACKS[str(cluster_ids[cell])])
            )
            label_counts["missing"] += 1
        metadata.loc[cell] = [label, cluster_ids[cell]]

    return metadata, label_counts


def _seeded_choice(values: tuple[str, ...]) -> str:
    return str(np.random.RandomState(2020).choice(values, 1)[0])


def _numeric_cluster_categories(cluster_ids: pd.Series) -> pd.Series:
    mapping = {
        category: index
        for index, category in enumerate(cluster_ids.cat.categories)
    }
    return cluster_ids.cat.rename_categories(mapping)


def _filter_published_expression(adata: AnnData) -> None:
    expression = np.asarray(adata.X)
    adata.obs["n_features_published"] = np.count_nonzero(
        expression >= 1,
        axis=1,
    )
    adata.var["n_barcodes_published"] = np.count_nonzero(
        expression >= 1,
        axis=0,
    )
    bt.omics.pp.filter_obs(
        adata,
        "n_features_published",
        lambda values: values >= 100,
    )
    bt.omics.pp.filter_var(
        adata,
        "n_barcodes_published",
        lambda values: values >= 5,
    )
    del adata.var["n_barcodes_published"]


def _select_lowess_hvg(
    adata: AnnData,
    *,
    expression: str,
    n_features: int,
    span: float,
    curve_neighbors: int,
) -> None:
    matrix = np.asarray(adata.layers[expression])
    if not 0 < n_features <= adata.n_vars:
        raise ValueError(
            f"n_features must be in [1, {adata.n_vars}], received {n_features}"
        )
    if not 0 < span <= 1:
        raise ValueError(f"span must be in (0, 1], received {span}")
    if curve_neighbors < 1:
        raise ValueError("curve_neighbors must be positive")

    means = np.mean(matrix, axis=0)
    standard_deviations = np.std(matrix, ddof=1, axis=0)
    trend = lowess(
        standard_deviations,
        means,
        return_sorted=False,
        frac=span,
    )
    scores = _signed_point_to_curve_distances(
        means,
        standard_deviations,
        trend,
        n_neighbors=curve_neighbors,
    )

    ordered = np.argsort(scores)[::-1]
    selected_indices = ordered[:n_features]
    selected = np.zeros(adata.n_vars, dtype=bool)
    selected[selected_indices] = True
    ranks = np.full(adata.n_vars, np.nan, dtype=float)
    ranks[selected_indices] = np.arange(1, n_features + 1, dtype=float)

    adata.var["highly_variable"] = selected
    adata.var["highly_variable_rank"] = ranks
    adata.var["highly_variable_score"] = scores
    adata.var["highly_variable_mean"] = means
    adata.var["highly_variable_std"] = standard_deviations
    adata.var["highly_variable_trend"] = trend
    adata.obsm["var_genes"] = matrix[:, selected_indices].copy()
    adata.uns["highly_variable"] = {
        "method": "stream_point_to_lowess_curve",
        "n_features": n_features,
        "n_selected": n_features,
        "expression": expression,
        "params": {
            "span": span,
            "ddof": 1,
            "curve_neighbors": curve_neighbors,
        },
    }


def _signed_point_to_curve_distances(
    means: np.ndarray,
    standard_deviations: np.ndarray,
    trend: np.ndarray,
    *,
    n_neighbors: int,
) -> np.ndarray:
    order = np.argsort(means)
    curve = np.column_stack((means[order], trend[order]))
    points = np.column_stack((means, standard_deviations))
    if curve.shape[0] < 2:
        raise ValueError("at least two features are required for HVG scoring")

    query_neighbors = min(n_neighbors, curve.shape[0])
    _, nearest_vertices = cKDTree(curve).query(points, k=query_neighbors)
    if nearest_vertices.ndim == 1:
        nearest_vertices = nearest_vertices[:, None]
    segments = np.clip(
        np.concatenate((nearest_vertices - 1, nearest_vertices), axis=1),
        0,
        curve.shape[0] - 2,
    )

    starts = curve[segments]
    vectors = curve[segments + 1] - starts
    offsets = points[:, None, :] - starts
    squared_lengths = np.sum(vectors * vectors, axis=2)
    projections = np.divide(
        np.sum(offsets * vectors, axis=2),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths != 0,
    )
    projections = np.clip(projections, 0, 1)
    closest = starts + projections[:, :, None] * vectors
    distances = np.sqrt(
        np.sum((points[:, None, :] - closest) ** 2, axis=2)
    ).min(axis=1)
    distances[standard_deviations - trend < 0] *= -1
    return distances


def _plot_hvg(adata: AnnData, outfile: Path) -> None:
    means = np.asarray(adata.var["highly_variable_mean"], dtype=float)
    standard_deviations = np.asarray(adata.var["highly_variable_std"], dtype=float)
    trend = np.asarray(adata.var["highly_variable_trend"], dtype=float)
    selected = np.asarray(adata.var["highly_variable"], dtype=bool)
    order = np.argsort(means)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        means[~selected],
        standard_deviations[~selected],
        s=4,
        alpha=0.2,
        color="#4C78A8",
        label="Other genes",
        rasterized=True,
    )
    ax.scatter(
        means[selected],
        standard_deviations[selected],
        s=5,
        alpha=0.75,
        color="#D62728",
        label="Highly variable genes",
        rasterized=True,
    )
    ax.plot(means[order], trend[order], color="#1F4E79", linewidth=2)
    ax.set(xlabel="Mean log2 expression", ylabel="Standard deviation")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def _map_and_merge_gene_symbols(
    adata: AnnData,
    *,
    identifiers: bt.resources.ncbi.GeneIdentifiers,
    biomart_cache: Path,
) -> dict[str, int]:
    original_ensembl = adata.var_names.astype(str).copy()
    adata.var["ensembl"] = original_ensembl
    bt.omics.pp.convert_gene_identifiers(
        adata,
        axis="var",
        input_type="ensembl_id",
        output_type="symbol",
        identifiers=identifiers,
        copy=False,
    )

    ncbi_names = adata.var_names.astype(str)
    ncbi_resolved = int((~ncbi_names.str.startswith("ENS")).sum())
    biomart_symbols = _load_biomart_symbols(biomart_cache)
    fallback = biomart_symbols.reindex(original_ensembl).to_numpy(dtype=object)
    replace = np.asarray(
        ncbi_names.str.startswith("ENS") & pd.notna(fallback) & (fallback != "")
    )
    updated_names = ncbi_names.to_numpy(copy=True)
    updated_names[replace] = fallback[replace]
    adata.var_names = pd.Index(updated_names, dtype=str)

    bt.omics.pp.convert_gene_identifiers(
        adata,
        axis="var",
        input_type="name",
        output_type="symbol",
        identifiers=identifiers,
        copy=False,
    )
    _propagate_duplicate_var_metadata(adata)

    n_vars_before_merge = adata.n_vars
    bt.omics.pp.merge_duplicate_vars(
        adata,
        keep="first",
        varm="mean",
        copy=False,
    )
    adata.var["symbol"] = adata.var_names.astype(str)
    adata.var["gene_symbol_resolved"] = ~adata.var_names.astype(str).str.startswith(
        "ENS"
    )

    unresolved = int((~adata.var["gene_symbol_resolved"]).sum())
    merged_duplicates = n_vars_before_merge - adata.n_vars
    adata.uns["gene_identifier_mapping"] = {
        "primary": "bonesistools bundled NCBI gene_info",
        "fallback": _BIOMART_URL,
        "fallback_cache": str(biomart_cache),
        "ncbi_resolved": ncbi_resolved,
        "biomart_rescued": int(replace.sum()),
        "merged_duplicates": merged_duplicates,
        "unresolved_ensembl": unresolved,
        "duplicate_strategy": "sum expression and preserve all Ensembl IDs",
    }
    adata.uns["highly_variable"]["n_selected_after_symbol_merge"] = int(
        adata.var["highly_variable"].sum()
    )
    return {
        "ncbi_resolved": ncbi_resolved,
        "biomart_rescued": int(replace.sum()),
        "merged_duplicates": merged_duplicates,
        "unresolved": unresolved,
    }


def _propagate_duplicate_var_metadata(adata: AnnData) -> None:
    groups: dict[str, list[int]] = {}
    for position, name in enumerate(adata.var_names.astype(str)):
        groups.setdefault(name, []).append(position)

    ensembl = adata.var["ensembl"].astype(str).to_numpy(copy=True)
    selected = np.asarray(adata.var["highly_variable"], dtype=bool).copy()
    ranks = np.asarray(adata.var["highly_variable_rank"], dtype=float).copy()
    scores = np.asarray(adata.var["highly_variable_score"], dtype=float).copy()

    for positions in groups.values():
        identifiers = list(dict.fromkeys(ensembl[positions]))
        merged_ensembl = "|".join(identifiers)
        merged_selected = bool(selected[positions].any())
        finite_ranks = ranks[positions][np.isfinite(ranks[positions])]
        merged_rank = float(finite_ranks.min()) if finite_ranks.size else np.nan
        merged_score = float(scores[positions].max())
        for position in positions:
            ensembl[position] = merged_ensembl
            selected[position] = merged_selected
            ranks[position] = merged_rank
            scores[position] = merged_score

    adata.var["ensembl"] = ensembl
    adata.var["highly_variable"] = selected
    adata.var["highly_variable_rank"] = ranks
    adata.var["highly_variable_score"] = scores


def _load_biomart_symbols(cache_file: Path) -> pd.Series:
    if cache_file.exists():
        table = pd.read_csv(
            cache_file,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
    else:
        table = _download_biomart_symbols()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(
            cache_file,
            sep="\t",
            index=False,
            compression="gzip",
        )

    table = _clean_biomart_table(table)
    return table.set_index("ensembl")["symbol"]


def _download_biomart_symbols() -> pd.DataFrame:
    query = urllib.parse.urlencode({"query": _BIOMART_QUERY})
    errors: list[str] = []
    opener = urllib.request.build_opener(_PermanentRedirectHandler())

    for endpoint in _BIOMART_ENDPOINTS:
        request = urllib.request.Request(
            f"{endpoint}?{query}",
            headers={"User-Agent": "scBOLT-case-study/1.0"},
        )
        try:
            with opener.open(request, timeout=30) as response:
                encoding = response.headers.get_content_charset() or "utf-8"
                response_text = response.read().decode(encoding)
        except (OSError, urllib.error.URLError) as error:
            errors.append(f"{endpoint}: {error}")
            continue

        if response_text.lstrip().startswith("Query ERROR"):
            errors.append(f"{endpoint}: {response_text[:500]}")
            continue

        try:
            table = pd.read_csv(
                StringIO(response_text),
                sep="\t",
                header=None,
                names=["ensembl", "symbol"],
                dtype=str,
                keep_default_na=False,
            )
            table = _clean_biomart_table(table)
        except (pd.errors.ParserError, ValueError) as error:
            errors.append(f"{endpoint}: invalid response ({error})")
            continue

        if len(table) < 10_000 or not table["ensembl"].str.startswith("ENSMUSG").all():
            errors.append(f"{endpoint}: incomplete mouse gene-symbol table")
            continue
        return table

    details = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(f"BioMart download failed:\n{details}")


class _PermanentRedirectHandler(urllib.request.HTTPRedirectHandler):
    def http_error_308(self, request, response, code, message, headers):
        return self.http_error_307(request, response, 307, message, headers)


def _clean_biomart_table(table: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ensembl", "symbol"}
    if not required_columns.issubset(table.columns):
        raise ValueError(
            f"invalid BioMart cache: expected columns {sorted(required_columns)!r}"
        )

    cleaned = table.loc[:, ["ensembl", "symbol"]].copy()
    cleaned = cleaned[
        cleaned["ensembl"].ne("") & cleaned["symbol"].ne("")
    ].drop_duplicates()
    symbols_per_identifier = cleaned.groupby("ensembl")["symbol"].nunique()
    ambiguous = symbols_per_identifier[symbols_per_identifier > 1].index
    cleaned = cleaned[~cleaned["ensembl"].isin(ambiguous)]
    return cleaned.drop_duplicates("ensembl", keep="first")


if __name__ == "__main__":
    main()
