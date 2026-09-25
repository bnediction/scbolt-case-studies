# APL case study

Reproduction of the PLZF-RARα acute promyelocytic leukemia (APL) analyses presented in the scBOLT manuscript, based on the dataset from [Poplineau et al. (2022)](https://ashpublications.org/blood/article/140/22/2358/486349/Noncanonical-EZH2-drives-retinoic-acid-resistance).

## Environment

From `apl/`:

```bash
conda env create -f ../envs/scbolt-cs.yml
conda activate scbolt-cs
```

## Entry points

| Parameter file  | Input source       | Output directory |
| --------------- | ------------------ | ---------------- |
| `params-gsm.yml` | GEO count matrices | `project_gsm/`   |
| `params-sra.yml` | SRA raw reads      | `project_sra/`   |

`params-gsm.yml` is the recommended entry point for reproducing the manuscript results.

`params-sra.yml` demonstrates the use of the full scBOLT framework starting from raw sequencing reads.

> **Note (SRA entry point only)**
> Unlike the GSM-based reproduction path, this entry point depends on genome annotations and RepeatMasker tracks retrieved from upstream providers. As these resources are not distributed through archived releases, some intermediate results may vary over time.

## Reproducing the analyses

### From GEO matrices

Initialize the project:

```bash
scbolt init params-gsm.yml
```

Before running the inference, you may inspect the project configuration and dependencies with:

```bash
scbolt check bn-submin
```

Run the inference:

```bash
scbolt bn-submin
```

### From SRA reads

Run the project:

```bash
scbolt init params-sra.yml
scbolt bn-submin
```

### Additional analyses

Potency scores reported in the manuscript can be reproduced from the GSM-based reproduction path with:

```bash
scbolt init params-gsm.yml
scbolt potency
```

Runtime, CPU, memory, and energy-related statistics for a full APL run can be
collected with:

```bash
scripts/run_pipeline_turbostat.sh
```

The wrapper runs `bn-submin` after resetting `load-matrix` and `load-cc`.
Run logs and summary tables are written to `stat/`.

## Notebooks

The `notebooks/` directory contains the readable figure-generation notebooks used during the preparation of the manuscript:

- `notebooks/omics.ipynb`: transcriptomic integration, potency analysis, macrostate characterisation, and binarisation diagnostics.
- `notebooks/bn.ipynb`: Boolean-network specification, inference, and downstream analyses.

After running `bn-submin` and `potency`, export the results required by the notebooks:

```bash
python scripts/build_notebook_data.py --force
```

The script detects the macrostate method available under `project_gsm/`, copies only
the Boolean-network models and configurations, preserves the selected-gene list and
inference specification, and prepares the compact notebook inputs under `results/`.
Figures are exported to `figures/`.

Regenerate the GO enrichment tables:

```bash
python scripts/goea.py results/omics/integrated.h5ad
```

Outputs are written to `results/goea/`.

Launch one of the notebooks:

```bash
jupyter lab notebooks/omics.ipynb
jupyter lab notebooks/bn.ipynb
```

If Jupyter does not list the environment as a kernel, register it once:

```bash
python -m ipykernel install --user --name scbolt-cs --display-name "Python (scbolt-cs)"
```

## Repository structure

```text
apl/
├── params-gsm.yml      # parameters for the GEO-based entry point
├── params-sra.yml      # parameters for the SRA-based entry point
├── spec.yml           # BoNesis specification
├── notebooks/         # exploratory analyses
├── resources/         # external published models
├── results/           # compact reproducibility artefacts
└── figures/           # manuscript figure generation
```
