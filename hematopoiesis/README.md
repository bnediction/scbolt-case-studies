# Hematopoiesis case study

Reproduction of the Nestorowa hematopoiesis analyses used in the scBOLT
manuscript.

## Environments

From `hematopoiesis/`, create the two notebook environments if needed:

```bash
conda env create -f ../envs/scbolt-cs-stream.yml
conda env create -f ../envs/scbolt-cs.yml
```

The scBOLT command and its runtime environments must also be installed.

## Reproducing the inference

`config/params-abc.yml` is the reference configuration. It uses the modern
mouse DoRothEA A/B/C prior and enumerates up to 1,000 subset-minimal Boolean
networks.

The tracked `.scbolt` locator already selects this configuration. Generate the
preprocessed Nestorowa dataset:

```bash
conda run -n scbolt-stream python scripts/load_nestorowa.py
```

Check and run the complete scBOLT workflow:

```bash
scbolt check bn-submin
scbolt bn-submin
```

The inferred networks are written under `project/infer/bn/submin/`.

## Notebooks

Export the compact results required by the notebooks:

```bash
conda run -n scbolt-cs-stream python scripts/build_notebook_data.py --force
```

This creates `results/` with the inferred networks, compact omics objects,
selected-gene list, and inference specification.

### Historical inference algorithm

Once `results/` has been generated, run the original joint
max-nodes/max-strong-constants algorithm directly from the exported specification:

```bash
CONFIG=config/params-abc.yml TIMEOUT=48h scripts/run_historical_inference_algorithm.sh
```

Each incumbent and its wall-clock time are recorded under
`results/historical_inference_algorithm/latest/`. The wrapper expects the scBOLT
source checkout at `../../scbolt`; set `SCBOLT_HOME` when it is stored elsewhere.

Launch the analyses:

```bash
conda run -n scbolt-cs-stream jupyter lab notebooks/omics.ipynb
conda run -n scbolt-cs jupyter lab notebooks/bn.ipynb
```

- `notebooks/omics.ipynb`: STREAM trajectory and macrostate analyses.
- `notebooks/bn.ipynb`: Boolean-network comparison, structure, enrichment, and
  attractor analyses.

Figures are written to `figures/`.

The published Chevalier Boolean model used for comparison is stored under
`resources/models/`.
