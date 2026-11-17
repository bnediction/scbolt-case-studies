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
conda run -n scbolt-core python scripts/load_nestorowa.py
```

The loader retrieves the mouse gene-symbol table from the Ensembl 116 BioMart
archive and caches it under `project/omics/annot/`.

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

### Monolithic inference algorithm

Once `results/` has been generated, the original joint
max-nodes/max-strong-constants algorithm can be run from the exported
specification. Three JSON profiles describe the experiments. The first captures
the reported `usc` run; the two others test the model-guided `bb,lin`
optimisation strategy with different levels of Clingo internal parallelism. All
profiles use the same inputs, five-clause bound, and 48-hour time limit:

```bash
./scripts/run_monolithic_algorithm.py config/benchmarks/mono_usc.json
./scripts/run_monolithic_algorithm.py config/benchmarks/mono_bb_lin.json
./scripts/run_monolithic_algorithm.py config/benchmarks/multi_bb_lin.json
```

`mono_usc.json` and `mono_bb_lin.json` use one Clingo thread;
`multi_bb_lin.json` uses 16 competing Clingo threads
(`--parallel-mode=16,compete`). Each JSON file records the benchmark-specific
parameters and is copied into its result directory. Runs are stored separately
under `results/monolithic_algorithm/<experiment>_<timestamp>/`; none is launched
automatically. For a controlled comparison, run the profiles sequentially on
the same compute node and with the same memory limit. The original `mono_usc`
server log, with its shell prompt removed and its output path made
repository-relative, is archived at
`results/monolithic_algorithm/mono_usc_20260921_091849.log`.
This archived transcript predates the consolidated JSON launcher; the
`mono_usc.json` profile records the equivalent current configuration.

Each run writes a uniquely named `<experiment>_<timestamp>.log` directly under
`results/monolithic_algorithm/`, while its other artefacts are stored in the
homonymous result directory. The log records the start and end of ASP grounding,
the transition into solving, and every incumbent with its wall-clock time. The
files `run.json` and `status.json` record the solver settings, current execution
phase, grounding duration, and final status. Full intermediate solutions are
appended durably to `incumbents.jsonl`, including their timestamps, elapsed
times, objective counts, and exact node sets. The current best node sets are
also updated under `best/` whenever the objective improves. The launcher
automatically uses the `scbolt-bonesis` Conda environment and expects the
scBOLT source checkout at `../../scbolt`; set `BONESIS_ENV` or `SCBOLT_HOME`
when either is stored elsewhere.

Launch the analyses:

```bash
conda run -n scbolt-cs-stream jupyter lab notebooks/omics.ipynb
conda run -n scbolt-cs jupyter lab notebooks/bn.ipynb
```

- `notebooks/omics.ipynb`: STREAM trajectory and macrostate analyses.
- `notebooks/bn.ipynb`: Boolean-network comparison, structure, enrichment, and
  attractor analyses.

Figures are written to `figures/`.

The Chevalier comparison uses the published 39-component model stored in
`resources/models/chevalier.bnet`. The original workflow obtained this final
model as the complement of 71 components marked as strong constants in the
retained solution over a 110-component intermediate domain. The DoRothEA prior
induced by these 39 components contains the 137 interactions reported in the
[publication](https://doi.org/10.1038/s41540-025-00569-z).
