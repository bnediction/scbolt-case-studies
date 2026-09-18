#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCBOLT_HOME="${SCBOLT_HOME:-${ROOT_DIR}/../../scbolt}"
BONESIS_ENV="${BONESIS_ENV:-scbolt-bonesis}"
PROJECT_CONFIG_TOOL="${SCBOLT_HOME}/scripts/utils/project_config.py"

if [[ -z "${CONFIG:-}" && -f .scbolt ]]; then
  while IFS='=' read -r key value; do
    if [[ "${key}" == "CONFIG" ]]; then
      CONFIG="${value%$'\r'}"
      break
    fi
  done < .scbolt
fi
CONFIG="${CONFIG:-config/params-abc.yml}"

config_dump="$({
  conda run --no-capture-output -n "${BONESIS_ENV}" \
    python "${PROJECT_CONFIG_TOOL}" export "${CONFIG}"
})"

declare -A config_values=()
while IFS=$'\t' read -r key value; do
  case "${key}" in
    JOBS|MAX_CLAUSES|PRIOR_KNOWLEDGE|ORGANISM|\
    GENEINFO_VERSION|OMNIPATH_VERSION|HCOP_VERSION|DOROTHEA_API|\
    DOROTHEA_COMPATIBILITY|DOROTHEA_LEVELS|BOUNDED_NONREACH)
      config_values["${key}"]="${value}"
      ;;
  esac
done <<< "${config_dump}"

JOBS="${JOBS:-${config_values[JOBS]:-1}}"
MAX_CLAUSES="${MAX_CLAUSES:-${config_values[MAX_CLAUSES]:-8}}"
PRIOR_KNOWLEDGE="${PRIOR_KNOWLEDGE:-${config_values[PRIOR_KNOWLEDGE]:-dorothea}}"
ORGANISM="${ORGANISM:-${config_values[ORGANISM]:-mouse}}"
GENEINFO_VERSION="${GENEINFO_VERSION:-${config_values[GENEINFO_VERSION]:-bundled}}"
OMNIPATH_VERSION="${OMNIPATH_VERSION:-${config_values[OMNIPATH_VERSION]:-latest}}"
HCOP_VERSION="${HCOP_VERSION:-${config_values[HCOP_VERSION]:-bundled}}"
DOROTHEA_API="${DOROTHEA_API:-${config_values[DOROTHEA_API]:-modern}}"
DOROTHEA_COMPATIBILITY="${DOROTHEA_COMPATIBILITY:-${config_values[DOROTHEA_COMPATIBILITY]:-true}}"
DOROTHEA_LEVELS="${DOROTHEA_LEVELS:-${config_values[DOROTHEA_LEVELS]:-A B C}}"
BOUNDED_NONREACH="${BOUNDED_NONREACH:-${config_values[BOUNDED_NONREACH]:-}}"

TIMEOUT="${TIMEOUT:-48h}"
CLINGO_MODE="${CLINGO_MODE:-${CLINGO_OPT_MODE:-opt}}"
CLINGO_OPT_STRATEGY="${CLINGO_OPT_STRATEGY:-usc}"
CLINGO_CONFIGURATION="${CLINGO_CONFIGURATION:-}"
NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scbolt-mpl-cache}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RESULTS_DIR="${RESULTS_DIR:-results}"
SPEC_DIR="${RESULTS_DIR}/spec"
RUN_ROOT="${RESULTS_DIR}/historical_inference_algorithm"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-${RUN_ROOT}/${RUN_ID}}"

for required_file in model.bo mstates.csv mandatory.txt forbidden.txt; do
  if [[ ! -f "${SPEC_DIR}/${required_file}" ]]; then
    printf 'error: missing %s; run scripts/build_notebook_data.py first\n' \
      "${SPEC_DIR}/${required_file}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}" "${OUTPUT}"
if [[ -e "${RUN_ROOT}/latest" && ! -L "${RUN_ROOT}/latest" ]]; then
  printf 'error: %s exists and is not a symbolic link\n' \
    "${RUN_ROOT}/latest" >&2
  exit 1
fi
if [[ "${OUTPUT}" != "${RUN_ROOT}/latest" ]]; then
  ln -sfn "$(realpath "${OUTPUT}")" "${RUN_ROOT}/latest"
fi

read -r -a dorothea_levels <<< "${DOROTHEA_LEVELS}"

command=(
  conda run --no-capture-output -n "${BONESIS_ENV}"
  python -u scripts/historical_inference_algorithm.py
  "${SPEC_DIR}/model.bo"
  "${SPEC_DIR}/mstates.csv"
  --asp "${OUTPUT}/objective.txt"
  --solution "${OUTPUT}/best/retained_nodes.txt"
  --output "${OUTPUT}"
  --mandatory-nodes "${SPEC_DIR}/mandatory.txt"
  --forbidden-nodes "${SPEC_DIR}/forbidden.txt"
  --domain "${PRIOR_KNOWLEDGE}"
  --organism "${ORGANISM}"
  --geneinfo-version "${GENEINFO_VERSION}"
  --omnipath-version "${OMNIPATH_VERSION}"
  --hcop-version "${HCOP_VERSION}"
  --dorothea-api "${DOROTHEA_API}"
  --dorothea-compatibility "${DOROTHEA_COMPATIBILITY}"
  --dorothea-levels "${dorothea_levels[@]}"
  --bonesis-mode hard
  --max-clauses "${MAX_CLAUSES}"
  --canonical true
  --clingo-mode "${CLINGO_MODE}"
  --clingo-opt-strategy "${CLINGO_OPT_STRATEGY}"
  --jobs "${JOBS}"
  --timeout "${TIMEOUT}"
)

if [[ -n "${BOUNDED_NONREACH}" ]]; then
  command+=(--bounded-nonreach "${BOUNDED_NONREACH}")
fi
if [[ -n "${CLINGO_CONFIGURATION}" ]]; then
  command+=(--clingo-configuration "${CLINGO_CONFIGURATION}")
fi

export SCBOLT_HOME NUMBA_DISABLE_JIT MPLCONFIGDIR PYTHONUNBUFFERED

{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'config=%s\n' "${CONFIG}"
  printf 'output=%s\n' "$(realpath "${OUTPUT}")"
  printf 'domain=full %s; dorothea_api=%s; dorothea_levels=%s\n' \
    "${PRIOR_KNOWLEDGE}" "${DOROTHEA_API}" "${DOROTHEA_LEVELS}"
  printf 'objective=maximize_nodes then maximize_strong_constants\n'
  printf 'command='
  printf ' %q' "${command[@]}"
  printf '\n'
} | tee "${OUTPUT}/run.log"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi

"${command[@]}" 2>&1 | tee -a "${OUTPUT}/run.log"
