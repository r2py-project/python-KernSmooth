#!/bin/bash
#$ -N ablation_placebo
#$ -t 1-10
#$ -cwd
#$ -j y
#$ -o ablation/logs/placebo_$TASK_ID.log
#$ -q long

# Condition P (placebo guides), primary tier, R=10 replicates.
# One SGE array task per replicate; SGE_TASK_ID supplies the index (1..10),
# so no manual substitution of the command's trailing index is needed.
#
# Submit with:
#   mkdir -p ablation/logs
#   qsub ablation/conversion_guides/submit_placebo_replicates.sh
#
# --dangerously-skip-permissions is required for a headless/array job: there is
# no terminal attached to approve tool calls, so the run must be pre-authorized
# to proceed unattended. This means every tool call (file writes, bash commands
# the conversion agent issues, etc.) executes without a human checkpoint for the
# full duration of the job.
#
# There is no admin-installed `claude` on this cluster, so the CLI runs from
# the Apptainer image built via ablation/claude_env.def:
#   apptainer build --fakeroot ablation/claude_env.img ablation/claude_env.def
#
# Apptainer auto-binds the host's current working directory by default, which
# would silently expose the whole repo -- including r2py_kernsmooth/ (the
# finished reference implementation this task is supposed to reproduce from
# scratch) and every other replicate's/condition's output directory (inviting
# copycat results instead of independent conversions). --no-mount cwd disables
# that default, and only the specific paths this replicate needs are bound
# explicitly (each to its own identical in-container path, so the command's
# relative-path arguments below need no changes). Anything not listed here
# -- r2py_kernsmooth/, sibling replicates, other conditions -- does not exist
# inside the container's mount namespace at all.

set -euo pipefail

REPLICATE="${SGE_TASK_ID}"
REPO_ROOT="/groups/jli9/Yufei/python-KernSmooth"
OUT_DIR="${REPO_ROOT}/ablation/conversion_guides/placebo/${REPLICATE}"
IMAGE="${REPO_ROOT}/ablation/claude_env.img"

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

BINDS=(
  --bind "${REPO_ROOT}/KernSmooth:${REPO_ROOT}/KernSmooth:ro"
  --bind "${REPO_ROOT}/structural_analysis:${REPO_ROOT}/structural_analysis:ro"
  --bind "${REPO_ROOT}/ablation/conversion_guides/placebo_conversion_guides:${REPO_ROOT}/ablation/conversion_guides/placebo_conversion_guides:ro"
  --bind "${REPO_ROOT}/.claude:${REPO_ROOT}/.claude:ro"
  --bind "${OUT_DIR}:${OUT_DIR}"
)

apptainer exec --no-mount cwd --pwd "${REPO_ROOT}" "${BINDS[@]}" "${IMAGE}" \
  claude -p "/convert-r-file-to-python KernSmooth/R/all.R structural_analysis/R/all.json structural_analysis/dependency_levels.csv ablation/conversion_guides/placebo_conversion_guides/ ${OUT_DIR}" \
  --model claude-sonnet-5 \
  --dangerously-skip-permissions
