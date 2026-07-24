#!/bin/bash
#$ -N ablation_no_guides_combine
#$ -t 1-10
#$ -cwd
#$ -j y
#$ -o ablation/logs/no_guides_combine_$TASK_ID.log
#$ -q long

# Condition NG (no guides), primary tier, R=10 replicates.
# One SGE array task per replicate; SGE_TASK_ID supplies the index (1..10),
# so no manual substitution of the command's trailing index is needed.
#
# Submit with:
#   mkdir -p ablation/logs
#   qsub ablation/conversion_guides/submit_no_guides_combine.sh
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
SRC_DIR="${REPO_ROOT}/ablation/conversion_guides/no_guides/${REPLICATE}/all.R"
OUT_DIR="${REPO_ROOT}/ablation/conversion_guides/no_guides/${REPLICATE}_kernsmooth"
TARGET_FILE="${OUT_DIR}/r2py_kernsmooth/__init__.py"
IMAGE="${REPO_ROOT}/ablation/claude_env.img"

cd "${REPO_ROOT}"

BINDS=(
  --bind "${REPO_ROOT}/.claude:${REPO_ROOT}/.claude:ro"
  --bind "${SRC_DIR}:${SRC_DIR}:ro"
  --bind "${OUT_DIR}:${OUT_DIR}"
)

apptainer exec --no-mount cwd --pwd "${REPO_ROOT}" "${BINDS[@]}" "${IMAGE}" \
  claude -p "/combine-python-functions-into-file ${SRC_DIR} ${TARGET_FILE}" \
  --model claude-sonnet-5 \
  --dangerously-skip-permissions
