#!/bin/bash
#$ -N ablation_guides
#$ -t 1-10
#$ -cwd
#$ -j y
#$ -o ablation/logs/guides_$TASK_ID.log
#$ -q long

# Condition G (guides), primary tier, R=10 replicates.
# One SGE array task per replicate; SGE_TASK_ID supplies the index (1..10),
# so no manual substitution of the command's trailing index is needed.
#
# Submit with:
#   mkdir -p ablation/logs
#   qsub ablation/conversion_guides/submit_guides_replicates.sh
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
# /groups is not auto-mounted by this cluster's apptainer.conf, so it must be
# bound explicitly for the container to see this repo.

set -euo pipefail

REPLICATE="${SGE_TASK_ID}"
REPO_ROOT="/groups/jli9/Yufei/python-KernSmooth"
OUT_DIR="${REPO_ROOT}/ablation/conversion_guides/guides/${REPLICATE}"
IMAGE="${REPO_ROOT}/ablation/claude_env.img"

cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

apptainer exec --bind /groups "${IMAGE}" \
  claude -p "/convert-r-file-to-python KernSmooth/R/all.R structural_analysis/R/all.json structural_analysis/dependency_levels.csv language_dependency_analysis/conversion_guides/ ${OUT_DIR}" \
  --dangerously-skip-permissions
