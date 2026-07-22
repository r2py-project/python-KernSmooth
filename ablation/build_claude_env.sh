#!/bin/bash
#$ -pe smp 1         # Specify parallel environment and legal core size
#$ -q long           # Specify queue
# Builds the Apptainer image used by ablation/conversion_guides/submit_guides_replicates.sh.
# Run from the repo root:
#   ablation/build_claude_env.sh
set -euo pipefail

REPO_ROOT="/groups/jli9/Yufei/python-KernSmooth"
cd "${REPO_ROOT}"

apptainer build --fakeroot ablation/claude_env.img ablation/claude_env.def
