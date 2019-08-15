#!/bin/bash
#SBATCH --job-name=PyHEIFConcat.py_TRANCODE
#SBATCH --time=3:0:0
#SBATCH --mem=1G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if [ -z "${SLURM_TMPDIR}" ]; then
  SLURM_TMPDIR=./tmp
fi
if [ -z "${SLURM_JOB_ID}" ]; then
  SLURM_JOB_ID=1
fi
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
  SLURM_ARRAY_TASK_ID=1
fi

SRC=$1
DEST=$2
REMOTE=
#REMOTE=beluga
TMP=${SLURM_TMPDIR}/${SLURM_ARRAY_TASK_ID}
#TMP=./data/${USER}/slurm-${SLURM_JOB_ID}/${SLURM_ARRAY_TASK_ID}
NUMBER=12
START=$((${NUMBER} * (${SLURM_ARRAY_TASK_ID} - 1)))

echo PWD=${PWD}
echo PATH=${PATH}
echo SLURM_TMPDIR=${SLURM_TMPDIR}
echo SLURM_JOB_ID=${SLURM_JOB_ID}
echo SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}
echo SRC=${SRC}
echo DEST=${DEST}
echo REMOTE=${REMOTE}
echo TMP=${TMP}
echo START=${START}
echo NUMBER=${NUMBER}
echo .
echo python pyheifconcat.py extract_archive hdf5 "${SRC}" "${DEST}" \
  --start ${START} --number ${NUMBER} \
  --transcode --ssh-remote "${REMOTE}" --tmp "${TMP}"

pyenv activate pyheifconcat

python pyheifconcat.py extract_archive hdf5 "${SRC}" "${DEST}" \
  --start ${START} --number ${NUMBER} \
  --transcode --ssh-remote "${REMOTE}" --tmp "${TMP}"

rm -Rf ${TMP}
