#!/bin/bash
#SBATCH --job-name=PyHEIFConcat.py_CONCAT
#SBATCH --time=3:0:0
#SBATCH --mem=1G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

SRC=$1
DEST=$2

echo PWD=${PWD}
echo PATH=${PATH}
echo SRC=${SRC}
echo DEST=${DEST}
echo .
echo python pyheifconcat.py concat "${SRC}" "${DEST}"

pyenv activate pyheifconcat

python pyheifconcat.py concat "${SRC}" "${DEST}"
