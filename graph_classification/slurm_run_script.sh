#!/bin/bash

echo "SLURM ID ${SLURM_JOB_ID}"

module purge
module load PyTorch-Geometric/2.5.1
export PATH=`echo $PATH | tr ":" "\n" | grep -v "/net/tscratch/people/plgpraskim/conda/condabin" | tr "\n" ":"`

# ============
# Runtime
# ============

export PYTHONPATH=$PYTHONPATH:$(pwd)/libs/src

python eval_models.py -p ${PARAMS} \
                        -m ${MODEL} \
                        -d ${DATASET} \
                        -s accuracy \
                        -o ${CSV} \
                        --inner-splits ${IN_SPLITS} \
                        --outer-splits ${OUT_SPLITS} \
                        --partial-folds ${PARTIAL_FOLDS} \
                        --allow-fail
                        

