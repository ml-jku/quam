#!/bin/bash

export RUN_ON_DATASET=imagenet-a-projected
export RNG_SEED=42

export ID_DATA="imagenet-projected"
export RUN_ON_SUBSAMPLE=0:7000
export DS_DUBSET=test
export PUBWORK="/system/user/publicwork/student/aichberg/quam"
export MODEL_DIR="pretrained_models"
export USE_BFLOAT=true

for i in {0..34}
do
    export WANDB_RNAME="$i"
    python -m source.run_method --config=quam_inenc --para=$i/35
done
