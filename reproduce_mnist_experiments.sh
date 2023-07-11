RNG_SEEDS="42 142 242"
DATASETS="mnist emnist fmnist kmnist omni"

DATASETS=($DATASETS)
echo $DATASETS

# general parameters
# reduce the N_SUBSAMPLE to run only on the subsample
export N_SUBSAMPLE=10000
export SAMPLE_RANGE="0:$N_SUBSAMPLE"
export BASE_WORKDIR=$(pwd)

# handle the base directory
if [ -d $BASE_WORKDIR ]; then
  echo 'Base directory exists $BASE_WORKDIR'
else
  mkdir $BASE_WORKDIR
  echo 'Created the base directory at $BASE_WORKDIR'
fi

# run methods
# run the MC Dropout for all the datasets
export USE_METHOD_CONF='baseline_mcdo_mnist'
echo "Computing samples for method $USE_METHOD_CONF..."
for ds in "${DATASETS[@]}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$ds" "$rand_seed"
    export RNG_SEED=$rand_seed
    export RUN_ON_DATASET=$ds
    export DS_BATCHSIZE=1000 #$N_SUBSAMPLE
    export RUN_ON_SUBSAMPLE=$SAMPLE_RANGE
    echo "Running with seed $RNG_SEED on dataset $RUN_ON_DATASET"
    python -m source.run_method --config baseline_mcdo_mnist;
#    if python -m source.run_method --config baseline_mcdo_mnist; then
#      exit
#    fi
  done
done
# evaluate the ood detection
export ID_DATA=mnist
echo "Computing uncertainties from samples for method $USE_METHOD_CONF..."
for ds in "${DATASETS[@]:1}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$met" "$rand_seed"
    export RNG_SEED=$rand_seed
    export OOD_DATA=$ds
    echo "Running with seed $RNG_SEED on dataset OOD_DATA"
    python -m source.run_experiment --config ood_nontempered_prerun;
  done
done

# run laplace
export USE_METHOD_CONF='baseline_laplace_mnist'
echo "Computing samples for method $USE_METHOD_CONF..."
export LAPLACE_HESSIAN_STRUCTURE=kron
for ds in "${DATASETS[@]}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$ds" "$rand_seed"
    export RNG_SEED=$rand_seed
    export RUN_ON_DATASET=$ds
    export DS_BATCHSIZE=$N_SUBSAMPLE
    export RUN_ON_SUBSAMPLE=$SAMPLE_RANGE
    echo "Running with seed $RNG_SEED on dataset $RUN_ON_DATASET"
    python -m source.run_method --config baseline_laplace_mnist;
  done
done
# evaluate the ood detection
export ID_DATA=mnist
echo "Computing uncertainties from samples for method $USE_METHOD_CONF..."
for ds in "${DATASETS[@]:1}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$met" "$rand_seed"
    export RNG_SEED=$rand_seed
    export OOD_DATA=$ds
    echo "Running with seed $RNG_SEED on dataset OOD_DATA"
    python -m source.run_experiment --config ood_nontempered_prerun;
  done
done

# run sghmc
export USE_METHOD_CONF='baseline_sghmc_mnist'
echo "Computing samples for method $USE_METHOD_CONF..."
# default parameters in the file are sufficient
for ds in "${DATASETS[@]}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$ds" "$rand_seed"
    export RNG_SEED=$rand_seed
    export RUN_ON_DATASET=$ds
    export DS_BATCHSIZE=$N_SUBSAMPLE
    export RUN_ON_SUBSAMPLE=$SAMPLE_RANGE
    echo "Running with seed $RNG_SEED on dataset $RUN_ON_DATASET"
    python -m source.run_method --config baseline_sghmc_mnist;
  done
done
# evaluate the ood detection
export ID_DATA=mnist
echo "Computing uncertainties from samples for method $USE_METHOD_CONF..."
for ds in "${DATASETS[@]:1}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$met" "$rand_seed"
    export RNG_SEED=$rand_seed
    export OOD_DATA=$ds
    echo "Running with seed $RNG_SEED on dataset OOD_DATA"
    python -m source.run_experiment --config ood_nontempered_prerun;
  done
done

# run ensembles
export USE_METHOD_CONF='baseline_ensembles_mnist'
echo "Computing samples for method $USE_METHOD_CONF..."
# default parameters in the file are sufficient
for ds in "${DATASETS[@]}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$ds" "$rand_seed"
    export RNG_SEED=$rand_seed
    export RUN_ON_DATASET=$ds
    export DS_BATCHSIZE=$N_SUBSAMPLE
    export RUN_ON_SUBSAMPLE=$SAMPLE_RANGE
    echo "Running with seed $RNG_SEED on dataset $RUN_ON_DATASET"
    python -m source.run_method --config baseline_ensembles_mnist;
  done
done
# evaluate the ood detection
export ID_DATA=mnist
echo "Computing uncertainties from samples for method $USE_METHOD_CONF..."
for ds in "${DATASETS[@]:1}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$met" "$rand_seed"
    export RNG_SEED=$rand_seed
    export OOD_DATA=$ds
    echo "Running with seed $RNG_SEED on dataset OOD_DATA"
    python -m source.run_experiment --config ood_nontempered_prerun;
  done
done

# run sghmc
export USE_METHOD_CONF='baseline_quam_mnist'
echo "Computing samples for method $USE_METHOD_CONF..."

export ATTACK_N_CLASSES=10
export TOTAL_CLASSES=10
export OPT_EPOCHS=2
#export OPT_GAMMA=1.e-2
#export LR_SCHEDULE
export OPT_C0=6.0
export OPT_ETA=1.9
export OPT_UPDATE_C_EVERY=14
export OPT_LR=5.e-3
export OPT_WEIGHT_DECAY=1.e-3

for ds in "${DATASETS[@]}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$ds" "$rand_seed"
    export RNG_SEED=$rand_seed
    export RUN_ON_DATASET=$ds
    export DS_BATCHSIZE=$N_SUBSAMPLE
    export RUN_ON_SUBSAMPLE=$SAMPLE_RANGE
    echo "Running with seed $RNG_SEED on dataset $RUN_ON_DATASET"
    python -m source.run_method --config baseline_quam_mnist;
  done
done
# evaluate the ood detection
export ID_DATA=mnist
echo "Computing uncertainties from samples for method $USE_METHOD_CONF..."
for ds in "${DATASETS[@]:1}"; do
  for rand_seed in $RNG_SEEDS; do
    echo "$met" "$rand_seed"
    export RNG_SEED=$rand_seed
    export OOD_DATA=$ds
    echo "Running with seed $RNG_SEED on dataset OOD_DATA"
    python -m source.run_experiment --config ood_tempered_prerun;
  done
done
