# Technical Reference

## Setting up for ImageNet Reproduction
### Hardware Requirements
Laplace approximation was running OOM even on the CPU even with the last layer, so that does not work. For QUAM to work on the EfficientNet last layer at least 24GB VRAM is necessary if operating with `bfloat16` and relatively short trajectories. For full `float32` runs a GPU with 50GB+ memory is necessary. The QUAM trajectories, when working with the ID dataset, can get bulky for larger numbers of samples. It is recommended to have at least 512GB of RAM available when running at 128+ samples per run.

The default configuration files provided have the compression enabled by default. However, mind that you would need on the order of 100s of GB of free space to store them. SSD storage is recommended.

### Getting the ImageNet-1K dataset
Once the ImageNet dataset is downloaded, please refer to `./source/constants.py` and set the ImageNet folder location accordingly; or to '' if you only intend to run on mnist.

#### Pre-encoding the Datasets

Encoding the datasets (ImageNet, ImageNet-O, ImageNet-A) is done by

```commandline
python ./experiments/project_imagenet.py
```

### Obtain QUAM Results

```commandline
chmod +x run_final_imagenet_exp.sh
./run_final_imagenet_exp.sh
chmod +x run_final_imagenet_exp_o.sh
./run_final_imagenet_exp_o.sh
chmod +x run_final_imagenet_exp_a.sh
./run_final_imagenet_exp_a.sh
```

### Evaluation

Running the baseline methods and creating all plots is executed by

```commandline
python ./experiments/analyze_sweeps.py
```

## General Usage Notes
### Hyperparameter Search using wandb sweeps

#### Initialize the hyperpapameter space 

First of all, one needs to initialize thy sweep hyperparameter space. To do so, run the following:
```commandline
wandb sweep --project quam ./source/config/wandb_sweep_configs/mnist_test_sweep.yaml
```
Use the corresponding config files, preferably add new ones to the `./source/config/wandb_sweep_configs/`. 
The sweep is initialized once per hyperparameter search. This command should also provide you with a handle for the 
created sweep. You should use this handle subsequently for all workers that work on this sweep.  Once a sweep exists on 
the wandb (we use cloud, dont forget to wandb login --cloud if you are logged in locally), we can start a worker. 

#### Start a worker

CUDA_VISIBLE_DEVICES is necessary if running on multiple gpus within the same machine.

```commandline
export CUDA_VISIBLE_DEVICES=i  # i is the gpu number
export CUDA_DEVICE="cuda:0"   # if only i is visible, CUDA only sees GPU i as cuda:0
echo $CUDA_DEVICE
python -m source.run_experiment --config ood_hparams --n_tries 50 --use_wandb_sweep "quam/1nafa3dm"
```
The command line arguments are quite self-explanatory, use_wandb_sweep provides the project id / sweep id of the sweep 
that this worker is going to work for. If you do not supply that, it will run an experiment standalone, for which 
separate config files should be used (which, for example, do not rely on WANDB_NAME variable and hyperparameters to be 
set, unless they have default values specified. 

#### How do I config?

Config file are just normal yaml files with a twist. The files in the root of the `./source/config` directory can be addressed directly by `load_master_config`.
When loaded with that functions, the following happens:
- automatic loading of subconfig and addition of them to the main config dictionary. Everything that is not a asubdictionary is treated as a yaml that should be loaded from a certain subdirectory.
- variable substitution: everything of the form ${VARIABLE_NAME|default_value} is substituted for a variable from os.environ, unless a default_value is provided, which can be any yaml type. Variable are all parsed with yaml directly after substitution, so all the datatypes should be fine (as long as they are written in a way that yaml understands them (specifically pay attention to floats!))
  - most of the substitution variables are provided in env: sections in various configs, however, one can set those variables manually in the terminal by running `export VARIABLE_NAME="value"`
  - the variables will be overwritten, if they are specified in any env: clause. Those are present only in method and experiment top level configs, though.
  - when an experiment runs provided methods, the method env will overwrite the experiment env. Method envs will overwrite each other in order of execution.
