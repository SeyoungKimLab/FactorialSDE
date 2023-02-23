# Factorial SDE for Multi-Output Gaussian Process Regression
This is the GitHub repository for the paper "Factorial SDE for Multi-Output Gaussian Process Regression". This repository includes implementations of the factorial SDE (FSDE) and the factorial SDE with sparse variational inference (FSDE-SVI), along with all of the baselines and the Python scripts used for experiments.

<br>

## 1. Setup
All of the factorial SDE models were implemented in [JAX](https://github.com/google/jax), and the baseline models were implemented in [GPflow](https://github.com/gpflow/gpflow). We provide the exported [conda](https://docs.conda.io/en/latest/) environment `.yaml` files for both Linux and Mac OS to replicate the setting under which all implementations were tested.

To set up the environment that we used for all experiments, you can take the following steps:

### 1.1. Install Conda
If you do not already have conda installed on your local machine, please install conda following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### 1.2. Import Conda Environments for JAX and GPflow
All of the exported [conda](https://docs.conda.io/en/latest/) environment `.yaml` files are provided under `./env`. We provide separate conda environments for running the factorial SDE implementations in JAX and for running the baseline model implementations in GPflow.

To import and create a new conda environment for JAX, run:
```
conda env create -f ./env/fsde_env_linux.yaml (if you are using Linux)
conda env create -f ./env/fsde_env_mac.yaml (if you are using Mac OS)
```

To import and create a new conda environment for GPflow, run:
```
conda env create -f ./env/fsde_gpflow_env_linux.yaml (if you are using Linux)
conda env create -f ./env/fsde_gpflow_env_mac.yaml (if you are using Mac OS)
```

You can also refer to conda's [documentation on managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information.

### 1.3. Activate the Conda Environment
After Step 1.2, you can check whether the environment (named `fsde_env`) was successfully created by running:
```
conda env list
```
which lists all of the conda environments available on your local machine. If `fsde_env` and `fsde_gpflow_env` are also listed, then you can activate each by running:
```
conda activate fsde_env (if you want the JAX environment)
conda activate fsde_gpflow_env (if you want the GPflow environment)
```

<br><br>

## 2. Running the Code

Below, we provide instructions for running each of the scripts that we used for experiments on simulated and real-world datasets. Both the simulated and real-world datasets are included under `./data`, and specific experimental settings (e.g., learning rate, number of inducing points) are hardcoded into each script. For detailed descriptions of the datasets, the preprocessing steps, and the overall experimental setups, please refer to Section 5 of the paper and the Supplementary Material.

Throughout, we assume the optimization setting where we use Adam to optimize the kernel hyperparameters and natural gradients to optimize the variational parameters. To also optimize the variational parameters with Adam, simply remove the `--natgrad` argument.

### 2.1. Small Simulation Data (Small-Sim): 5-Fold Cross-Validation 
To run the 5-fold cross-validation experiments on the Small-Sim data, move to `./fsde/sim_exps_small` and run the following on the command line:

```
python3 small_sim_fsde_CV.py --fsde --natgrad (Factorial SDE)
python3 small_sim_fsde_CV.py --svi --natgrad (Factorial SDE with Sparse VI)
python3 small_sim_baselines_CV.py --lmc_gpr (LMC with exact inference)
python3 small_sim_baselines_CV.py --imc_svgp --natgrad (IMC with Sparse VI)
python3 small_sim_baselines_CV.py --lmc_svgp --natgrad (LMC with Sparse VI)
```

All results will be saved under `./fsde/sim_exps_small/sim_results` in `.pkl` format.

### 2.2.1. Large Simulation Data (Large-Sim): 5-Fold Cross-Validation
To run the 5-fold cross-validation experiments on the Large-Sim data, move to `./fsde/sim_exps_large` and run the following command:

```
python3 large_sim_fsde_CV.py --svi --natgrad (Factorial SDE with Sparse VI)
python3 large_sim_baselines_CV.py --imc_svgp --natgrad (IMC with Sparse VI)
python3 large_sim_baselines_CV.py --lmc_svgp --natgrad (LMC with Sparse VI)
```

Note that the 5-fold cross-validation experiments on Large-Sim data use 1,000 inducing points. All results will be saved under `./fsde/sim_exps_large/sim_results` in `.pkl` format.

### 2.2.2. Large Simulation Data (Large-Sim): Runtime vs. Number of Inducing Points
To run the runtime benchmarking experiments with varying numbers of inducing points, move to `./fsde/sim_exps_large` and run the following command:
```
python3 runtime_fsde_svi.py --ind_id <Number between 1 and 4> --natgrad (Factorial SDE with SVI)
python3 runtime_imc.py --ind_id <Number between 1 and 4> --natgrad (IMC baseline)
python3 runtime_lmc.py --ind_id <Number between 1 and 4> --natgrad (LMC baseline)
```
The `--ind_id` argument, which accepts numbers between 1 and 4, specifies the number of inducing points to use. 1 corresponds to 200 inducing points, 2 to 400 inducing points, 3 to 600 inducing points, and 4 to 800 inducing points. By default, the first of the 5-fold train-test splits of the Large-Sim dataset is used. To use a different train-test split, you can add the `--split_id` option, which accepts a number between 1 and 5 that specifies the particular train-test split. 

For example, to train the factorial SDE with SVI with 400 inducing points on the third train-test split of the Large-Sim dataset, you can run:
```
python3 runtime_fsde_svi.py --ind_id 2 --split_id 3 --natgrad
```

All results will be saved under `./fsde/sim_exps_large/runtime_results` in `.pkl` format.

### 2.3. Real Data Experiments
To run the real data experiments, move to `./fsde/real_exps` and run the following command:

```
python3 real_fsde_svi.py <Number between 1 and 4> --natgrad
python3 real_imc.py <Number between 1 and 4> --natgrad
python3 real_lmc.py <Number between 1 and 4> --natgrad
```

The required positional argument, which accepts numbers between 1 and 4, specifies which real dataset to run the experiment on. 1 corresponds to the COVID-19 dataset, 2 to the Stock dataset, 3 to the Energy dataset, and 4 to the Air Quality dataset. All results will be saved under `./fsde/real_exps/<dataset>_results` in `.pkl` format.
