# Code of Paper: Balanced Neural ODEs: nonlinear model order reduction and koopman operator approximation

# Set up the project

Recommended setup:
 - a GPU with CUDA and 4GB+ of VRAM
 - a CPU with at least 4 cores and 32GB+ RAM (less for linux, multiprocessing is more efficient)
 - some gigabytes of free space (20GB should be enough for the beginning)
 - Windows (the FMUs require Windows, but the code runs on Linux as well). Cross-Compile FMUs for Linux can be generated with Dymola.

[1] Create a python environment with python 3.9.18

    conda create -n "bnode" python=3.9

[2] Install torch 2.3, preferably with CUDA support (code supports CPU and CUDA)
     https://pytorch.org/get-started/locally/

    for Windows and Linux:

    # CUDA 11.8
    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
    # CUDA 12.1
    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    # CPU Only
    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 cpuonly -c pytorch

[3] install packages as in requirements.txt

    pip install -r requirements.txt

[Alternatively] torchdiffeq should is downloaded from pip as well. To enable local changes, you can download torchdiffeq as specified in .gitmodules and use it as a submodule. To install it, run from the "python" directory

    python torchdiffeq/setup.py install

[4] Install this repository editable
    (this will install with setuptools the packages defined in "setup.py" (i.e. the code of this project))
  
    pip install --editable .

please see that

    config.py
    filepath.py

are already in the python root if you run this file from the root 'python' and are not included in the setup.py
But as the root path is added to the sys path by installing setup.py, they can be opened from anywhere.

# Using the code
The repo has several functionalities.

1) Generating datasets with Co-Simulation FMUs, e.g. exported from Dymola.
    https://fmi-standard.org/ ; https://www.claytex.com/tech-blog/fmi-basics-co-simulation-vs-model-exchange/

2) Train Variational Autoencoders with it (VAE) [for the example in 2.2]

3) Train time-stepper surrogates: State Space Neural ODEs and Balanced Neural ODEs

4) View Results with plot_gui.py

5) Run the analysis required for generating the figures in our paper.

This functionalities are explained in the following sections.
But first, we will explain the usage of hydra and mlflow in this project.

### Usage of hydra

This code uses hydra to differentiate between code and parameters/settings, that are stored seperatedly in configuration files (/conf/...).

https://hydra.cc/docs/intro/

- config.py describes pydantic dataclasses, that configuration files are validated with. It therefore uses a object-oriented approach.

- The configuration files are stored in the conf directory and are yaml files. We differentiate between code for physical models ('pModel') and neural networks ('nn_model').

- you can override the configuration files with command line arguments, e.g.:

        python my_app.py param_real=1.0 param_bool=true

    and start multiruns with (don't forget the -m flag)

        python my_app.py param_real=1.0,2.0 param_bool=true,false -m

- hydra is added as a decorator to the main function of the called file, e.g.:

        @hydra.main(config_path="conf", config_name="data_gen")

- hydra will save the validated configuration files and the logging information under *outputs/DATE/TIME/.hydra* or *multiruns/DATE/TIME/RUN/.hydra*. We can access this directory as well to store checkpoints or other files. At the end of the run, we log all of this information to mlflow artifacts, so that we can access it in an organized way in the mlflow web interface. (decorator "log_hydra_to_mlflow")

### Usage of mlflow for experiment tracking

This code uses mlflow to track experiments and store metrics and artifacts. Start it in a seperate terminal with

    mlflow server

if you want to change the artifacts location, you can run

    mlflow server--artifacts-destination file:///mnt/e/mlflow/mlartifacts

This for example sets the artifacts locations on a linux machine to a directory on the e-drive. 

You can also set the artifacts location and the backend store location as environment variables, e.g. as in the following example

    export MLFLOW_ARTIFACTS_DESTINATION=file:///mnt/e/mlflow/mlartifacts
    export MLFLOW_BACKEND_STORE_URI=file:///${HOME}/docker/mlflow/mlruns

Note: mlflow needs to run before starting code involving training, as metrics and artifacts are saved to mlflow.
We recommend to set the environment variable MLFLOW_ARTIFACTS_DESTINATION to use the function *filepath_from_ml_artifacts_uri* from *filepath.py* to resolve the path to the mlflow artifacts directory.

see https://mlflow.org/docs/latest/tracking.html for more information

see at the end of this file for "mlflow garbage collection" if you want to delete runs and experiments that were marked as deleted in the Web UI.

## 1) Generate datasets

This envolves the files raw_data_generation.py and data_preparation.py in the data_generation directory. *raw_data_generation.py* generates the raw data from the FMUs with multiprocessing. *data_preparation.py* prepares the data for training, this includes splitting the data into training and test sets, and can include cutting the time series lengths, converting units etc.

To start, fill out the configuration file in *conf/pModel*

After filling out the fmu_path from a config file, you can use

        python data_generation/src/print_fmu_variable_names.py

to help you fill out the config file. This will print all variables in the FMU in the required format. The pModel file will then contain ranges variables are sampled, solver settings etc.

To generate and prepare the data, run
    
        python data_generation/raw_data_generation pModel=myModel

        python data_generation/data_preparation.py pModel=myModel

We provide in this Repo the Modelica Code as well as Co-Simulation FMUs generated by Dymola from the Code to reproduce the results of our paper. The FMUs are stored in *../../modelica/...*. The FMUs require Windows.
To generate the data used in our paper, run with the following pModel configs:

    python data_generation/raw_data_generation.py pModel=ThermalNetwork,SHF,SHF_step,KoopmanTu,PowerPlant,PowerPlant_step -m
    
    python data_generation/data_preparation.py pModel=ThermalNetwork,SHF,SHF_step,KoopmanTu,PowerPlant,PowerPlant_step -m

!!! The SteamCycle_01 model will run several hours. The other models should run in a few minutes.

Datasets are stored in *../data/raw_data* or *../data/datasets*.

We decided to use the datasets as a unified exchange format, with following variable types:

    - states
    - outputs
    - controls
    - parameters
    - time
 
 Like this, the model training scripts will automatically adjust to the data dimensions and hydra files only contain nn_model configurations.

 additionally, variable names are saved '*xxx_names'

dimension convention:

     n_samples x n_variables x n_timepoints

## 2) Train Variational Autoencoders

To train a VAE, you can use the configuration file in *conf/vae* and the file *networks/vae/vae_train_test.py*. 
The superclass configuration file is *conf/train_test_vae.yaml*. (Remember to start mlflow)

    python networks/vae/vae_train_test.py dataset_name=myDataset nn_model=my_nn_model

To train the VAE used in our paper, run with the following nn_model configs:

    python networks/vae/vae_train_test.py nn_model=vae dataset_name=ThermalNetwork_v1_p-R__n-2048


To view the results, you can use the plot_gui.py file in the utils/plots directory.

    python utils/plots/plot_gui.py

    >> Enter path to hdf5 file:

    C:/Users/exampleUser/outputs/DATE/TIME/dataset.hdf5

## 3) Train time-stepper surrogates

As State Space Neural ODE and B-NODEs both contain Neural ODEs, they require a highly adapted training procedure. We decided to use the same training script for both, and define unified interfaces for the neural models to be trained. The training script is located in *networks/neural_ode/trainer.py*. The superclass configuration file is *conf/train_test_neural_ode.yaml*.

To train the time-stepper surrogates, you can use the configuration files in *conf/nn_model* and the file *networks/nn_model/trainer.py*. (Remember to start mlflow)

    python networks/nn_model/trainer.py dataset_name=myDataset nn_model=my_nn_model

To train the State Space Neural ODE of our paper, run:

To reproduce training of the B-NODEs and NODEs of our paper, run the trainings specified in *commands_for_experiments.md*

To test the models with other data sets, retrieve the model checkpoint from mlflow artifacts and run

    python networks/nn_model/trainer.py dataset_name=myDataset nn_model=my_nn_model nn_model.training.load_trained_model_for_test=True nn_model.training.path_trained_model=PATH_TO_MODEL

## 4) View Results with plot_gui.py

To view the results, you can use the plot_gui.py file in the utils/plots directory.

    python utils/plots/plot_gui.py

    >> Enter path to hdf5 file:

    C:/Users/exampleUser/outputs/DATE/TIME/dataset.hdf5

## 5) Analysis for generating the figures in our paper

The analysis for generating the figures in our paper is done in the *analysis* directory.
To each script, you need to provide the path to datasets of the respective model. The datasets can be retrieved from mlflow artifacts.

# Additional information

## mlflow garbage collection

To perform garbage collection (delete all runs and experiments that were marked as deleted in the Web UI) in mlflow on a local host, you can set the environment variables `MLFLOW_TRACKING_URI` and `MLFLOW_ARTIFACT_URI`. Set for example

    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_ARTIFACT_URI=${MLFLOW_ARTIFACTS_DESTINATION}

This will set the tracking URI to `http://localhost:5000` and the artifact URI to the `mlartifacts` directory defined by the MLFLOW_ARTIFACTS_DESTINATION environment variable. With these environment variables set, mlflow will be able to properly manage and clean up the artifacts and tracking data.

Then run

    mlflow gc

     mlflow gc --artifacts-destination file:///mnt/e/mlflow/mlartifacts --backend-store-uri file:////home/akajuliu/documents/balanced-neural-odes-internal/code/python/mlruns

to delete all files permanently that are associated with the runs and experiments that were marked as deleted in the Web UI.