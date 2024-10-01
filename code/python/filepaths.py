import os
import sys
import hydra
import logging

from pathlib import Path
from config import data_gen_config, convert_cfg_to_dataclass

def create_path(path: Path, log: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        if log:
            logging.info('Created path: {}'.format(path))
    else:
        if log:
            logging.info('Path already exists: {}'.format(path))
    
def log_overwriting_file(path: Path) -> None:
    if not path.exists():
        logging.info('Writing on file: {}'.format(path))
    else:
        logging.info('Overwriting file: {}'.format(path))

def raw_data_name(cfg: data_gen_config):
    RawData = cfg.pModel.RawData
    name = RawData.modelName + '_' + RawData.versionName 
    if RawData.initial_states_include:
        name = name + '_s-' + RawData.initial_states_sampling_strategy 
    if RawData.parameters_include:
        name = name + '_p-' + RawData.parameters_sampling_strategy
    if RawData.controls_include:
        name = name + '_c-' + RawData.controls_sampling_strategy
    return name

def dataset_name(cfg: data_gen_config, n_samples: int):
    name = raw_data_name(cfg) + '__n-' + str(n_samples)
    return name

def dir_raw_data(cfg: data_gen_config, log: bool = False):
    path = Path('data') / 'raw_data' / raw_data_name(cfg)
    create_path(path, log)
    return path

def filepath_raw_data(cfg: data_gen_config):
    file = dir_raw_data(cfg) / (raw_data_name(cfg) + '_raw_data.hdf5')
    return file

def filepath_raw_data_config(cfg: data_gen_config):
    file = dir_raw_data(cfg) / (raw_data_name(cfg) + '_RawData_config.yaml')
    return file

def dir_datasets(log=False):
    path = Path('data') / 'datasets'
    create_path(path, log)
    return path

def dir_specific_dataset_from_cfg(cfg: data_gen_config, n_samples: int, log: bool = False):
    path =  dir_datasets() / dataset_name(cfg, n_samples)
    create_path(path, log)
    return path

def dir_specific_dataset_from_name(name: str, log: bool = False):
    path =  dir_datasets() / name
    return path

def filepath_dataset(cfg: data_gen_config, n_samples: int):
    file = dir_specific_dataset_from_cfg(cfg, n_samples) / (dataset_name(cfg, n_samples) + '_dataset.hdf5')
    return file

def filepath_dataset_config(cfg: data_gen_config, n_samples: int):
    file = dir_specific_dataset_from_cfg(cfg, n_samples) / (dataset_name(cfg, n_samples) + '_pModel_config.yaml')
    return file

def filepath_dataset_from_name(name: str):
    file = dir_specific_dataset_from_name(name) / (name + '_dataset.hdf5')
    return file

def filepath_dataset_config_from_name(name: str):
    file = dir_specific_dataset_from_name(name) / (name + '_pModel_config.yaml')
    return file

def dir_current_hydra_output():
    return Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

def filepath_model_current_hydra_output(phase: int = None):
    if phase is not None:
        return dir_current_hydra_output() / 'model_phase_{}.pt'.format(phase)
    else:
        return dir_current_hydra_output() / 'model.pt'

def filepath_pretrained_model_current_hydra_output():
    return dir_current_hydra_output() / 'model_pretrained.pt'

def filepath_dataset_current_hydra_output():
    return dir_current_hydra_output() / 'dataset.hdf5'

def filepath_optimizer_current_hydra_output(phase: int = None):
    if phase is not None:
        return dir_current_hydra_output() / 'optimizer_phase_{}.pt'.format(phase)
    else:
        return dir_current_hydra_output() / 'optimizer.pt'
    

def filepath_from_ml_artifacts_uri(mlflow_uri):
    try:
        # try resolving the path from the environment variable
        _dir = os.environ['MLFLOW_ARTIFACTS_DESTINATION']
        _dir = _dir.replace('file://', '')
    except:
        logging.error('MLFLOW_ARTIFACTS_DESTINATION not set')
        logging.error('please set the environment variable MLFLOW_ARTIFACTS_DESTINATION to the path where the mlflow artifacts are stored')
        logging.error('or provide the full path to the dataset')
        raise ValueError('MLFLOW_ARTIFACTS_DESTINATION not set as environment variable')
    mlflow_uri = _dir + os.sep + mlflow_uri.replace('mlflow-artifacts:/', '')
    mlflow_uri = Path(mlflow_uri)
    return mlflow_uri

def filepath_from_local_or_ml_artifacts(mlflow_path: str):
    if mlflow_path.startswith('mlflow-artifacts:'):
        _path = filepath_from_ml_artifacts_uri(mlflow_path)
    else:
        _path = mlflow_path
    return Path(_path)

@hydra.main(config_path=str(Path('conf').absolute()), config_name='data_gen', version_base=None)
def test(cfg: data_gen_config):
    cfg = convert_cfg_to_dataclass(cfg)
    print(raw_data_name(cfg))
    print(dir_raw_data(cfg, log=True))
    print(filepath_raw_data(cfg))
    log_overwriting_file(filepath_raw_data(cfg))

if __name__ == '__main__':
    test()