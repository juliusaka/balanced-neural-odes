import hydra
import os
import sys
import numpy as np
import h5py
from pathlib import Path
import logging
from datetime import datetime
import shutil
from omegaconf import OmegaConf

from config import data_gen_config, cs, convert_cfg_to_dataclass, RawDataClass
from filepaths import filepath_raw_data, log_overwriting_file, filepath_raw_data_config, filepath_dataset, filepath_dataset_config

def load_and_validate_raw_data(cfg):
    """
    loads the config file, validates it and compares it to the current config.
    loads the raw data file
    """
    path_raw_data = filepath_raw_data(cfg)
    path_raw_data_config = filepath_raw_data_config(cfg)
    
    if not path_raw_data.exists():
        raise FileNotFoundError(f'Raw data file does not exist: {path_raw_data}')
    if not path_raw_data_config.exists():
        raise FileNotFoundError(f'Raw data config file does not exist: {path_raw_data_config}')
    
    # load raw data config
    logging.info('Loading raw data config from {}'.format(path_raw_data_config))
    _raw_data_config_dict = OmegaConf.load(path_raw_data_config)
    _raw_data_config_dict = OmegaConf.to_object(_raw_data_config_dict) # make dict
    raw_data_config = RawDataClass(**_raw_data_config_dict) # validate and convert to dataclass
    logging.info('Validated raw data config from {}'.format(path_raw_data_config))

    # compare raw data config to actual config and raise errors / warnings
    logging.info('Comparing raw data config to current config. Creating copy of raw data config without creation date for comparison.')
    _raw_data_config_wo_creation_date = RawDataClass(**_raw_data_config_dict)
    _raw_data_config_wo_creation_date.creation_date = None
    _flag = False
    if cfg.pModel.RawData != _raw_data_config_wo_creation_date:
        for key in cfg.pModel.RawData.__dict__.keys():
            if cfg.pModel.RawData.__dict__[key] != _raw_data_config_wo_creation_date.__dict__[key]:
                logging.warning(f'Raw data config does not match current config. Specifically key {key} does not match.')
                _flag = True
    if _flag:
        logging.info('Overwriting raw data config with raw data config loaded from {}'.format(path_raw_data_config))
        cfg.pModel.RawData = raw_data_config
    else: 
        logging.info('Current config matches loaded raw data config. No overwriting of raw data config.')

    # load raw data
    raw_data = h5py.File(path_raw_data, 'r')
    logging.info('Loaded raw data from {}'.format(path_raw_data))

    return raw_data, raw_data_config

def transform_raw_data(cfg: data_gen_config, temp_raw_data: h5py.File, raw_data_config: RawDataClass):
    """
    performs transformations on raw data according to the config
    """

    def get_position_in_raw_data_file(variable: str):
        # returns dataset name and position in dataset
        search_datasets = [key for key in temp_raw_data.keys() if key.endswith('names')]
        temp = []
        for dataset in search_datasets:
            _temp = np.array(temp_raw_data[dataset], dtype=str)
            if variable in _temp:
                temp.append([dataset, np.where(_temp == variable)[0][0]])
        if len(temp) == 0:
            raise ValueError(f'Variable {variable} not found in raw data file.')
        elif len(temp) > 1:
            raise ValueError(f'Variable {variable} found in multiple datasets in raw data file.')
        else:
            temp[0][0] = temp[0][0].replace('_names', '')
            return temp[0]

    for variable in cfg.pModel.dataset_prep.transforms.keys():
        dataset_name, idx = get_position_in_raw_data_file(variable)
        if cfg.pModel.dataset_prep.transforms[variable] == 'temperature_k_to_degC':
            temp_raw_data[dataset_name][:,idx] = temp_raw_data[dataset_name][:,idx] - 273.15
        elif cfg.pModel.dataset_prep.transforms[variable] == 'power_w_to_kw':
            temp_raw_data[dataset_name][:,idx] = temp_raw_data[dataset_name][:,idx] / 1000
        else:
            raise NotImplementedError(f'Transform {cfg.pModel.dataset_prep.transforms[variable]} not implemented.')
        logging.info(f'Transformed variable {variable} in dataset {dataset_name} with transform {cfg.pModel.dataset_prep.transforms[variable]}')
    pass

def replace_hdf5_dataset(dataset_name: str, raw_data: h5py.File, data: np.ndarray):
    """
    replaces dataset in raw data file with new data
    """
    if dataset_name not in raw_data.keys():
        raise ValueError(f'Dataset {dataset_name} not found in raw data file.')
    if data.shape != raw_data[dataset_name].shape:
        del raw_data[dataset_name]
        raw_data.create_dataset(dataset_name, data=data)
    else:
        raw_data[dataset_name][...] = data

@hydra.main(config_path=str(Path('conf').absolute()), config_name='data_gen_HeatPump', version_base=None)
def main(cfg: data_gen_config):
    cfg = convert_cfg_to_dataclass(cfg)
    
    # load and validate raw data, copy data to temp file
    raw_data, raw_data_cfg = load_and_validate_raw_data(cfg)
    temp_raw_data_path = Path('temp_raw_data.hdf5')
    temp_raw_data = h5py.File(temp_raw_data_path, 'w')
    for key in raw_data.keys():
        raw_data.copy(key, temp_raw_data)
    logging.info('Copied raw data to temporary file {}'.format(temp_raw_data_path))
    
    # remove failed runs from raw data
    failed_idx = raw_data['failed_idx']
    failed_idx = np.array(failed_idx, dtype=int)
    failed_idx = np.sort(failed_idx)
    logging.info('Removing failed runs from raw data: {}'.format(failed_idx))
    for key in ['states', 'states_der', 'controls', 'outputs', 'parameters']:
        if key in temp_raw_data.keys():
            _temp= np.delete(temp_raw_data[key][:], failed_idx, axis=0)
            replace_hdf5_dataset(key, temp_raw_data, data = _temp)
            logging.info('\tRemoved failed runs from {}'.format(key))
        else:
            logging.info('\tNo {} in raw data. Skipping removal of failed runs.'.format(key))
    raw_data_cfg.n_samples = raw_data_cfg.n_samples - len(failed_idx)
    logging.info('Updated n_samples in raw_data_config to {}'.format(raw_data_cfg.n_samples))

    # perform transforms on raw data
    transform_raw_data(cfg, temp_raw_data, raw_data_cfg)

    # only select variables of interest / states, controls, outputs, parameters

    # helper functions
    def get_idx(names_list: h5py.Dataset, chosen_variables: list):
        if chosen_variables == ['all'] or chosen_variables == ['der(all)']:
            return np.arange(len(names_list))
        else:
            names_list = np.array(names_list, dtype=str).tolist()
            return [names_list.index(variable) for variable in chosen_variables]
    
    def select_variables_of_interest(type: str, variables: list):
        # type is states, states_der, controls, outputs, parameters
        if type in temp_raw_data.keys():
            _type_with_names = f'{type}_names' # type_names_str = 'states_names' or 'states_der_names' or 'controls_names' or 'outputs_names' or 'parameters_names'
            idx = get_idx(temp_raw_data[_type_with_names], variables) # get idx
            replace_hdf5_dataset(type, temp_raw_data, data = temp_raw_data[type][:,idx]) # replace dataset
            replace_hdf5_dataset(_type_with_names, temp_raw_data, data = temp_raw_data[_type_with_names][idx])
            logging.info(f'Selected {type} {variables} from raw data.')
        else:
            logging.info(f'No {type} in raw data. Skipping selection of {type}.')
    
    logging.info('... Selecting variables of interest in raw data.')
    select_variables_of_interest('states', cfg.pModel.dataset_prep.states)
    select_variables_of_interest('states_der', ['der({})'.format(state) for state in cfg.pModel.dataset_prep.states])
    select_variables_of_interest('controls', cfg.pModel.dataset_prep.controls)
    select_variables_of_interest('outputs', cfg.pModel.dataset_prep.outputs)
    select_variables_of_interest('parameters', cfg.pModel.dataset_prep.parameters)

    # only select certain timeframe
    def idx_timeframe(time: np.ndarray, start_time: float, end_time: float):
        idx = np.where((time >= start_time) & (time <= end_time))[0]
        logging.info(f'... Selecting timeframe from {start_time} to {end_time} in raw data.')
        return idx
    
    def replace_timeseries_if_exist(idx, dataset_name: str, raw_data: h5py.File):
        if dataset_name in raw_data.keys():
            replace_hdf5_dataset(dataset_name, raw_data, data = raw_data[dataset_name][:,:,idx])
            logging.info(f'Selected timeframe from {dataset_name} in raw data.')
        else:
            logging.info(f'No {dataset_name} in raw data. Skipping selection of {dataset_name}.')
    
    idx = idx_timeframe(temp_raw_data['time'][:], cfg.pModel.dataset_prep.start_time, cfg.pModel.dataset_prep.end_time)
    cfg.pModel.dataset_prep.sequence_length = len(idx)
    replace_hdf5_dataset('time', temp_raw_data, data = temp_raw_data['time'][idx])
    replace_timeseries_if_exist(idx, 'states', temp_raw_data)
    replace_timeseries_if_exist(idx, 'states_der', temp_raw_data)
    replace_timeseries_if_exist(idx, 'controls', temp_raw_data)
    replace_timeseries_if_exist(idx, 'outputs', temp_raw_data)

    #############################################################
    # special routines, e.g. chunking together from 0 to N for each time-series
    
    # could be added here

    #############################################################

    # save common test and validation sets to temporary raw data file
    logging.info('opening common test and validation sets')
    temp_raw_data.create_group('common_test')
    temp_raw_data.create_group('common_validation')

    # determine idx in raw data set of test and validation sets
    validation_idx_start_total = int(np.floor(raw_data_cfg.n_samples * (1 - cfg.pModel.dataset_prep.validation_fraction - cfg.pModel.dataset_prep.test_fraction)))
    test_idx_start_total = int(np.floor(raw_data_cfg.n_samples * (1 - cfg.pModel.dataset_prep.test_fraction)))
    
    # save idx to cfg
    cfg.pModel.dataset_prep.validation_idx_start = validation_idx_start_total
    cfg.pModel.dataset_prep.test_idx_start = test_idx_start_total
    logging.info('set validation_idx_start to {}, test_idx_start to {} in cfg.'.format(validation_idx_start_total, test_idx_start_total))

    # save common validation and test sets
    for key in ['states', 'states_der', 'controls', 'outputs', 'parameters']:
        if key in temp_raw_data.keys():
            temp_raw_data.create_dataset('common_validation/' + key, data=temp_raw_data[key][validation_idx_start_total:test_idx_start_total])
            temp_raw_data.create_dataset('common_test/' + key, data=temp_raw_data[key][test_idx_start_total:])
            logging.info('Saved common test and validation sets for {} to temporary raw data file.'.format(key))
        else:
            logging.info('No {} in raw data. Skipping saving common test and validation sets for {}.'.format(key, key))

    # add generation date to datasets
    creation_date = datetime.now()
    temp_raw_data.attrs['creation_date'] = str(creation_date)

    _reached_max_samples = False
    # sample dataset sizes and save datasets
    for n_samples_dataset in cfg.pModel.dataset_prep.n_samples:
        if _reached_max_samples:
            logging.warning('Reached maximum number of samples in raw data. Skipping further dataset creation.')
            break
        if n_samples_dataset > raw_data_cfg.n_samples:
            logging.warning('n_samples_dataset must be smaller than n_samples in raw data. Setting n_samples_dataset={} to n_samples={}'.format(n_samples_dataset, raw_data_cfg.n_samples))
            n_samples_dataset = raw_data_cfg.n_samples
            _reached_max_samples = True
        path_dataset = filepath_dataset(cfg, n_samples_dataset)
        log_overwriting_file(path_dataset)
        dataset_file = h5py.File(path_dataset, 'w')
        dataset_file.create_dataset('time', data=temp_raw_data['time'])
        for key in ['states', 'states_der', 'controls', 'outputs', 'parameters']:
            if key in temp_raw_data.keys():
                if n_samples_dataset > raw_data_cfg.n_samples:
                    raise ValueError('n_samples_dataset must be smaller than n_samples in raw data. Reaching this line should not happen.')
                dataset_file.create_dataset(key + '_names', data=temp_raw_data[key + '_names'])
                # get idx
                train_idx_stop = int((n_samples_dataset/raw_data_cfg.n_samples) * validation_idx_start_total)
                common_validation_idx_stop = int((n_samples_dataset/raw_data_cfg.n_samples) * len(temp_raw_data['common_validation/' + key]))
                common_test_idx_stop = int((n_samples_dataset/raw_data_cfg.n_samples) * len(temp_raw_data['common_test/' + key]))
                # train, validate, test
                dataset_file.create_dataset('train/' + key, data = temp_raw_data[key][:train_idx_stop])
                dataset_file.create_dataset('validation/' + key, data=temp_raw_data['common_validation/' + key][:common_validation_idx_stop])
                dataset_file.create_dataset('test/' + key, data=temp_raw_data['common_test/' + key][:common_test_idx_stop])
                
                logging.info('Saved {} data with {} samples to {}'.format(key, n_samples_dataset, path_dataset))
                # add common datasets
                dataset_file.create_dataset('common_validation/' + key, data=temp_raw_data['common_validation/' + key])
                dataset_file.create_dataset('common_test/' + key, data=temp_raw_data['common_test/' + key])
                logging.info('Added common test and validation sets for {} to {} dataset.'.format(key, path_dataset))
            else:
                logging.info('No {} in raw data. Skipping saving {}'.format(key, key))
        # save config: create new config object, set raw data config, set n_samples to n_samples_dataset, add preparation info
        _conf = OmegaConf.create(cfg)
        _conf.pModel.RawData = raw_data_cfg
        _conf.pModel.dataset_prep.n_samples = [n_samples_dataset]
        # add preparation info
        _conf.pModel.dataset_prep = cfg.pModel.dataset_prep
        path_dataset_config = filepath_dataset_config(cfg, n_samples_dataset)
        log_overwriting_file(path_dataset_config)
        OmegaConf.save(_conf.pModel, path_dataset_config)
        logging.info('Saved pModel config to {}'.format(path_dataset_config))
        # close dataset
        dataset_file.attrs['creation_date'] = str(creation_date)
        dataset_file.close()
        logging.info('Closed dataset {}'.format(path_dataset))

    # delete temporary file
    temp_raw_data.close()
    os.remove(temp_raw_data_path)
    pass

if __name__ == '__main__':
    main()