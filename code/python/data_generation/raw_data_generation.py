import hydra
import os
import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import logging
from omegaconf import OmegaConf
from datetime import datetime
from time import time, sleep
import dask
from dask.diagnostics import ProgressBar
from scipy.interpolate import CubicSpline, Akima1DInterpolator

from config import data_gen_config, cs, convert_cfg_to_dataclass
from filepaths import filepath_raw_data, log_overwriting_file, filepath_raw_data_config

def random_sampling_parameters(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.parameters[key][0], cfg.pModel.RawData.parameters[key][1]] for key in cfg.pModel.RawData.parameters.keys()]
    param_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.parameters.keys())))
    for i in range(len(cfg.pModel.RawData.parameters.keys())):
        param_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return param_values

def random_sampling_controls(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    for i in range(len(cfg.pModel.RawData.controls.keys())):
        ctrl_values[:, i, :] = np.random.uniform(bounds[i][0], bounds[i][1], (cfg.pModel.RawData.n_samples, cfg.pModel.RawData.Solver.sequence_length))
    # last control input is not used.
    return ctrl_values

def random_sampling_controls_w_offset(cfg: data_gen_config, seq_len: int = None, n_samples: int = None):
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples if n_samples is None else n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length if seq_len is None else seq_len))
    for j in range(ctrl_values.shape[0]):
        for i in range(len(cfg.pModel.RawData.controls.keys())):
            # get offset
            offset = np.random.uniform(bounds[i][0], bounds[i][1])
            # get amplitude
            amplitude = np.random.uniform(0, bounds[i][1] - bounds[i][0])
            # reduce amplitude if offset is close to bounds
            amplitude_upper = amplitude if bounds[i][1] - amplitude > offset else bounds[i][1] - offset
            amplitude_lower = amplitude if bounds[i][0] + amplitude < offset else offset - bounds[i][0]
            ctrl_values[j, i, :] = np.random.uniform(offset - amplitude_lower, offset + amplitude_upper, ctrl_values.shape[2])
    # last control input is not used.
    return ctrl_values

def random_sampling_controls_w_offset_cubic_splines_old_clip_manual(cfg: data_gen_config):
    '''
    also known as ROCS
    ROCS fills out the control space more than RROCS, because after the cubic spline interpolation, which tend to exceeds the bounds, 
    the values are clipped to the bounds.
    '''
    freq_sequence = np.random.choice(np.arange(cfg.pModel.RawData.controls_frequency_min_in_timesteps, cfg.pModel.RawData.controls_frequency_max_in_timesteps + 1), cfg.pModel.RawData.n_samples)
    # find out at which entry we reached the sequence length
    seq_len_sampling = np.where(np.cumsum(freq_sequence) > cfg.pModel.RawData.Solver.sequence_length)[0][0] + 1
    # sample data
    ctrl_values_sampled = random_sampling_controls_w_offset(cfg, seq_len_sampling+1)
    # create cubic splines
    x = np.concatenate((np.array([0]),
                       np.cumsum(freq_sequence[:seq_len_sampling]))
                       )
    xnew = np.arange(cfg.pModel.RawData.Solver.sequence_length)
    ctrl_values = CubicSpline(x, ctrl_values_sampled, axis=2)(xnew)
    # normalize values to bounds
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    for i in range(ctrl_values.shape[0]):
        for j in range(ctrl_values.shape[1]):
            min_val = np.min(ctrl_values[i, j, :])
            max_val = np.max(ctrl_values[i, j, :])

            exceeds_bounds = max_val - min_val > bounds[j][1] - bounds[j][0]
            delta = max_val - min_val if  exceeds_bounds else bounds[j][1] - bounds[j][0]

            # calculate base:
            if exceeds_bounds:
                base = bounds[j][0]
            elif min_val < bounds[j][0]:
                base = bounds[j][0]
            elif max_val > bounds[j][1]:
                base = bounds[j][1] - delta
            else:
                base = min_val
            ctrl_values[i, j, :] = (ctrl_values[i, j, :] - min_val) / delta * (bounds[j][1] - bounds[j][0]) + base
            if ctrl_values[i, j, :].min() < bounds[j][0] or ctrl_values[i, j, :].max() > bounds[j][1]:
                print('error in random_sampling_controls_w_offset_cubic_splines')
    return ctrl_values

def random_sampling_controls_w_offset_cubic_splines_clip_random(cfg: data_gen_config):
    '''
    also known as RROCS
    RROCS fills out the control space less than ROCS, because after the cubic spline interpolation, which tend to exceeds the bounds,
    the values are not just clipped to the bounds, but the base and delta are again randomly sampled.
    This ensures that on differen levels with differnt degree of variation, the control space is filled out more evenly.
    '''
    # freq_sequence = np.random.choice(np.arange(cfg.pModel.RawData.controls_frequency_min_in_timesteps, cfg.pModel.RawData.controls_frequency_max_in_timesteps + 1), cfg.pModel.RawData.n_samples)
    # # find out at which entry we reached the sequence length
    # seq_len_sampling = np.where(np.cumsum(freq_sequence) > cfg.pModel.RawData.Solver.sequence_length)[0][0] + 1
    # # sample data
    # ctrl_values_sampled = random_sampling_controls_w_offset(cfg, seq_len_sampling+1)
    # # create cubic splines
    # x = np.concatenate((np.array([0]),
    #                    np.cumsum(freq_sequence[:seq_len_sampling]))
    #                    )
    # xnew = np.arange(cfg.pModel.RawData.Solver.sequence_length)
    # ctrl_values = CubicSpline(x, ctrl_values_sampled, axis=2)(xnew)
    
    # normalize values to bounds
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    # loop over samples
    for i in range(ctrl_values.shape[0]):
        # loop over controls
        for j in range(ctrl_values.shape[1]):
            freq_sequence = np.random.choice(np.arange(cfg.pModel.RawData.controls_frequency_min_in_timesteps, cfg.pModel.RawData.controls_frequency_max_in_timesteps + 1), cfg.pModel.RawData.Solver.sequence_length)
            # find out at which entry we reached the sequence length
            seq_len_sampling = np.where(np.cumsum(freq_sequence) > cfg.pModel.RawData.Solver.sequence_length)[0][0] + 1
            # sample data
            ctrl_values_sampled = random_sampling_controls_w_offset(cfg, seq_len_sampling+1, n_samples=1)
            # create cubic splines
            x = np.concatenate((np.array([0]),
                            np.cumsum(freq_sequence[:seq_len_sampling]))
                            )
            xnew = np.arange(cfg.pModel.RawData.Solver.sequence_length)
            ctrl_values[i, j, :] = CubicSpline(x, ctrl_values_sampled[0, j])(xnew)

            # normalize values to bounds
            min_val = np.min(ctrl_values[i, j, :])
            max_val = np.max(ctrl_values[i, j, :])
            # normalize data to min 0 and max 1
            _values = (ctrl_values[i, j, :] - min_val) / (max_val - min_val)
            # randomly samply base and delta
            base = np.random.uniform(bounds[j][0], bounds[j][1])
            delta = np.random.uniform(0, bounds[j][1]-bounds[j][0])
            # calculate new base if delta is too large
            if base + delta > bounds[j][1]:
                base = bounds[j][1] - delta
            elif base - delta < bounds[j][0]:
                base = bounds[j][0]
            # calculate new values
            ctrl_values[i, j, :] = _values * delta + base
            if ctrl_values[i, j, :].min() < bounds[j][0] or ctrl_values[i, j, :].max() > bounds[j][1]:
                print('error in random_sampling_controls_w_offset_cubic_splines')
    return ctrl_values

def random_steps_sampling_controls(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    
    i_step = cfg.pModel.RawData.Solver.sequence_length // 2
    for i in range(len(cfg.pModel.RawData.controls.keys())):
        #ctrl_values[:, i, :] = np.random.uniform(bounds[i][0], bounds[i][1], (cfg.pModel.RawData.n_samples, cfg.pModel.RawData.Solver.sequence_length))
        _signal_value_before_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        _signal_value_after_step = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
        ctrl_values[:, i, :i_step] = _signal_value_before_step[:, None]
        ctrl_values[:, i, i_step:] = _signal_value_after_step[:, None]
    
    # last control input is not used.
    return ctrl_values

def random_frequency_response_sampling_controls(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.controls[key][0], cfg.pModel.RawData.controls[key][1]] for key in cfg.pModel.RawData.controls.keys()]
    ctrl_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls.keys()), cfg.pModel.RawData.Solver.sequence_length))
    
    _max_frequency = cfg.pModel.RawData.controls_frequency_min_in_timesteps * 4 # because this is only half the frequency
    _min_frequency = cfg.pModel.RawData.controls_frequency_max_in_timesteps * 4 

    i_step = cfg.pModel.RawData.Solver.sequence_length // 2
    len_frequency = cfg.pModel.RawData.Solver.sequence_length - i_step

    freq_fun = lambda x: _min_frequency + (_max_frequency - _min_frequency) * (x/len_frequency)
    turns = np.zeros(len_frequency)
    for i in range(1,len_frequency):
        turns[i] = turns[i-1] + (1/freq_fun(i))
    phi = turns * (2 * np.pi) 
    sine = np.sin(phi)
    for i in range(cfg.pModel.RawData.n_samples):
        for j in range(len(cfg.pModel.RawData.controls.keys())):
            _signal_value_start = np.random.uniform(bounds[j][0], bounds[j][1],1)
            ctrl_values[i, j, :i_step] = _signal_value_start[:, None]
            _amplitude = np.random.uniform(0, bounds[j][1] - bounds[j][0])
            if bounds[j][1]  < _signal_value_start + _amplitude:
                _amplitude = bounds[j][1] - _signal_value_start
            if bounds[j][0] > _signal_value_start - _amplitude:
                _amplitude = _signal_value_start - bounds[j][0]
            assert bounds[j][0] + _amplitude <= _signal_value_start <= bounds[j][1] - _amplitude
            _signal_value_end = _signal_value_start + sine * _amplitude
            ctrl_values[i, j, i_step:] = _signal_value_end[:]
    return ctrl_values

def load_controls_from_file(cfg: data_gen_config):
    # load controls from file by control variable name
    _df = pd.read_csv(cfg.pModel.RawData.controls_file_path)
    _list = []
    for key in cfg.pModel.RawData.controls.keys():
        # append to list column that matches the key
        _list.append(_df[key].values)
    time_ctrls = _df['time'].values
    # resample to time vector TODO: better make time vector only once
    time = np.arange(cfg.pModel.RawData.Solver.simulationStartTime, cfg.pModel.RawData.Solver.simulationEndTime + cfg.pModel.RawData.Solver.timestep, cfg.pModel.RawData.Solver.timestep)
    ctrl_values = [np.interp(time, time_ctrls, ctrl) for ctrl in _list]
    ctrl_values = np.array(ctrl_values)
    ctrl_values = np.expand_dims(ctrl_values, axis=0)
    ctrl_values = np.repeat(ctrl_values, cfg.pModel.RawData.n_samples, axis=0)
    return ctrl_values

def random_sampling_initial_states(cfg: data_gen_config):
    bounds = [[cfg.pModel.RawData.states[key][0], cfg.pModel.RawData.states[key][1]] for key in cfg.pModel.RawData.states.keys()]
    initial_state_values = np.zeros((cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states.keys())))
    for i in range(len(cfg.pModel.RawData.states.keys())):
        initial_state_values[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], cfg.pModel.RawData.n_samples)
    return initial_state_values

def data_generation(cfg: data_gen_config,
                    initial_state_values: np.ndarray = None,
                    param_values: np.ndarray = None,
                    ctrl_values: np.ndarray = None):
    from data_generation.src.fmu_simulate import fmu_simulate # import here to avoid circular import
    
    # wrap fmu_simulate to include idx and catch exceptions
    def fmu_simulate_wrapped(idx, *args, **kwargs): #TODO: add routine to catch exceptions and retry with different sampling or other solver settings...
                                                    # We could move all the sampling strategies to the fmu_simulate_wrapped function and retry if the simulation fails, 
                                                    # but I believe it is better to leave it outside to implement e.g. latin hypercube and other strategies, that
                                                    # are not just random but take the previous results into account.
        t0 = time()
        try:
            res = fmu_simulate(*args, **kwargs)
            success = True
        except Exception as e:
            res = None
            success = False
        return {'idx': idx, 'result': res, 'time': time() - t0, 'success': success}
    
    # create dask client
    from dask.distributed import Client, as_completed, LocalCluster
    cluster = LocalCluster(n_workers = os.cpu_count()-2 if cfg.multiprocessing_processes is None else cfg.multiprocessing_processes, threads_per_worker = 1, processes = True)
    client = Client(cluster)
    # set logging level to warning
    logging.getLogger('distributed').setLevel(logging.WARNING)
    logging.info(client)
    futures = []
    t0 = time()
    logging.info('view diagnostic dashboard at: http://localhost:8787')
    logging.info('view per worker diagnostics at: http://127.0.0.1:8787/info/main/workers.html')
    logging.info('\t logs on this page show fmu simulation progress')
    for i in range(cfg.pModel.RawData.n_samples):
        futures.append(client.submit(fmu_simulate_wrapped, i,
                fmu_path = str(Path(cfg.pModel.RawData.fmuPath).resolve()),
                state_names = cfg.pModel.RawData.states.keys(),
                get_state_derivatives = cfg.pModel.RawData.states_der_include,
                initial_state_values = initial_state_values[i] if initial_state_values is not None else None,
                parameter_names = cfg.pModel.RawData.parameters.keys() if cfg.pModel.RawData.parameters is not None else None,
                parameter_values = param_values[i] if param_values is not None else None,
                control_names = cfg.pModel.RawData.controls.keys(),
                control_values = ctrl_values[i] if ctrl_values is not None else None,
                control_from_model_names = cfg.pModel.RawData.controls_from_model if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model else None,
                output_names = cfg.pModel.RawData.outputs,
                start_time = cfg.pModel.RawData.Solver.simulationStartTime, 
                stop_time = cfg.pModel.RawData.Solver.simulationEndTime, 
                fmu_simulate_step_size = cfg.pModel.RawData.Solver.timestep,
                fmu_simulate_tolerance = cfg.pModel.RawData.Solver.tolerance,
            )
        )
    _n_completed = 0
    _n_failed = 0
    idx_failed = []
    # open raw data file
    raw_data = h5py.File(filepath_raw_data(cfg), 'a')
    for future, res in as_completed(futures, with_results=True):
        if res['success'] is False:
            _n_failed += 1
            idx_failed.append(res['idx'])
        elif res['success'] is True:
            _n_completed += 1
            # unpack results
            idx = res['idx']
            outputs, states, states_der, controls_from_model = res['result']
            if cfg.pModel.RawData.outputs is not None:
                raw_data['outputs'][idx] = outputs
            if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model is True:
                raw_data['controls'][idx] = controls_from_model
            raw_data['states'][idx] = states
            if cfg.pModel.RawData.states_der_include:
                raw_data['states_der'][idx] = states_der
        logging.info('{}/{} completed, {}/{} failed. fmu {} took {}s'.format(_n_completed, cfg.pModel.RawData.n_samples, _n_failed, cfg.pModel.RawData.n_samples, res['idx'], res['time']))
    
    logging.info('multiprocessing time: {}'.format(time() - t0))
    # add failed idx to raw data file
    raw_data.create_dataset('failed_idx', data=np.array(idx_failed))
    logging.info('failed idx: {}'.format(idx_failed))
    logging.info('Added failed idx to raw data file')
    # close raw data file
    raw_data.close()
    logging.info('closed raw data file, all data saved. Proceeding errors have no influence on the data.')
    for future in futures:
        future.release()
    client.shutdown()
    cluster.close()

def sample_all_values(cfg):
    # sample initial states, parameters and controls with given sampling strategy
    if cfg.pModel.RawData.initial_states_include:
        if cfg.pModel.RawData.initial_states_sampling_strategy == 'R':
            initial_state_values = random_sampling_initial_states(cfg)
        logging.info('initial_state_values.shape: {}'.format(initial_state_values.shape))
    else:
        initial_state_values = None
        logging.info('No initial state sampling included in raw data generation')

    if cfg.pModel.RawData.parameters_include:
        if cfg.pModel.RawData.parameters_sampling_strategy == 'R':
            param_values = random_sampling_parameters(cfg)
    else:
        # save default parameter values
        if cfg.pModel.RawData.parameters is not None:
            _param_default = [cfg.pModel.RawData.parameters[key][2] for key in cfg.pModel.RawData.parameters.keys()]
            param_values = [_param_default for _ in range(cfg.pModel.RawData.n_samples)]
            param_values = np.array(param_values)
        else:
            param_values = None
        logging.info('No parameter sampling included in raw data generation')
    if param_values is not None:
        logging.info('param_values.shape: {}'.format(param_values.shape))
    
    if cfg.pModel.RawData.controls_include:
        if cfg.pModel.RawData.controls_sampling_strategy == 'R':
            ctrl_values = random_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RO':
            ctrl_values = random_sampling_controls_w_offset(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'ROCS':
            ctrl_values = random_sampling_controls_w_offset_cubic_splines_old_clip_manual(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RROCS':
            ctrl_values = random_sampling_controls_w_offset_cubic_splines_clip_random(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RS':
            ctrl_values = random_steps_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'RF':
            ctrl_values = random_frequency_response_sampling_controls(cfg)
        elif cfg.pModel.RawData.controls_sampling_strategy == 'file':
            ctrl_values = load_controls_from_file(cfg)
        logging.info('ctrl_values.shape: {}'.format(ctrl_values.shape))
    else:
        ctrl_values = None
        logging.info('No control sampling included in raw data generation')
    return initial_state_values, param_values, ctrl_values
    

@hydra.main(config_path=str(Path('conf').absolute()), config_name='data_gen', version_base=None)
def main(cfg: data_gen_config):
    cfg = convert_cfg_to_dataclass(cfg)
    
    # create hdf5 file for raw data
    log_overwriting_file(filepath_raw_data(cfg))
    raw_data = h5py.File(filepath_raw_data(cfg), 'w')

    # sample initial states, parameters and controls with given sampling strategy
    initial_state_values, param_values, ctrl_values = sample_all_values(cfg)

    if initial_state_values is not None:
        raw_data.create_dataset('initial_states', data=initial_state_values)

    if param_values is not None:
        raw_data.create_dataset('parameters', data=param_values)
        raw_data.create_dataset('parameters_names', data=np.array(list(cfg.pModel.RawData.parameters.keys()), dtype='S'))

    if ctrl_values is not None and cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model is False:
        raw_data.create_dataset('controls', data=ctrl_values)
        raw_data.create_dataset('controls_names', data=np.array(list(cfg.pModel.RawData.controls.keys()), dtype='S'))

    # generate time vector
    time = np.arange(cfg.pModel.RawData.Solver.simulationStartTime, cfg.pModel.RawData.Solver.simulationEndTime + cfg.pModel.RawData.Solver.timestep, cfg.pModel.RawData.Solver.timestep)

    # allocate memory in hdf5 file for raw data
    raw_data.create_dataset('time', data=time)
    if cfg.pModel.RawData.outputs is not None:
        raw_data.create_dataset('outputs', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.outputs), len(time)))
        raw_data.create_dataset('outputs_names', data=np.array(list(cfg.pModel.RawData.outputs), dtype='S'))
    if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model is True:
        raw_data.create_dataset('controls', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.controls_from_model), len(time)))
        raw_data.create_dataset('controls_names', data=np.array(list(cfg.pModel.RawData.controls_from_model), dtype='S'))
    raw_data.create_dataset('states', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states), len(time)))
    raw_data.create_dataset('states_names', data=np.array(list(cfg.pModel.RawData.states.keys()), dtype='S'))
    if cfg.pModel.RawData.states_der_include:
        raw_data.create_dataset('states_der', (cfg.pModel.RawData.n_samples, len(cfg.pModel.RawData.states), len(time)))
        raw_data.create_dataset('states_der_names', data=np.array(list('der({})'.format(key) for key in cfg.pModel.RawData.states.keys()), dtype='S'))

    # add creation date (YYYY-MM-DD HH:MM:SS)
    creation_date = datetime.now()
    raw_data.attrs['creation_date'] = str(creation_date)
    cfg.pModel.RawData.creation_date = str(creation_date)
    logging.info('added creation date: {} to hdf5-file and config.yaml'.format(creation_date))

    # add config fields to hdf5 file
    raw_data.attrs['config'] = OmegaConf.to_yaml(cfg.pModel.RawData)
    # close hdf5 file
    raw_data.close()

    # generate raw data and save it to hdf5 file
    data_generation(cfg, initial_state_values, param_values, ctrl_values)

    # save pModel config as yaml
    log_overwriting_file(filepath_raw_data_config(cfg))
    OmegaConf.save(cfg.pModel.RawData, filepath_raw_data_config(cfg))

if __name__ == '__main__':
    main()