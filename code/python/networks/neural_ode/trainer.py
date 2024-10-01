import torch
import hydra
from pathlib import Path
import numpy as np
import sys
import os
import mlflow
import logging
import shutil
import h5py
import filepaths
import time as pyTime
import copy
import traceback

from pydantic.dataclasses import dataclass as pydantic_dataclass
from h5py import Dataset as hdf5_dataset_class

from torch.nn.utils import clip_grad_norm_

from networks.neural_ode.neural_ode_architecture import NeuralODE
from networks.neural_ode.latent_ode_architecture import BalancedNeuralODE
from config import train_test_config_class, neural_ode_network_class, base_training_settings_class, convert_cfg_to_dataclass, latent_ode_network_class
from networks.src.load_data import load_dataset_and_config, make_stacked_dataset, TimeSeriesDataset
from networks.src.early_stopping import EarlyStopping
from networks.src.capacity_scheduler import capacity_scheduler as CapacityScheduler
from utils.hydra_mlflow_decorator import log_hydra_to_mlflow

torch.backends.cudnn.benchmark = True

def initialize_model(cfg: train_test_config_class, train_dataset: TimeSeriesDataset, hdf5_dataset: hdf5_dataset_class, dataset_config):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')
    # create model (insert specific creations here)
    if type(cfg.nn_model.network) is neural_ode_network_class:
        model = NeuralODE(states_dim=train_dataset[0]['states'].shape[0],
                        controls_dim=train_dataset[0]['controls'].shape[0] if dataset_config.RawData.controls_include is True else 0,
                        parameters_dim=train_dataset[0]['parameters'].shape[0] if dataset_config.RawData.parameters_include is True else 0,
                        outputs_dim=train_dataset[0]['outputs'].shape[0] if dataset_config.RawData.outputs is not None else 0,
                        controls_to_output_nn=cfg.nn_model.network.controls_to_output_nn,
                        hidden_dim=cfg.nn_model.network.linear_hidden_dim, 
                        n_layers=cfg.nn_model.network.n_linear_layers,
                        hidden_dim_output_nn=cfg.nn_model.network.hidden_dim_output_nn,
                        n_layers_output_nn=cfg.nn_model.network.n_layers_output_nn,
                        activation=eval(cfg.nn_model.network.activation),
                        intialization=cfg.nn_model.training.pre_training.initialization_type,
                        initialization_ode=cfg.nn_model.training.initialization_type_ode,)
        # initialize normalizations
        model.normalization_init(hdf5_dataset)
        # save model file to hydra output directory
        shutil.copy(Path(NeuralODE.__module__.replace('.', os.sep)+'.py'), filepaths.dir_current_hydra_output())
        logging.info('copied file to file: {}'.format(filepaths.dir_current_hydra_output()))
    elif type(cfg.nn_model.network) is latent_ode_network_class:
        model = BalancedNeuralODE(
                        states_dim=train_dataset[0]['states'].shape[0],
                        lat_states_mu_dim=cfg.nn_model.network.lat_states_dim,
                        parameters_dim=train_dataset[0]['parameters'].shape[0] if dataset_config.RawData.parameters_include is True else 0,
                        lat_parameters_dim=cfg.nn_model.network.lat_parameters_dim,
                        controls_dim=train_dataset[0]['controls'].shape[0] if dataset_config.RawData.controls_include is True else 0,
                        lat_controls_dim=cfg.nn_model.network.lat_controls_dim,
                        outputs_dim=train_dataset[0]['outputs'].shape[0] if dataset_config.RawData.outputs is not None else 0,
                        hidden_dim=cfg.nn_model.network.linear_hidden_dim,
                        n_layers=cfg.nn_model.network.n_linear_layers,
                        controls_to_decoder=cfg.nn_model.network.controls_to_decoder,
                        activation=eval(cfg.nn_model.network.activation),
                        initialization_type=cfg.nn_model.training.initialization_type,
                        initialization_type_ode=cfg.nn_model.training.initialization_type_ode,
                        initialization_type_ode_matrix=cfg.nn_model.training.initialization_type_ode_matrix,
                        lat_ode_type=cfg.nn_model.network.lat_ode_type,
                        params_to_state_and_control_encoder=cfg.nn_model.network.params_to_state_and_control_encoder,
                        state_encoder_linear = cfg.nn_model.network.state_encoder_linear,
                        control_encoder_linear = cfg.nn_model.network.control_encoder_linear,
                        parameter_encoder_linear = cfg.nn_model.network.parameter_encoder_linear,
                        ode_linear = cfg.nn_model.network.ode_linear,
                        decoder_linear = cfg.nn_model.network.decoder_linear,
                        lat_state_mu_independent = cfg.nn_model.network.lat_state_mu_independent,
                        )
        # initialize normalizations
        model.normalization_init(hdf5_dataset)
    else:
        raise ValueError('{} is an unimplemented neural network class. Please implement in train_all_phases or adjust defaults in .yaml-file'.format(type(cfg.nn_model.network)))
    logging.info('Initialized model: {}'.format(model))
    logging.info('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model.to(device)
    logging.info('moved model to {}'.format(device))
    return model

@log_hydra_to_mlflow
def train_all_phases(cfg: train_test_config_class):
    logging.info('Start training all phases....')
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')
    logging.info('Using device: {}'.format(device))
    
    # load hdf5 dataset
    hdf5_dataset, dataset_config = load_dataset_and_config(cfg)
    mlflow.log_param('dataset_name', cfg.dataset_name)
    
    # collect jobs
    # job_list=[] filled with dict of style: {'skip': bool, 'test': bool, 'train_cfg': cfg, 'pre_train': bool}
    job_list = []
    # pre-training
    job_list.append({'skip': not cfg.nn_model.training.pre_train or cfg.nn_model.training.load_pretrained_model or cfg.nn_model.training.load_trained_model_for_test,
                     'test': False, 'train_cfg': cfg.nn_model.training.pre_training, 'pre_train': True})
    # main training
    for idx, main_train_cfg in enumerate(cfg.nn_model.training.main_training):
        job_list.append({'skip': cfg.nn_model.training.load_trained_model_for_test, 'test': False, 'train_cfg': main_train_cfg, 'pre_train': False})
    # test
    job_list.append({'skip': False, 'test': True, 'train_cfg': cfg.nn_model.training.main_training[-1], 'pre_train': False})
    logging.info('Created job list: {}'.format(job_list))

    # flags
    _created_datasets_and_loaders=False
    _loaded_seq_len=-1
    _loaded_batch_size=-1
    _created_model=False
    _epoch_0 = 0
    _reload_dataloaders_required = False
    for idx, job in enumerate(job_list):
        while True: # loop to catch memory errors
            try:
                if job['skip'] is False: # create dataloaders for this job
                    if job['pre_train'] is True:
                        logging.info('Starting Pre-Training with settings {}'.format(job['train_cfg']))
                    elif job['test'] is True:
                        logging.info('Starting Testing with settings {}'.format(job['train_cfg']))
                    else:
                        logging.info('Starting Train Job {} with settings {}'.format(idx, job['train_cfg']))
                    # loading datasets and initializing dataloaders
                    # set seq_len
                    if job['pre_train'] is True:
                        _load_seq_len = job['train_cfg'].load_seq_len
                        _seq_len_batches = 1
                    elif job['test'] is True:
                        _load_seq_len = None
                        _seq_len_batches = None
                    else:
                        _load_seq_len = job['train_cfg'].load_seq_len
                        _seq_len_batches = job['train_cfg'].seq_len_train
                    if _created_datasets_and_loaders is False or _load_seq_len != _loaded_seq_len: 
                        if _created_datasets_and_loaders is True:
                            _keys = list(datasets.keys())
                            for key in _keys:
                                del datasets[key]
                        # make torch tensor datasets
                        datasets = {}
                        for context in ['train', 'test', 'validation', 'common_test']:
                            datasets[context] = make_stacked_dataset(dataset_config, hdf5_dataset, context, _load_seq_len, _seq_len_batches)
                        _loaded_seq_len = _load_seq_len
                        _reload_dataloaders_required = True
                    else:
                        for context in ['train', 'test', 'validation', 'common_test']:
                            datasets[context].set_seq_len(_seq_len_batches)
                        _reload_dataloaders_required = True # TODO; check if this is necessary
                    _batch_size = job['train_cfg'].batch_size if job['test'] is False else cfg.nn_model.training.batch_size_test
                    _drop_last = True if job['test'] is False else False
                    _shuffle = True if job['test'] is False else False
                    if _created_datasets_and_loaders is False or _loaded_batch_size != _batch_size or _reload_dataloaders_required is True or job['test'] is True:
                        # initialiaze batch_loader, as batch size can't be set to a new value
                        if _created_datasets_and_loaders is True:
                            #del dataloaders
                            _keys = list(dataloaders.keys())
                            for key in _keys:
                                del dataloaders[key]
                        # create new
                        dataloaders={}
                        for context in ['train', 'test', 'validation', 'common_test']:
                            if job['test'] is True and len(datasets[context]) == 0: # when only testing, datasets can be empty
                                dataloaders[context] = None
                                logging.info('Only Testing: No data for context {} in dataset. Skipping loading dataloader for this context'.format(context))
                            else:
                                _num_workers = cfg.n_workers_train_loader if context == 'train' else cfg.n_workers_other_loaders
                                if context == 'train' and job['pre_train'] is True:
                                    _num_workers = 1 * _num_workers
                                if _batch_size > len(datasets[context]):
                                    _batch_size_here = int(len(datasets[context])/2)+3
                                    logging.warning('Batch size {} is larger than dataset size {} for context {}. Setting batch size to {}'.format(_batch_size, len(datasets[context]), context, _batch_size_here))
                                else:
                                    _batch_size_here = _batch_size
                                if len(datasets[context]) == 0:
                                    raise ValueError('While creating dataloaders, dataset for context {} is empty. Aborting.'.format(context))
                                dataloaders[context] = torch.utils.data.DataLoader(datasets[context], batch_size=_batch_size_here, shuffle=_shuffle,
                                                                                    num_workers = _num_workers, persistent_workers=True, 
                                                                                    pin_memory=True, drop_last=_drop_last, prefetch_factor=cfg.prefetch_factor)
                        _created_datasets_and_loaders = True
                        _loaded_batch_size = _batch_size
                    
                    _created_model_this_job = False	
                    # initialize model
                    if _created_model is False:
                        model = initialize_model(cfg, datasets['train'], hdf5_dataset, dataset_config)
                        _created_model, _created_model_this_job = True, True
                    if cfg.nn_model.training.load_pretrained_model is True and _created_model_this_job is True:
                        _path = filepaths.filepath_from_ml_artifacts_uri(cfg.nn_model.training.path_pretrained_model)
                        model.load(path=_path, device=device)
                        logging.info('Loaded pretrained model from {}'.format(_path))
                        if cfg.nn_model.training.pre_trained_model_seq_len is not None: 
                            job_list[idx]['train_cfg'].seq_len_epoch_start = cfg.nn_model.training.pre_trained_model_seq_len
                            logging.info('Set seq_len_epoch_start for next job to {}'.format(cfg.nn_model.training.pre_trained_model_seq_len))
                        else:
                            job_list[idx]['train_cfg'].seq_len_epoch_start = job['train_cfg'].seq_len_train
                            logging.info('Set seq_len_epoch_start for this job to seq_len_train {} as no pre_trained_model_seq_len is given in config'.format(job['train_cfg'].seq_len_train))
                    if cfg.nn_model.training.load_trained_model_for_test is True:
                        _path = cfg.nn_model.training.path_trained_model
                        _path = filepaths.filepath_from_local_or_ml_artifacts(_path)
                        model.load(path=_path, device=device)
                        logging.info('Loaded trained model from {}'.format(_path))

                if job['skip'] is True:
                    if job['pre_train'] is True:
                        logging.info('Skipping Pre-Training')
                    else:
                        logging.info('Skipping Train Job {} as trained model is loaded in following phases'.format(idx))
                else:
                    if job['test'] is False:
                        # train one phase
                        _epoch_0 = train_one_phase(cfg, model, dataloaders, job['train_cfg'], job['test'], job['pre_train'], idx, _epoch_0)
                        # set seq_len_epoch_start for next job
                        if len(job_list) > idx+1:
                            # consequently, seq_len_epoch_start should be seq_len_train
                            job_list[idx+1]['train_cfg'].seq_len_epoch_start = job['train_cfg'].seq_len_train if job['pre_train'] is False else 1
                            logging.info('Set seq_len_epoch_start for next job to {}, the seq_len_train of this job'.format(job_list[idx+1]['train_cfg'].seq_len_epoch_start))
                    else:
                        logging.info('Testing model and adding outputs to dataset')
                        hdf5_dataset.close()
                        # copy dataset to hydra output directory
                        _path = filepaths.filepath_dataset_current_hydra_output()
                        shutil.copy(filepaths.filepath_dataset_from_name(cfg.dataset_name), _path)
                        logging.info('copied dataset to file: {}'.format(_path))
                        hdf5_dataset = h5py.File(_path, 'r+')
                        for context in ['train', 'test', 'validation', 'common_test']:
                            if dataloaders[context] is None:
                                logging.info('No data for context {} in dataset. Skipping.'.format(context))
                            else:
                                logging.info('Adding outputs to dataset for context {}'.format(context))
                                ret_vals, model_outputs = test_or_validate_one_epoch(model, dataloaders[context], job['train_cfg'], job['pre_train'], device, all_batches=True, return_model_outputs=True)
                                # log stats with logging
                                logging.info('Stats for context {}: {}'.format(context, ret_vals))
                                # log stats with mlflow
                                mlflow.log_metrics(append_context_to_dict_keys(ret_vals, context), step=0)
                                # save loss function values
                                for key, value in ret_vals.items():
                                    hdf5_dataset.create_dataset(context+'/'+key, data=value) 
                                # save reconstructed timeseries and raw loss function values
                                for key, value in model_outputs.items():
                                    if cfg.nn_model.training.test_save_internal_variables is True:
                                        _save = True
                                    else:
                                        if key in ['states_hat', 'states_der_hat', 'outputs_hat']:
                                            _save = True
                                        elif cfg.nn_model.training.test_save_internal_variables_for == context:
                                            _save = True
                                            logging.info('Saving internal variable {} as test_save_internal_variables_for context is {}'.format(key, context))
                                        else:
                                            _save = False
                                            logging.info('Not saving internal variable {} as test_save_no_internal_variables is True'.format(key))
                                    if _save is True:
                                        hdf5_dataset.create_dataset(context+'/'+key, data=value)
                        hdf5_dataset.close()
                        # save this file
                        shutil.copy(Path(__file__), filepaths.dir_current_hydra_output())
                        logging.info('copied current trainer.py: {} \nto: \n{}'.format(Path(__file__), filepaths.dir_current_hydra_output()))
                if cfg.use_cuda:
                    torch.cuda.empty_cache() 
                break # break the exception loop
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e) or 'CUDA memory is almost full' in str(e):
                    logging.warning('CUDA out of memory error. Trying again in 10 seconds')
                    pyTime.sleep(10)
                    logging.info('Setting batch size to {}'.format(int(_batch_size * 0.7)))
                    if not job['test']:
                        job['train_cfg'].batch_size = int(_batch_size * 0.7)
                    else:
                        cfg.nn_model.training.batch_size_test = int(_batch_size * 0.7)
                    if cfg.use_cuda:
                        torch.cuda.empty_cache()
                else:
                    raise e
    return ret_vals['loss']

            
# define train loop for one epoch
def train_one_epoch(model, optimizer, train_loader, scaler, train_cfg, pre_train, device, epoch, use_amp, use_cuda, batch_print_interval, epoch_this_phase):
    model.train()
    _time_forward = 0
    _time_backward = 0
    _time_step = 0
    _time_loader = 0
    _time_l = pyTime.time()
    batches_per_epoch = len(train_loader) if train_cfg.batches_per_epoch is None else train_cfg.batches_per_epoch
    if epoch_this_phase in [0, 1] and pre_train is False: # evaluate at control times only in first epoch to get good estimate for memory usage
        logging.info('Evaluating at control times to get good estimate for memory usage')
        train_cfg = copy.deepcopy(train_cfg)
        train_cfg.evaluate_at_control_times = True
    _batches_this_phase = epoch_this_phase * batches_per_epoch
    for batch_idx in range(batches_per_epoch):
        data_batch = next(iter(train_loader))
        # seq_len_increase_in_batches
        _batches_this_phase = epoch_this_phase * batches_per_epoch + batch_idx
        if pre_train is False:
            if _batches_this_phase < train_cfg.seq_len_increase_in_batches:
                _seq_len_now = train_cfg.seq_len_epoch_start + int(_batches_this_phase/train_cfg.seq_len_increase_in_batches * (train_cfg.seq_len_train - train_cfg.seq_len_epoch_start))
                _seq_len_now = min(_seq_len_now, train_cfg.seq_len_train)
                for keys in data_batch.keys():
                    if len(data_batch[keys].shape) == 3:
                        data_batch[keys] = data_batch[keys][:,:,:_seq_len_now]
                if batch_idx % batch_print_interval == 0:
                    logging.info('\t \t Increasing sequence length to {} in batch since phase start {}/{} of increase_in_batches'.format(_seq_len_now, _batches_this_phase, train_cfg.seq_len_increase_in_batches))
            else:
                _seq_len_now = train_cfg.seq_len_train
        else:
            _seq_len_now = 1
        _time_loader += pyTime.time() - _time_l
        _time = pyTime.time()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp and use_cuda):
            #model_and_loss_evaluation(self, data_batch, train_cfg, pre_train, device, return_model_outputs, test = False):
            ret_vals_train = model.model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, return_model_outputs = False, test = False, last_batch = batch_idx == batches_per_epoch-1)
        loss = ret_vals_train['loss']
        _time_forward += pyTime.time() - _time
        _time = pyTime.time()
        scaler.scale(loss).backward()
        #loss.backward()
        _flag_break_cuda_memory = False
        if use_cuda:
            mlflow.log_metric('CUDA_memory_reserved_GB', torch.cuda.memory_reserved()/(1024^3), step=epoch)
            if torch.cuda.memory_reserved() > 0.6 * torch.cuda.get_device_properties(0).total_memory and epoch_this_phase == 0:
                _flag_break_cuda_memory = True
        if pre_train is False and use_cuda:
            if (train_cfg.seq_len_train/_seq_len_now) * torch.cuda.memory_reserved() > 0.6 * torch.cuda.get_device_properties(0).total_memory and epoch_this_phase == 0:
                _flag_break_cuda_memory = True
            if torch.cuda.memory_reserved() > 0.98 * torch.cuda.get_device_properties(0).total_memory:
                _flag_break_cuda_memory = True
        if _flag_break_cuda_memory is True:
            logging.warning('CUDA memory is almost full. Raising exception to catch in train_all_phases')
            logging.info('Current number of batches for whole dataset: {}'.format(len(train_loader)))
            raise RuntimeError('CUDA memory is almost full')
        _ode_calls_backward = model.ode_fun_count if hasattr(model, 'ode_fun_count') else 0
        _time_backward += pyTime.time() - _time
        _time = pyTime.time()
        scaler.unscale_(optimizer)
        # clip gradients
        _norm = clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        if _norm > train_cfg.clip_grad_norm:
            logging.info('Gradient norm {} is larger than clip_grad_norm {}. Clipping Gradient.'.format(_norm, train_cfg.clip_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        _time_step += pyTime.time() - _time
        if batch_idx % batch_print_interval == 0:
            _total_time = _time_forward + _time_backward + _time_step + _time_loader
            _total_time = _total_time
            _ode_calls_forward = ret_vals_train['ode_calls_forward'] if 'ode_calls_forward' in ret_vals_train.keys() else 0 
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%) tot.: {}] Loss: {:.6f}, avg. time per batch: {:.3f} [load. {:.1f}%, forw. {:.1f}%, backw. {:.1f}%, step {:.1f}%], ODE calls forw/backw {}/{}'.format(
                epoch+1, batch_idx+1, batches_per_epoch,
                100. * batch_idx / batches_per_epoch, len(train_loader),
                loss.item(), _total_time/(batch_idx+1),_time_loader/_total_time*100, _time_forward/_total_time*100, _time_backward/_total_time*100, _time_step/_total_time*100,
                _ode_calls_forward, _ode_calls_backward))
        _time_l = pyTime.time()
    ret_vals_train = dict({key: value.item() if type(value)==torch.Tensor else value for key, value in ret_vals_train.items()})
    ret_vals_train['grad_norm'] = _norm
    ret_vals_train['seq_len_now'] = _seq_len_now
    ret_vals_train['time_forward'] = _time_forward
    ret_vals_train['time_backward'] = _time_backward
    ret_vals_train['time_optimizer_step'] = _time_step
    ret_vals_train['time_loader'] = _time_loader
    ret_vals_train['time_total'] = _time_forward + _time_backward + _time_step + _time_loader
    ret_vals_train['time_per_batch'] = ret_vals_train['time_total'] / batches_per_epoch
    ret_vals_train['time_per_batch_forward'] = ret_vals_train['time_forward'] / batches_per_epoch
    ret_vals_train['time_per_batch_backward'] = ret_vals_train['time_backward'] / batches_per_epoch
    ret_vals_train['time_per_batch_optimizer_step'] = ret_vals_train['time_optimizer_step'] / batches_per_epoch
    ret_vals_train['time_per_batch_loader'] = ret_vals_train['time_loader'] / batches_per_epoch
    if pre_train is False:
        ret_vals_train['ode_calls_backward'] = _ode_calls_backward
    return ret_vals_train  

def test_or_validate_one_epoch(model, data_loader, train_cfg, pre_train, device, all_batches=False, return_model_outputs=False, activate_deterministic_mode=False):
    model.eval()
    if all_batches is True:
        ret_vals = []
        for batch_idx, data_batch in enumerate(data_loader):
            logging.info('Testing batch {}/{}'.format(batch_idx+1, len(data_loader)))
            with torch.no_grad():
                ret_vals.append(model.model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, return_model_outputs=return_model_outputs, test=True))
        if return_model_outputs is True:
            model_outputs = {key: np.concatenate([x[1][key] for x in ret_vals], axis=0) for key in ret_vals[0][1].keys()}
            ret_vals = {key: np.mean([x[0][key] for x in ret_vals]) for key in ret_vals[0][0].keys()}
        else:
            ret_vals = {key: np.mean([x[key] for x in ret_vals]) for key in ret_vals[0].keys()}
    else:
        data_batch = next(iter(data_loader))
        with torch.no_grad():
            ret_vals = model.model_and_loss_evaluation(data_batch, train_cfg, pre_train, device, return_model_outputs=return_model_outputs, test=True, activate_deterministic_mode=activate_deterministic_mode)
        if return_model_outputs is True:
            model_outputs = ret_vals[1]
            ret_vals = ret_vals[0]
    return ret_vals if return_model_outputs is False else (ret_vals, model_outputs)

def append_context_to_dict_keys(dictionary: dict, context: str, pre_train: bool = False):
        if pre_train is True:
            return dict({'pre_{}_{}'.format(key, context): value for key, value in dictionary.items()})
        else:
            return dict({'{}_{}'.format(key, context): value for key, value in dictionary.items()})

def train_one_phase(cfg: train_test_config_class, model: torch.nn.Module, dataloaders: dict, train_cfg: base_training_settings_class, test: bool, pre_train: bool, job_idx: int, epoch_0: int = 0):
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.use_cuda else 'cpu')
    logging.info('Start next training phase....')
    
    if test is False:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr_start, weight_decay=train_cfg.weight_decay, betas=(train_cfg.beta1_adam, train_cfg.beta2_adam))
        if pre_train is False:
            if train_cfg.reload_optimizer is True:
                try:
                    optimizer.load_state_dict(torch.load(filepaths.filepath_optimizer_current_hydra_output(job_idx-1)))
                    logging.info('Reloaded optimizer from {}'.format(filepaths.filepath_optimizer_current_hydra_output(job_idx-1)))
                    # set learning rate to start value
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = train_cfg.lr_start
                        logging.info('Set learning rate to {} after reloading optimizer'.format(train_cfg.lr_start))
                except:
                    logging.warning('Could not reload optimizer from {}'.format(filepaths.filepath_optimizer_current_hydra_output(job_idx-1)))
                    logging.warning('Initializing optimizer with new parameters')
        _path_best_model = filepaths.filepath_pretrained_model_current_hydra_output() if pre_train is True else filepaths.filepath_model_current_hydra_output(job_idx)
        _path_optimizer_best_model = filepaths.filepath_optimizer_current_hydra_output() if pre_train is True else filepaths.filepath_optimizer_current_hydra_output(job_idx)
        _path_current_model = filepaths.filepath_model_current_hydra_output() # contiuously updated
        _path_current_optimizer = filepaths.filepath_optimizer_current_hydra_output() # contiuously updated
        early_stopping = EarlyStopping(patience=train_cfg.early_stopping_patience, verbose=True, threshold=train_cfg.early_stopping_threshold,
                                       threshold_mode=train_cfg.early_stopping_threshold_mode, path = _path_best_model, optimizer_path=_path_optimizer_best_model,
                                         trace_func=logging.info)
        nan_counter = 0
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_cuda and cfg.use_amp)
        logging.info('Training with automatic mixed precision: {}'.format(cfg.use_amp and cfg.use_cuda))
        if pre_train is True:
            _batches_per_epoch = len(dataloaders['train'])
            epochs_for_seq_len_increase = 0
        else:
            _batches_per_epoch = len(dataloaders['train']) if train_cfg.batches_per_epoch is None else train_cfg.batches_per_epoch
            if train_cfg.seq_len_epoch_start is not None:
                if train_cfg.seq_len_epoch_start < train_cfg.seq_len_train:
                    epochs_for_seq_len_increase = int(train_cfg.seq_len_increase_in_batches / _batches_per_epoch)
                else:
                    epochs_for_seq_len_increase = 0
                    train_cfg.seq_len_increase_in_batches = 0
            else:
                epochs_for_seq_len_increase = 0
                train_cfg.seq_len_increase_in_batches = 0
        max_epochs = train_cfg.max_epochs + epochs_for_seq_len_increase
        epoch_stop = epoch_0 + max_epochs # can be changed after seq_len increase has stopped.
        # flag to not early stop if seq_len_increase is active
        if pre_train is False:
            if epochs_for_seq_len_increase == 0:
                _flag_out_of_seq_len_increase = True
            else:
                _flag_out_of_seq_len_increase = False
        else:
            _flag_out_of_seq_len_increase = True
        _stable_epochs = 0 # count epochs with loss_validation < 2 * loss_train. Used for ending sequence length increase
        '''Training'''
        try:
            _flag_break_after_epoch = False
            _flag_first_epoch_this_phase = True
            for epoch in range(epoch_0, epoch_0 + max_epochs): # the upper range is a maximum value, and can be changed during training and escaped with if...break
                if epoch == epoch_stop:
                    break
                _flag_max_epoch = epoch == epoch_stop - 1
                _flag_early_stopping = early_stopping.early_stop and _flag_out_of_seq_len_increase is True
                _flag_break_after_loss_of = early_stopping.best_score < train_cfg.break_after_loss_of if train_cfg.break_after_loss_of is not None and early_stopping.best_score is not None else False
                _flag_nan_counter = nan_counter > 3
                if _flag_max_epoch or _flag_early_stopping or _flag_break_after_loss_of or _flag_nan_counter:
                    if _flag_max_epoch:
                        logging.info('Reached max epochs')
                        mlflow.log_param('job {} ended by'.format(job_idx), 'max epochs')
                    elif _flag_early_stopping:
                        logging.info("Early stopping")
                        mlflow.log_param('job {} ended by'.format(job_idx), 'early stopping')
                    elif _flag_break_after_loss_of:
                        logging.info('Break phase after reaching loss level of {}'.format(train_cfg.break_after_loss_of))
                        mlflow.log_param('job {} ended by'.format(job_idx), 'break after loss')
                    elif _flag_nan_counter:
                        logging.info('Break phase after 4 NaNs in loss')
                        mlflow.log_param('job {} ended by'.format(job_idx), '10 NaNs in loss')
                    else:
                        raise ValueError('This should not happen')
                    _flag_break_after_epoch = True
                    # load the last checkpoint with the best model
                    model.load(path=_path_best_model, device=device)
                    logging.info('loaded best model from {}'.format(_path_best_model))
                # end sequence length increase if stable epochs is reached
                if pre_train is False:
                    if _stable_epochs > train_cfg.seq_len_increase_abort_after_n_stable_epochs and _flag_out_of_seq_len_increase is False:
                        train_cfg.seq_len_increase_in_batches = _batches_per_epoch * (epoch - epoch_0)
                        epoch_stop = epoch_0 + train_cfg.max_epochs + (epoch - epoch_0) # new epoch stop
                if not _flag_break_after_epoch and not _flag_first_epoch_this_phase:
                    try:
                        ret_vals_train = train_one_epoch(model, optimizer, dataloaders['train'], scaler, train_cfg, pre_train, device, epoch, cfg.use_amp, cfg.use_cuda, cfg.batch_print_interval, epoch-epoch_0)
                    except AssertionError as e:
                        if 'underflow' in str(e):
                            logging.warning('Underflow in automatic mixed precision. Trying again without autocast')
                            ret_vals_train = train_one_epoch(model, optimizer, dataloaders['train'], scaler, train_cfg, pre_train, device, epoch, False, cfg.use_cuda, cfg.batch_print_interval, epoch-epoch_0)
                    if np.isnan(ret_vals_train['loss']):
                        if not nan_counter > 0:
                            try:
                                model.load(path=_path_current_model, device=device)
                                optimizer.load_state_dict(torch.load(_path_current_optimizer))
                                logging.warning('Loss is NaN. Loaded last model and corresponding optimizer from {}'.format(_path_current_model))
                            except:
                                logging.error('Loss is NaN. Could not load last model and corresponding optimizer from {}'.format(_path_current_model))
                                logging.error('The reason for this is that not even the first epoch had stable resuls. Aborting.')
                                raise ValueError('Loss is NaN. First epoch did not have stable results.')
                        else:
                            # fallback to last best model
                            model.load(path=_path_best_model, device=device)
                            optimizer.load_state_dict(torch.load(_path_optimizer_best_model))
                            logging.warning('Loss is NaN. Loaded last best model and corresponding optimizer from {}'.format(_path_best_model))
                        mlflow.log_metric('loss_nan_reload', 1, step=epoch)
                        nan_counter += 1
                    else:
                        mlflow.log_metric('loss_nan_reload', 0, step=epoch)
                        nan_counter = 0
                        model.save(path=_path_current_model)
                        torch.save(optimizer.state_dict(), _path_current_optimizer)
                else: 
                    _activate_deterministic_mode = train_cfg.activate_deterministic_mode_after_this_phase and _flag_break_after_epoch
                    ret_vals_train = test_or_validate_one_epoch(model, dataloaders['train'], train_cfg, pre_train, device, all_batches=False, return_model_outputs=False, 
                                                                activate_deterministic_mode=_activate_deterministic_mode)
                    if _activate_deterministic_mode:
                        logging.info('Activated deterministic mode')
                        model.save(path=_path_best_model)
                        logging.info('Saved model with deterministic mode activated to {}'.format(_path_best_model))
                    _flag_first_epoch_this_phase = False
                    ret_vals_train['ode_calls_backward'] = 0 # to avoid error in logging
                mlflow.log_metrics(append_context_to_dict_keys(ret_vals_train, 'train', pre_train), step=epoch)
                ret_vals_validation = test_or_validate_one_epoch(model, dataloaders['validation'], train_cfg, pre_train, device, all_batches=False, return_model_outputs=False)
                early_stopping(ret_vals_validation['loss'], model, epoch, optimizer)
                # count stable epochs to end seq_len_increase early
                if ret_vals_validation['loss'] < 2 * ret_vals_train['loss']:
                    _stable_epochs += 1
                    if _flag_out_of_seq_len_increase is False and pre_train is False:
                        logging.info('\t \t \t Stable seq_len_increase epochs: {}/{}'.format(_stable_epochs, train_cfg.seq_len_increase_abort_after_n_stable_epochs))
                else:
                    _stable_epochs = 0
                mlflow.log_metrics(append_context_to_dict_keys(ret_vals_validation, 'validation', pre_train), step=epoch)
                ret_vals_test = test_or_validate_one_epoch(model, dataloaders['test'], train_cfg, pre_train, device, all_batches=False, return_model_outputs=False)
                mlflow.log_metrics(append_context_to_dict_keys(ret_vals_test, 'test', pre_train), step=epoch)
                mlflow.log_metric('lr', optimizer.param_groups[0]['lr'], step=epoch)
                mlflow.log_metric('Stable_epochs', _stable_epochs, step=epoch)
                _progress_string = model.get_progress_string(ret_vals_train, ret_vals_validation, ret_vals_test, pre_train)
                logging.info('Epoch: {}/{} EarlyStopping: {}/{} |-| {}'.format(epoch+1, epoch_stop, early_stopping.counter, early_stopping.patience, _progress_string))
                if _flag_break_after_epoch is True:
                    break
                _batches_this_phase = (epoch - epoch_0 + 1)* _batches_per_epoch
                # set early stopping active if seq_len_increase_in_batches is reached through flag and reset early stopping counter
                if pre_train is False:
                    if _batches_this_phase > train_cfg.seq_len_increase_in_batches and _flag_out_of_seq_len_increase is False:
                        logging.info('Out of seq_len_increase_in_batches')
                        _flag_out_of_seq_len_increase = True
                        early_stopping.reset_counter()
                mlflow.log_metric('EarlyStopping_counter', early_stopping.counter, step=epoch)
        except KeyboardInterrupt:
            logging.info('Interrupted by user')
            mlflow.log_param('ended by', 'keyboard interrupt')
            # load the last checkpoint with the best model
            try:
                model.load(path=_path_best_model, device=device)
            except:
                logging.warning('Could not load best model from {}'.format(_path_best_model))
                for i in range(job_idx, 0):
                    _path_best_model = filepaths.filepath_model_current_hydra_output(i)
                    try:
                        model.load(path=_path_best_model, device=device)
                        logging.info('loaded best model from {}'.format(_path_best_model))
                        break
                    except:
                        logging.warning('Could not load best model from {}'.format(_path_best_model))
            logging.info('loaded best model from {}'.format(_path_best_model))
        mlflow.log_metric('job_{}_final_epoch'.format(job_idx), value=epoch)
    return epoch + 1

        
        

@hydra.main(config_path=str(Path('conf').absolute()), config_name='train_test_neural_ode', version_base=None)
def main(cfg: train_test_config_class):
    # try:
    val = train_all_phases(cfg)
    # except Exception as e:
    #     logging.error('Exception occured: {}'.format(e))
    #     logging.error(traceback.format_exc())
    #     if True:
    #         raise e
    return val

if __name__ == '__main__':
    main()