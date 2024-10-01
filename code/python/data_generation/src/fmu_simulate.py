import uuid
#import sdf
import datetime
import logging
import time

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import fmpy
import numpy as np
import shutil
import sys
import os

from data_generation.src.read_dymola_data import read_dymola_data

class logger_max_rate:
    def __init__(self, logger: logging.Logger, max_rate: float = 30):
        self.last_log_time = time.time() - 1.1*max_rate
        self.logger = logger
        self.max_freq = max_rate
    
    def info(self, msg):
        if time.time() - self.last_log_time > self.max_freq:
            self.logger.info(msg)
            self.last_log_time = time.time()


def fmu_simulate(fmu_path,
                state_names = None,
                get_state_derivatives = False,
                initial_state_values = None,
                parameter_names = None,
                parameter_values = None,
                control_names = None,
                control_values = None,
                control_from_model_names = None,
                output_names = None,
                start_time=0.0, 
                stop_time=1800.0, 
                fmu_simulate_step_size=1,
                fmu_simulate_tolerance=1e-4,
                load_result_from_file = False, 
                filepath_multiprocessing = os.path.join(os.getcwd(), '_wrk'),
                copy_fmu = False):
    """
    Simulate an FMU with the given parameters and return the results.
    It is possible to read the results from a file instead of communicating with the FMU.
    This can be useful for multiprocessing, because communication overhead might be reduced.
    However, saving the mat-file and opening was slower than communicating with the FMU on Windows. 
    Maybe the mat-file generating code is not able to be parallelized on Windows.
    On Linux, it works. However, I did not test if writing and reading the mat-file is faster than 
    communicating with the FMU on Linux.
    After simulation, the results are scaled by multiplying and substracting as in the config file.
    The copy_fmu option was turned off, because it wasnt seen that unpacking the FMU from the same directory
    multiple times caused problems. 

    Parameters
    ----------
    modelName : str
        name of the model
    input_names : list
        list of parameter names
    output_names : list
        list of output names
    parameter_values : list
        list of parameter values
    fmu_path : str
        path to the fmu
    start_time : float
        start time of the simulation
    stop_time : float
        stop time of the simulation
    fmu_simulate_step_size : float
        step size of the simulation
    fmu_simulate_tolerance : float 
        tolerance of the simulation
    verbose : bool
        if True, print additional information (default: False)
    load_result_from_file : bool
        if True, the simulation communicates with the fmu at every simulation timestep. 
        if False, the simulation reads the results from a mat-file (default: False)
        not properly implemented anymore, so not recommended do use
    filepath_multiprocessing : str
        path to the multiprocessing directory (default: os.path.join(working_directory, '_wrk'))
    """
    t_start = time.time()
    # determine if there is a logger 'distributed.worker'
    if not logging.getLogger('distributed.worker').hasHandlers():
        logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger('distributed.worker')
    logger_delayed = logger_max_rate(logger, 10)
    logger.info('fmu_simulate started')
    # create uniqueId for the fmu result file
    uniqueId = str(uuid.uuid4())
    
    # create own working directory and copy the fmu to that if load_result_from_file == False
    results_filename = os.path.join(os.getcwd(), uniqueId + "_internal.mat")
    if copy_fmu:
        os.makedirs(os.path.join(filepath_multiprocessing), exist_ok=True)
        os.makedirs(os.path.join(filepath_multiprocessing, uniqueId), exist_ok=True)
        new_fmu_path = os.path.join(filepath_multiprocessing, uniqueId, os.path.split(fmu_path)[1])
        shutil.copy(fmu_path, new_fmu_path)
    else:
        new_fmu_path = fmu_path

    '''simulation'''
    # read the model description
    model_description = read_model_description(new_fmu_path)
    model_variables = model_description.modelVariables
    # collect the value references and indices
    vrs = {}
    indices = {}
    variability = {}
    i = 0
    for variable in model_variables:
        vrs[variable.name] = variable.valueReference
        indices[variable.name] = i
        variability[variable.name] = variable.variability
        i += 1

    # get the value references for the states, parameters, inputs and outputs
    state_refs = [vrs[name] for name in state_names] if state_names is not None else None
    state_der_refs = [vrs['der({})'.format(name) ] for name in state_names] if state_names is not None and get_state_derivatives is True else None
    parameter_refs = [vrs[name] for name in parameter_names] if parameter_names is not None else None
    control_refs = [vrs[name] for name in control_names] if control_names is not None else None
    output_refs = [vrs[name] for name in output_names] if output_names is not None else None
    control_from_model_refs = [vrs[name] for name in control_from_model_names] if control_from_model_names is not None else None

    # extract the FMU
    unzipdir = extract(new_fmu_path)
    fmu = FMU2Slave(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.coSimulation.modelIdentifier,
                    instanceName=uniqueId)
    logger.info('fmu extracted')

    # initialize
    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time, tolerance=fmu_simulate_tolerance)
    # set the initial state
    if initial_state_values is not None:
        fmu.setReal(vr=state_refs, value=initial_state_values)
    # set the parameters
    if parameter_values is not None:
        # fmu.setReal(vr=parameter_refs, value=parameter_values)
        for i in range(len(parameter_values)):
            # integer or real method
            if np.dtype(parameter_values[i]) == np.int64 or np.dtype(parameter_values[i]) == np.int32:
                fmu.setInteger(vr=[parameter_refs[i]], value=[parameter_values[i]])
            else:
                fmu.setReal(vr=[parameter_refs[i]], value=[parameter_values[i]])

    # HERE CONTROL
    if control_values is not None:
        fmu.setReal(vr=control_refs, value=control_values[:,0])
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()
    logger.info('fmu initialized')

    records = []  # list to record the results
    t = start_time

    outputs = []
    states = []
    states_der = []
    controls_from_model = []


    def record_data():
        if output_names is not None:
            outputs.append(fmu.getReal(output_refs))
        if control_from_model_names is not None: #TODO: first order input interpolation better for training?
            controls_from_model.append(fmu.getReal(control_from_model_refs))
        if state_names is not None:
            states.append(fmu.getReal(state_refs))
            if get_state_derivatives is True:
                states_der.append(fmu.getReal(state_der_refs))
    
    # simulation loop
    logger.info('start simulation loop')
    steps = int(np.ceil((stop_time - start_time) / fmu_simulate_step_size))
    for i in range(steps):
        # set the controls
        if control_values is not None:
            fmu.setReal(vr=control_refs, value=control_values[:,i])
            if load_result_from_file == True:
                raise ValueError('cannot set control values if load_result_from_file == True')
        # record the data
        if load_result_from_file == False:
            record_data() 
        # advance the time
        if load_result_from_file == False:
            fmu.doStep(currentCommunicationPoint=t, communicationStepSize=fmu_simulate_step_size)
            t += fmu_simulate_step_size
        else:
            fmu.doStep(currentCommunicationPoint=t, communicationStepSize=stop_time)
            t += stop_time
        # break after one step if load_result_from_file == True
        if load_result_from_file == True:
            break
        #print('step ' + str(i) + ' of ' + str(int(np.ceil((stop_time - start_time) / fmu_simulate_step_size))) + ' done.')
        logger_delayed.info('step ' + str(i) + '/' + str(steps) + ' done.')
        # final length of outputs is i+1
    # record the final output
    if load_result_from_file == False:
        record_data()
    
    # terminate
    fmu.terminate()
    #fmu.freeInstance()
    fmu.fmi2FreeInstance(fmu.component)
    fmpy.freeLibrary(fmu.dll._handle)

    # gather results
    if load_result_from_file == False:
        outputs = np.array(outputs).transpose()
        if control_from_model_names is not None:
            controls_from_model = np.array(controls_from_model).transpose()
        if state_names is not None:
            states = np.array(states).transpose()
            if get_state_derivatives is True:
                states_der = np.array(states_der).transpose()
    else:
        logger.warning('load_result_from_file is not tested yet')
        if not os.path.exists(results_filename):
            print('MatFile does not exist for UUID ' + str(uniqueId))
            print(results_filename)
        outputs = read_dymola_data(results_filename, output_names, start_time, stop_time, fmu_simulate_step_size)
        if control_from_model_names is not None:
            controls_from_model = read_dymola_data(results_filename, control_from_model_names, start_time, stop_time, fmu_simulate_step_size)
        if state_names is not None:
            states = read_dymola_data(results_filename, state_names, start_time, stop_time, fmu_simulate_step_size)
            if get_state_derivatives is True:
                get_state_derivatives = read_dymola_data(results_filename, ['der({})'.format(name) for name in state_names], start_time, stop_time, fmu_simulate_step_size)
        # adjust to dimension convention
        outputs = outputs.transpose()
        if control_from_model_names is not None:
            controls_from_model = controls_from_model.transpose()  
        if state_names is not None:
            states = states.transpose()

    # delete results mat file and fmu directory
    if os.path.exists(results_filename):
        os.remove(results_filename)
    shutil.rmtree(unzipdir, ignore_errors=True)
    if copy_fmu:
        shutil.rmtree(os.path.split(new_fmu_path)[0], ignore_errors=True)
    logger.info('fmu_simulate took {:.4f} seconds  for {} steps'.format(time.time() - t_start, i))
    # print('fmu_simulate took {:.4f} seconds  for {} steps'.format(time.time() - t_start, i))
    return outputs, states, states_der, controls_from_model

from config import data_gen_config, cs, convert_cfg_to_dataclass
from pathlib import Path
import numpy as np
import hydra
import os
from pathlib import Path

@hydra.main(config_path=str(Path('conf').absolute()), config_name='data_gen_HeatPump', version_base=None)
def main(cfg: data_gen_config) -> None:
    """
    test the fmu_simulate function
    """
    cfg = convert_cfg_to_dataclass(cfg)

    # sample control values
    from data_generation.raw_data_generation import sample_all_values
    cfg.pModel.RawData.n_samples = 1
    initial_state_values, param_values, ctrl_values = sample_all_values(cfg)
    initial_state_values = initial_state_values[0,:] if initial_state_values is not None else None
    param_values = param_values[0,:] if param_values is not None else None
    ctrl_values = ctrl_values[0,:,:] if ctrl_values is not None else None
    # print all values
    if initial_state_values is not None:
        print('initial_state_values')
        for key, val in zip(cfg.pModel.RawData.states.keys(), initial_state_values):
            print('{}: {}'.format(key, val))
    else:
        print('initial_state_values: None')
    if cfg.pModel.RawData.parameters is not None:
        print('param_values')
        for key, val in zip(cfg.pModel.RawData.parameters.keys(), param_values if param_values is not None else [None]):
            print('{}: {}'.format(key, val))
    else:
        print('param_values: None')
    print('ctrl_values')
    if ctrl_values is not None:
        for key, val in zip(cfg.pModel.RawData.controls.keys(), ctrl_values if ctrl_values is not None else [None]):
            print('{}: {} ...'.format(key, val[0:3]))
    else:
        print('ctrl_values: None')

    t0 = time.time()
    outputs, states, states_der, controls_from_model = fmu_simulate(
        fmu_path = str(Path(cfg.pModel.RawData.fmuPath).resolve()),
        state_names = cfg.pModel.RawData.states.keys(),
        get_state_derivatives=cfg.pModel.RawData.states_der_include,
        initial_state_values = initial_state_values,
        parameter_names = cfg.pModel.RawData.parameters.keys() if cfg.pModel.RawData.parameters is not None else None,
        parameter_values = param_values,
        control_names = cfg.pModel.RawData.controls.keys(),
        control_values = ctrl_values if cfg.pModel.RawData.controls_include else None,
        control_from_model_names = cfg.pModel.RawData.controls_from_model if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model else None,
        output_names = cfg.pModel.RawData.outputs,
        start_time = cfg.pModel.RawData.Solver.simulationStartTime, 
        stop_time = cfg.pModel.RawData.Solver.simulationEndTime, 
        fmu_simulate_step_size = cfg.pModel.RawData.Solver.timestep,
        fmu_simulate_tolerance = cfg.pModel.RawData.Solver.tolerance,
    )
    logging.info('Simulation took ' + str(time.time() - t0) + ' seconds')

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(5,1, sharex=True)
    # turn grid on
    for i in range(4):
        ax[i].grid()
    x = np.arange(cfg.pModel.RawData.Solver.simulationStartTime, cfg.pModel.RawData.Solver.simulationEndTime + cfg.pModel.RawData.Solver.timestep, cfg.pModel.RawData.Solver.timestep)
    # x, states, states_der, ctrl_values = x[1:], states[:,1:], states_der[:,1:], ctrl_values[:,1:]
    if cfg.pModel.RawData.outputs is not None:
        for i in range(len(cfg.pModel.RawData.outputs)):
            ax[0].plot(x, outputs[i,:], label=cfg.pModel.RawData.outputs[i])
    ax[0].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
    ax[0].set_title('outputs')
    for i in range(len(cfg.pModel.RawData.states.keys())):
        ax[1].plot(x, states[i,:], label=list(cfg.pModel.RawData.states.keys())[i])
    ax[1].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
    ax[1].set_title('states')
    if cfg.pModel.RawData.states_der_include:
        for i in range(len(cfg.pModel.RawData.states.keys())):
            ax[2].plot(x, states_der[i,:], label=list(cfg.pModel.RawData.states.keys())[i])
        ax[2].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
        ax[2].set_title('state derivatives')
    if cfg.pModel.RawData.controls_include:
        for i in range(len(cfg.pModel.RawData.controls.keys())):
            ax[3].plot(x, ctrl_values[i,:], label=list(cfg.pModel.RawData.controls.keys())[i])
        ax[3].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
        ax[3].set_title('controls')
    if cfg.pModel.RawData.controls_only_for_sampling_extract_actual_from_model:
        for i in range(len(cfg.pModel.RawData.controls_from_model)):
            ax[4].plot(x, controls_from_model[i,:], label=cfg.pModel.RawData.controls_from_model[i])
        ax[4].legend(fontsize=6, bbox_to_anchor=(1.005, 1), loc='upper left')
        ax[4].set_title('controls from model')
    plt.show()

if __name__ == '__main__':
    """
    you can run this script with e.g. to set parameters
    python data_generation/src/fmu_simulate.py pModel.RawData.parameters.u_wall=1.8
    or for multirun
    python data_generation/src/fmu_simulate.py pModel.RawData.parameters.u_wall=1.8,2.0,2.2 --multirun
    """
    main()