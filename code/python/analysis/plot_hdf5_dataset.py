# what shall this do?
# return csv of data
# plot data
# give input: figsize, fontsize, linewidth
# give input: which dataset, id in that dataset.
# give input: controls, states, outputs as list of strings. in seperated plots.
# give input: parameters: true/false
# include if to check if data is available
# returns: figure, ax, csv
import h5py
import numpy as np
from analysis.plotting import settings, fig_size, cb_line_cycler, cb_line_cycler_solid, cb_marker_cycler, savefig
from filepaths import filepath_from_local_or_ml_artifacts
import matplotlib
settings()
import matplotlib.pyplot as plt
from cycler import cycler

def plot_dataset(
        file: h5py.File,
        dataset_type: str = 'common_test',
        sample_id: int = 0,
        figsize: tuple = (10, 10),
        fontsize: int = 9,
        linewidth: float = 1.0,
        controls: list = [],
        states: list = [],
        outputs: list = [],
        parameters: bool = False,
        ylim_range = False,
        xlim_range = None,
        save_path: str = None,
        display_states: list = [],
        display_controls: list = [],
        display_outputs: list = []
        ):


    # check if dataset is available
    if dataset_type not in file:
        raise ValueError(f"Dataset {dataset_type} not available in file {file.filename}")

    # calculate number of subplots
    n_subplots = len(controls) + len(states) + len(outputs) + int(parameters)
    
    # plot jobs timevarying
    jobs = [] # of shape [type, name, idx, display_name]
    for x in controls:
        jobs.append(['controls', x])
    for x in states:
        jobs.append(['states', x])
    for x in outputs:
        jobs.append(['outputs', x])

    # get position of variable and variable hat in variable_array
    for job in jobs:
        _names = np.array(file[job[0] + '_names'][:],dtype='str')
        _id = np.where(_names == job[1])[0][0]
        job.append(_id)
    # add display names if available, else use variable names
    for i, job in enumerate(jobs):
        if i < len(controls):
            if display_controls:
                job.append(display_controls[i])
            else:
                job.append(job[1])
        elif i < len(controls) + len(states):
            if display_states:
                job.append(display_states[i-len(controls)])
            else:
                job.append(job[1])
        else:
            if display_outputs:
                job.append(display_outputs[i-len(controls)-len(states)])
            else:
                job.append(job[1])

    fig, axs = plt.subplots(n_subplots, 1, figsize=figsize)
    plt.subplots_adjust(hspace=0.4, right=0.98)
    if n_subplots > 6:
        plt.subplots_adjust(top=0.93, bottom=0.03)
    for i in range(1, n_subplots-1):
        axs[i].sharex(axs[0])
    for i in range(n_subplots-1):
        # remove xtick labels
        plt.setp(axs[i].get_xticklabels(), visible=False)
        # remove xtick labels
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False)
    # if no params, delete second last subplot
    if not parameters:
        plt.setp(axs[-2].get_xticklabels(), visible=False)  
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False)
        axs[-1].sharex(axs[0])
    
    # set line cycler
    _cycler= (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["--", "-", "-.", ":", "-", "--", "-."]))
    # for states and outputs
    for ax in axs:
        ax.set_prop_cycle(_cycler)
    # for parameters
    if parameters:
        axs[-1].set_prop_cycle(cb_line_cycler)
    # for controls
    if controls:
        _cycler= (cycler(color=["#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", ":", "-", "--", "-."]))
        for i in range(len(controls)):
            axs[i].set_prop_cycle(_cycler)

        
    time = file['time'][:]
    # iterate over jobs
    for i, job in enumerate(jobs):
        # plot reconstruction if not controls
        if job[0] != 'controls':
            data_hat = file[dataset_type][job[0] + '_hat'][sample_id, job[2], :]
            axs[i].plot(time, data_hat, label=job[1] + '_hat', linewidth=linewidth)
        # plot data
        data = file[dataset_type][job[0]][sample_id, job[2], :]
        axs[i].plot(time, data, label=job[1], linewidth=linewidth)
        # set title
        _title = job[3]
        if job[0] == 'controls':
            _title = '$u(t)$: ' + _title
        elif job[0] == 'states':
            _title = '$x(t)$: ' + _title
        elif job[0] == 'outputs':
            _title = '$y(t)$: ' + _title
        axs[i].set_title(_title, fontsize=fontsize, pad = 1.0, loc = 'right')
        if ylim_range == 'min_max':
        # adjust y lims
            _min = file[dataset_type][job[0]][:, job[2], :].min()
            _max = file[dataset_type][job[0]][:, job[2], :].max()
            axs[i].set_ylim(_min, _max)
        elif type(ylim_range) == float or type(ylim_range) == int:
            _delta = 100 - ylim_range
            _min = np.percentile(file[dataset_type][job[0]][:, job[2], :], _delta/2)
            _max = np.percentile(file[dataset_type][job[0]][:, job[2], :], 100 - _delta/2)
            if _min > 0.9 * min(data):
                _min = min(data) * 0.9
            if _max < 1.1 * max(data):
                _max = max(data) * 1.1
            axs[i].set_ylim(_min, _max)

    # set x lim
    if xlim_range is not None:
        axs[0].set_xlim(xlim_range)
    else:
        axs[0].set_xlim(time[0], time[-1])
    # set x label
    # axs[len(jobs)-1].set_xlabel('time [s]', fontsize=fontsize-2, labelpad=-8.0, loc = 'right')

    # add parameters as bars
    if parameters:
        _parameters = file[dataset_type]['parameters'][sample_id, :]
        # get min and max of each parameter
        _param_min = np.min(file['train']['parameters'][:], axis=0)
        _param_max = np.max(file['train']['parameters'][:], axis=0)
        # normalize parameters
        _param_range = _param_max - _param_min
        _parameters = (_parameters - _param_min) / _param_range
        axs[-1].bar(np.arange(len(_parameters)), _parameters, label='parameters')
        # axs[-1].set_title('parameters', fontsize=fontsize, pad = 1.0, loc = 'left')
        _parameter_names = np.array(file['parameters_names'][:],dtype='str')
        axs[-1].set_xticks(np.arange(len(_parameters)))
        axs[-1].set_xticklabels(_parameter_names, rotation=20, ha='right', fontsize=fontsize-2)
        axs[-1].set_ylim(0, 1.0)
        axs[-1].set_ylabel(r'$p_{\text{norm}}$', fontsize=fontsize-2, labelpad=1.0)


    # make ticks smaller
    axs[len(jobs)-1].tick_params(axis='x', labelsize=fontsize-2, pad=2.0)
    if parameters:
        axs[-1].tick_params(axis='x', labelsize=fontsize-2)
    for ax in axs:
        ax.tick_params(axis='y', labelsize=fontsize-2)
        # also for offset
        # ax.yaxis.set_offset_position('left')
        ax.yaxis.get_offset_text().set_fontsize(fontsize-2)

    # add legend for prediction and true data at the top of the figure
    _handles, _labels = axs[len(controls)].get_legend_handles_labels()
    handles = _handles
    labels = ['true', 'pred.']
    if len(controls) > 0:
        _handles = axs[0].get_legend_handles_labels()
        handles.append(_handles[0][0])
        labels.append('$u(t)$')
    if parameters:
        _handles = axs[-1].get_legend_handles_labels()
        handles.append(_handles[0][0])
        labels.append('$p$')
    _ncols = 2 if len(labels) > 2 else 1
    fig.legend(handles, labels, loc='upper left', fontsize=fontsize-2, ncols = _ncols)

    plt.show()
    if save_path is not None:
        savefig(fig, save_path, pgf=False)
    return fig, axs


if __name__ == '__main__':
    path = filepath_from_local_or_ml_artifacts('mlflow-artifacts:/953802857536084821/0b79c425fb1c4a8bbbd451a74c587169/artifacts/dataset.hdf5')
    file = h5py.File(path, 'r')
    dataset_type = 'common_test'
    sample_id = 0
    size = fig_size(3.0, n_figs_per_width=2)
    controls = ['voltage']
    states = ['inductor1.i']
    outputs = ['resistor1.v']
    parameters = True
    fig = plot_dataset(file,
                dataset_type=dataset_type,
                sample_id=sample_id,
                figsize=size,
                fontsize=9,
                linewidth=1.0,
                controls=controls,
                states=states,
                outputs=outputs,
                parameters=parameters,
                ylim_range = False,
                save_path = None,
                # display_states = ['i'],
                # display_controls = ['v'],
                # display_outputs = ['v']
                )