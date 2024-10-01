import torch
import torch.nn as nn
import logging
import warnings
import h5py
from pathlib import Path
from networks.src.initialize_w_b import initialize_weights_biases
from networks.src.normalization import NormalizationLayerTimeSeries, NormalizationLayer1D
from networks.src.kullback_leibler import kullback_leibler, count_populated_dimensions
from networks.src.mixed_norm_for_torchdiffeq import _mixed_norm_tensor
import time as pyTime

from torchdiffeq import odeint, odeint_adjoint
import torchdiffeq as torchdiffeq
    
class GeneralEncoder(nn.Module):

    def __init__(self, 
                 input_dim: int,
                 lat_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 activation: nn.Module = nn.ELU,
                 initialization: str = 'identity',
                 include_parameters: bool = False,
                 param_dim: int = 0,
                 linear: bool = False,
                 ):
        # include_parameters and param_dim was added to incorporate a state and controls encoder that is dependent on parameters
        super().__init__()
        if param_dim > 0 and include_parameters is False:
            raise ValueError('param_dim > 0 but include_parameters is False')
        self.input_dim = input_dim + param_dim if include_parameters else input_dim
        self.lat_dim = lat_dim

        self.include_parameters = include_parameters
        # define layers
        self.normalization = NormalizationLayer1D(input_dim)
        if include_parameters:
            self.normalization_params = NormalizationLayer1D(input_dim)

        if not linear:
            modules = [
                nn.Linear(self.input_dim, hidden_dim),
                activation(),
            ]
            if n_layers < 2:
                logging.warning('n_layers must be at least 2, setting n_layers to 2')
            for i in range(n_layers-2):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(activation())
            modules.append(nn.Linear(hidden_dim, 2 * self.lat_dim))
            self.net = nn.Sequential(*modules)
        else:
            self.net= nn.Sequential(nn.Linear(input_dim, 2 * self.lat_dim, bias=True))
        
        initialize_weights_biases(self.net, initialization)

        # initialize mask
        self.register_buffer("mask_set", torch.tensor(False))
        self.register_buffer("mask", torch.ones(self.lat_dim))
    
    def set_mask(self, mask):
        self.mask_set.set_(torch.tensor(True, device=mask.device))
        self.mask.set_(torch.tensor(mask.clone().detach(), device=mask.device, dtype=torch.float32))

    def forward(self, x, params = None):
        """
        x: [batch_size, input_dim]
        """
        x = self.normalization(x)
        if self.include_parameters:
            params = self.normalization_params(params)
            x = torch.cat([x, params], dim=1)
        mu, logvar = torch.split(self.net(x), self.lat_dim, dim=1)

        if self.mask_set:
            mu = mu * self.mask
            logvar = torch.zeros(logvar.size(), device=logvar.device) # set logvar to zero if mask is set

        return mu, logvar
    
class LatentODEFunc(nn.Module):

    def __init__(self,
                 lat_state_mu_dim: int,
                 lat_control_dim: int,
                 lat_parameter_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 activation: nn.Module = nn.ELU,
                 initialization_ode: str = 'identity',
                 initialization_system_matrix: str = None,
                 lat_ode_type: str = 'variance_constant', # must be one of ['variance_constant', 'variance_dynamic', 'vanilla']
                 linear: bool = False,
                 lat_state_mu_independent: bool = False,
                 ):
        if initialization_system_matrix is not None and linear is False:
            logging.warning('initialization_system_matrix is only used for linear models, setting it to None')
            initialization_system_matrix = None
        if initialization_system_matrix is not None and linear is True and lat_parameter_dim > 0:
            logging.warning('initialization_system_matrix is only used for linear models without parameters, setting it to None')
            logging.warning('if you want to do eigenvalue shifting with parameters, you can use the move_eigvals_net method')
            initialization_system_matrix = None
        
        super().__init__()
        self.lat_state_dim = lat_state_mu_dim if not lat_ode_type == 'variance_dynamic' else 2 * lat_state_mu_dim
        self.lat_state_mu_dim = lat_state_mu_dim
        self.lat_control_dim = lat_control_dim
        self.lat_parameter_dim = lat_parameter_dim

        self.include_controls = True if lat_control_dim > 0 else False
        self.include_parameters = True if lat_parameter_dim > 0 else False

        self.lat_ode_type = lat_ode_type
        self.lat_state_mu_independent = lat_state_mu_independent
        if self.lat_ode_type != 'variance_dynamic' and self.lat_state_mu_independent:
            logging.warning('lat_state_mu_independent is only used for lat_ode_type variance_dynamic, setting it to False')
            self.lat_state_mu_independent, lat_state_mu_independent = False, False
        self.linear = linear

        # define layers
        if not linear:
            if lat_ode_type == 'variance_dynamic':
                if self.lat_state_mu_independent is False:
                    modules = self.construct_net(self.lat_state_dim, self.lat_state_dim, hidden_dim, n_layers, activation)
                    self.net = nn.Sequential(*modules)
                else:
                    modules = self.construct_net(lat_state_mu_dim, lat_state_mu_dim, hidden_dim, n_layers, activation) # (f(\mu(x), u, p))
                    self.net = nn.Sequential(*modules)
                    modules = self.construct_net(self.lat_state_dim, lat_state_mu_dim, hidden_dim, n_layers, activation) # (f(\mu(x), \sigma(x), u, p))
                    self.net_sigma = nn.Sequential(*modules)
            else:
                modules = self.construct_net(self.lat_state_dim, self.lat_state_dim, hidden_dim, n_layers, activation)
                self.net = nn.Sequential(*modules)
            initialize_weights_biases(self.net, initialization_ode)
            if self.lat_state_mu_independent:
                initialize_weights_biases(self.net_sigma, initialization_ode)
        else:
            # construct a linear state space model: xdot = Ax + Bu
            if not self.include_parameters:
                # if no parameters are included, we can use a simple linear model
                self.A = nn.Linear(self.lat_state_dim, self.lat_state_dim, bias=False)
                initialize_weights_biases(self.A, initialization_system_matrix)
                if self.include_controls:
                    self.B = nn.Linear(self.lat_control_dim, self.lat_state_dim, bias=False)
                    if initialization_ode == 'move_eigvals_net':
                        logging.warning('move_eigvals_net is an unsuitable initialization method for the control matrix')
                    initialize_weights_biases(self.B, initialization_ode)
            else:
                # if parameters are included, we need to calculate the matrices A and B depending on the parameters
                self.A_from_param = nn.Linear(self.lat_parameter_dim, self.lat_state_dim * self.lat_state_dim, bias=True)
                if self.include_controls:
                    self.B_from_param = nn.Linear(self.lat_parameter_dim, self.lat_state_dim * self.lat_control_dim, bias=True)
                _in_n_args = [self.lat_state_dim, self.lat_parameter_dim]
                if self.include_controls:
                    _in_n_args.append(self.lat_control_dim)
                _out_features = self.lat_state_dim                  
                initialize_weights_biases(self, initialization_ode, in_n_args=_in_n_args, out_features=_out_features)

            if self.lat_state_mu_independent is True: # TODO: test this
                # create a mask of the form for the system matrix A:
                # [[1,0]   * [\mu(x),
                #  [1,1]]    \sigma(x)]
                self.register_buffer("mask_A", torch.zeros(self.lat_state_dim, self.lat_state_dim))
                mask_A = torch.zeros(self.lat_state_dim, self.lat_state_dim)
                mask_A[:self.lat_state_mu_dim, :self.lat_state_mu_dim] = 1
                mask_A[self.lat_state_mu_dim:, :] = 1
                self.mask_A.set_(mask_A)
                assert self.mask_A.requires_grad == False

        # initialize mask
        self.register_buffer("mask_set", torch.tensor(False))
        self.register_buffer("mask", torch.ones(self.lat_state_dim))
    
    def set_mask(self, mask):
        '''
        mask: [lat_state_dim] or [lat_state_mu_dim] if lat_ode_type == 'variance_dynamic': mask for the latent state (boolean tensor or tensor with values 0 or 1)
        '''
        self.mask_set.set_(torch.tensor(True, device=mask.device))
        if self.lat_ode_type == 'variance_dynamic':
            _mask = torch.zeros(self.lat_state_dim, device=mask.device)
            _mask[:self.lat_state_mu_dim] = mask # set mask for mu
            # sigma mask is set to 1, as we do not want to include dynamics for sigma
            self.mask.set_(torch.tensor(_mask.clone().detach(), device=mask.device, dtype=torch.float32))
        else:
            self.mask.set_(torch.tensor(mask.clone().detach(), device=mask.device, dtype=torch.float32))
    
    def construct_net(self, state_in_dim, state_out_dim, hidden_dim, n_layers, activation):
        modules = [
                nn.Linear(state_in_dim + self.lat_control_dim + self.lat_parameter_dim, hidden_dim),
                activation(),
            ]
        if n_layers < 2:
            logging.warning('n_layers must be at least 2, setting n_layers to 2')#
        for i in range(n_layers-2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(activation())
        modules.append(nn.Linear(hidden_dim, state_out_dim))
        return modules

    def forward(self, lat_states, lat_parameters = None, lat_controls=None, init_training = False):
        """
        lat_state: [batch_size, lat_state_dim]
        lat_parameters: [batch_size, lat_parameter_dim]
        lat_controls: [batch_size, lat_control_dim]
        """
        if not self.linear:
            if not self.lat_ode_type == 'variance_dynamic' or (self.lat_ode_type == 'variance_dynamic' and self.lat_state_mu_independent is False):
                if self.include_controls and self.include_parameters:
                    x = torch.cat([lat_states, lat_parameters, lat_controls], dim=1)
                elif self.include_controls:
                    x = torch.cat([lat_states, lat_controls], dim=1)
                elif self.include_parameters:
                    x = torch.cat([lat_states, lat_parameters], dim=1)
                else:
                    x = lat_states
                lat_states_dot = self.net(x)
            else: # independent mu from sigma
                _states_mu = lat_states[:, :self.lat_state_mu_dim]
                if self.include_controls and self.include_parameters:
                    x_mu = torch.cat([_states_mu, lat_parameters, lat_controls], dim=1)
                    x_sigma = torch.cat([lat_states, lat_parameters, lat_controls], dim=1)
                elif self.include_controls:
                    x_mu = torch.cat([_states_mu, lat_controls], dim=1)
                    x_sigma = torch.cat([lat_states, lat_controls], dim=1)
                elif self.include_parameters:
                    x_mu = torch.cat([_states_mu, lat_parameters], dim=1)
                    x_sigma = torch.cat([lat_states, lat_parameters], dim=1)
                else:
                    x_mu = _states_mu
                    x_sigma = lat_states
                lat_state_mu_dot = self.net(x_mu)
                lat_state_sigma_dot = self.net_sigma(x_sigma)
                lat_states_dot = torch.cat([lat_state_mu_dot, lat_state_sigma_dot], dim=1)
        else:
            if self.include_parameters:
                A = self.A_from_param(lat_parameters)
                A = A.view(-1, self.lat_state_dim, self.lat_state_dim)
                if self.lat_state_mu_independent: # TODO: test this
                    A = A * self.mask_A # element-wise multiplication with mask
                lat_states_dot = torch.matmul(A, lat_states.unsqueeze(-1)).squeeze(-1)
                if self.include_controls:
                    B = self.B_from_param(lat_parameters)
                    B = B.view(-1, self.lat_state_dim, self.lat_control_dim)
                    lat_states_dot += torch.matmul(B, lat_controls.unsqueeze(-1)).squeeze(-1)
            else:
                if self.lat_state_mu_independent: # TODO: test this
                    self.A.weight = torch.nn.Parameter(self.A.weight * self.mask_A)
                lat_states_dot = self.A(lat_states)
                if self.include_controls:
                    lat_states_dot += self.B(lat_controls)
        
        if self.mask_set:
            lat_states_dot = lat_states_dot * self.mask

        return lat_states_dot
    
class Decoder(nn.Module):

    def __init__(self, 
                 lat_state_mu_dim: int,
                 lat_control_dim: int,
                 lat_parameter_dim: int,
                 state_dim: int,
                 outputs_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 activation: nn.Module = nn.ELU,
                 initialization: str = 'identity',
                 linear: bool = False,
                 ):
        super().__init__()
        self.lat_state_mu_dim = lat_state_mu_dim
        self.lat_control_dim = lat_control_dim
        self.lat_parameter_dim = lat_parameter_dim

        self.out_dim = state_dim + outputs_dim
        self.state_dim = state_dim
        self.outputs_dim = outputs_dim

        self.include_controls = True if lat_control_dim > 0 else False
        self.include_parameters = True if lat_parameter_dim > 0 else False
        self.include_outputs = True if outputs_dim > 0 else False

        self.linear = linear

        # define layers
        self.state_normalization = NormalizationLayer1D(self.out_dim)
        self.outputs_normalization = NormalizationLayer1D(outputs_dim) if outputs_dim > 0 else None

        if not linear:
            modules = [
                nn.Linear(self.lat_state_mu_dim + self.lat_control_dim + self.lat_parameter_dim, hidden_dim),
                activation(),
            ]
            if n_layers < 2:
                logging.warning('n_layers must be at least 2, setting n_layers to 2')
            for i in range(n_layers-2):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(activation())
            modules.append(nn.Linear(hidden_dim, self.out_dim))
            self.net = nn.Sequential(*modules)

            initialize_weights_biases(self.net, initialization)
        else:
            # construct a linear state space model: y = Cx + Du
            if not self.include_parameters:
                self.C = nn.Linear(self.lat_state_mu_dim, self.out_dim, bias=True) # a bias shouldn't be harmful
                initialize_weights_biases(self.C, initialization)
                if self.include_controls:
                    self.D = nn.Linear(self.lat_control_dim, self.out_dim, bias=False)
                    initialize_weights_biases(self.D, initialization)
            else:
                self.C_and_constant_from_param = nn.Linear(self.lat_parameter_dim, self.out_dim * self.lat_state_mu_dim + self.out_dim, bias=True) # to get C matrix + constant
                initialize_weights_biases(self.C_and_constant_from_param, initialization)
                if self.include_controls:
                    self.D_from_param = nn.Linear(self.lat_parameter_dim, self.out_dim * self.lat_control_dim, bias=True)
                    initialize_weights_biases(self.D_from_param, initialization)

    def forward(self, 
                lat_state: torch.Tensor,
                lat_parameters: torch.Tensor = None,
                lat_controls: torch.Tensor = None,
                ):
        """
        lat_state: [batch_size, lat_state_mu_dim]
        lat_parameters: [batch_size, lat_parameter_dim]
        lat_controls: [batch_size, lat_control_dim]
        """
        if not self.linear:
            if self.include_controls and self.include_parameters:
                lat_state = torch.cat([lat_state, lat_parameters, lat_controls], dim=1)
            elif self.include_controls:
                lat_state = torch.cat([lat_state, lat_controls], dim=1)
            elif self.include_parameters:
                lat_state = torch.cat([lat_state, lat_parameters], dim=1)
            x = self.net(lat_state)
        else:
            if not self.include_parameters:
                x = self.C(lat_state)
                if self.include_controls:
                    x += self.D(lat_controls)
            else:
                _C = self.C_and_constant_from_param(lat_parameters)
                C, constant = torch.split(_C, [self.out_dim * self.lat_state_mu_dim, self.out_dim], dim=1) # TODO: check if this is correct
                C = C.view(-1, self.out_dim, self.lat_state_mu_dim)
                constant = constant.reshape(-1, self.out_dim)
                x = torch.matmul(C, lat_state.unsqueeze(-1)).squeeze(-1) + constant
                if self.include_controls:
                    D = self.D_from_param(lat_parameters)
                    D = D.view(-1, self.out_dim, self.lat_control_dim)
                    x += torch.matmul(D, lat_controls.unsqueeze(-1)).squeeze(-1)
        state_norm, outputs_norm = torch.split(x, [self.state_dim, self.outputs_dim], dim=1) if self.include_outputs else (x, None)
        state = self.state_normalization(state_norm, denormalize=True) 
        outputs = self.outputs_normalization(outputs_norm, denormalize=True) if self.include_outputs else None
        return torch.cat([state_norm, outputs_norm, state, outputs], dim=1) if self.include_outputs else torch.cat([state_norm, state], dim=1)
    
    def split_return(self, x):
        if self.include_outputs:
            state_norm, outputs_norm, state, outputs = torch.split(x, [self.state_dim, self.outputs_dim, self.state_dim, self.outputs_dim], dim=1) 
        else:
            state_norm, state = torch.split(x, [self.state_dim, self.state_dim], dim=1)
            outputs_norm, outputs = None, None
        return state, outputs, state_norm, outputs_norm
    
class BalancedNeuralODE(nn.Module): 
    
    def __init__(self,
                 states_dim: int,
                 lat_states_mu_dim: int,
                 parameters_dim: int,
                 lat_parameters_dim: int,
                 controls_dim: int,
                 lat_controls_dim: int,
                 outputs_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 controls_to_decoder: bool = True,
                 activation: nn.Module = nn.ELU,
                 initialization_type: str = 'identity',
                 initialization_type_ode: str = 'identity',
                 initialization_type_ode_matrix: str = None,
                 kl_timeseries_aggregation_mode: str = 'mean', # must be one of ['sum', 'mean', 'max']
                 lat_ode_type: str = 'variance_constant', # must be one of ['variance_constant', 'variance_dynamic', 'vanilla']
                 params_to_state_and_control_encoder: bool = False,
                 state_encoder_linear: bool = False,
                 control_encoder_linear: bool = False,
                 parameter_encoder_linear: bool = False,
                 ode_linear: bool = False,
                 decoder_linear: bool = False,
                 lat_state_mu_independent: bool = False,
                 ):
        super().__init__()

        # set dimensions
        self.states_dim = states_dim
        self.lat_states_mu_dim = lat_states_mu_dim 
        # we differentiate between lat_states_mu_dim and lat_states_dim
        # lat_states_dim is [lat_states_mu_dim + lat_states_logvar_dim] if lat_ode_type=='variance_dynamic' 
        # and [lat_states_mu_dim] if lat_ode_type=='variance_constant' or 'vanilla'
        self.parameters_dim = parameters_dim
        self.lat_parameters_dim = lat_parameters_dim
        self.controls_dim = controls_dim
        self.lat_controls_dim = lat_controls_dim
        self.outputs_dim = outputs_dim

        self.ode_fun_count = 0
        
        # kl settings
        self.kl_timeseries_aggregation_mode = kl_timeseries_aggregation_mode
        self.lat_ode_type = lat_ode_type

        # control input integration (see set_input)
        self.eps_lat_controls = None
        self.current_lat_controls = None
        self.current_controls = None
        self.current_times = None
        self.current_lat_parameters = None
        if lat_ode_type == 'variance_constant': 
            self.lat_state_0_logvar = None
        
        # allocate alpha values
        if lat_ode_type == 'variance_constant':
            self.alpha_mu = None
        if lat_ode_type == 'variance_dynamic':
            self.alpha_mu = None
            self.alpha_sigma = None

        # set include flags from dimensions
        self.include_parameters = True if parameters_dim > 0 else False
        self.include_controls = True if controls_dim > 0 else False
        self.include_outputs = True if outputs_dim > 0 else False
        
        # set additional flags for encoder and decoder
        self.controls_to_decoder = controls_to_decoder and self.include_controls
        if self.include_parameters:
            if params_to_state_and_control_encoder:
                self.params_to_state_and_control_encoder = True
            else:
                self.params_to_state_and_control_encoder = False
        else:
            self.params_to_state_and_control_encoder = False

        # initialize models
        self.latent_ode_func = LatentODEFunc(lat_states_mu_dim, 
                                             lat_controls_dim if self.include_controls else 0,
                                             lat_parameters_dim if self.include_parameters else 0,
                                             hidden_dim, n_layers, activation, initialization_type_ode, 
                                             initialization_system_matrix=initialization_type_ode_matrix,
                                             lat_ode_type=lat_ode_type, linear=ode_linear,
                                             lat_state_mu_independent=lat_state_mu_independent)
        
        self.state_encoder = GeneralEncoder(states_dim, lat_states_mu_dim, hidden_dim, n_layers, activation, initialization_type, 
                                            include_parameters=self.params_to_state_and_control_encoder, 
                                            param_dim=parameters_dim if self.params_to_state_and_control_encoder else 0, 
                                            linear=state_encoder_linear)
        self.parameter_encoder = GeneralEncoder(parameters_dim, lat_parameters_dim, hidden_dim, n_layers, activation, initialization_type,
                                                linear=parameter_encoder_linear) if self.include_parameters else None
        self.controls_encoder = GeneralEncoder(controls_dim, lat_controls_dim, hidden_dim, n_layers, activation, initialization_type, 
                                               include_parameters=self.params_to_state_and_control_encoder, 
                                               param_dim=parameters_dim if self.params_to_state_and_control_encoder else 0,
                                               linear=control_encoder_linear) if self.include_controls else None

        self.decoder = Decoder(lat_states_mu_dim, lat_controls_dim if self.controls_to_decoder else 0, lat_parameters_dim if self.include_parameters else 0,
                               states_dim, outputs_dim, hidden_dim, n_layers, activation, initialization_type, linear=decoder_linear)

        # initialize boolean for deterministic mode
        self.register_buffer("deterministic_mode_active_masks_set", torch.tensor(False))
    
    def activate_deterministic_mode(self, mask_lat_states, mask_lat_controls, mask_lat_parameters):
        self.latent_ode_func.set_mask(mask_lat_states)
        self.state_encoder.set_mask(mask_lat_states)
        if self.include_controls:
            self.controls_encoder.set_mask(mask_lat_controls)
        if self.include_parameters:
            self.parameter_encoder.set_mask(mask_lat_parameters)
        self.deterministic_mode_active_masks_set.set_(torch.tensor(True, device=mask_lat_states.device))
        logging.info('All masks set for deterministic mode, deterministic mode activated')
        
    def normalization_init(self, dataset: h5py.File):
        def reshape_array(array):
            arr = array.transpose(1,0,2).reshape(array.shape[1],-1).transpose(1,0)
            return arr
        
        # states
        _states = reshape_array(dataset['train']['states'][:])
        self.state_encoder.normalization.initialize_normalization(_states)
        self.decoder.state_normalization.initialize_normalization(_states)

        # state derivatives
        # None

        # parameters
        if self.include_parameters:
            _parameters = dataset['train']['parameters'][:]
            self.parameter_encoder.normalization.initialize_normalization(_parameters)
            if self.params_to_state_and_control_encoder:
                self.state_encoder.normalization_params.initialize_normalization(_parameters)
                if self.include_controls:
                    self.controls_encoder.normalization_params.initialize_normalization(_parameters)
        
        # controls
        if self.include_controls:
            _controls = reshape_array(dataset['train']['controls'][:])
            self.controls_encoder.normalization.initialize_normalization(_controls)

        # outputs
        if self.include_outputs:
            _outputs = reshape_array(dataset['train']['outputs'][:])
            self.decoder.outputs_normalization.initialize_normalization(_outputs)

        logging.info('Normalization layers initialized')

    def set_input(self, lat_controls = None, controls = None, times = None, lat_parameters = None, reparam_active = False):
        self.current_lat_controls = lat_controls
        self.current_controls = controls
        self.current_times = times
        self.current_lat_parameters = lat_parameters
        self.current_reparam_active = reparam_active
        
        if self.include_controls:
            if controls is not None: # switch between use_adjoint and not use_adjoint depending on call of this function
                self.use_adjoint = True
                # sample and save new eps
                self.eps_lat_controls = torch.randn(controls.shape[0], self.lat_controls_dim, controls.shape[2], device=controls.device).detach() if self.include_controls else None
            else:
                self.use_adjoint = False

    def forward_ODE(self, t, lat_states):
        self.ode_fun_count += 1

        # determine idx in equidistant time vector and choose control input at this time
        try:
            idx = torch.nonzero(self.current_times > t)
            if len(idx) == 0:
                idx = -1
            else:
                idx = idx[0][0].detach() - 1
        except:
            raise ValueError('something went wrong when trying to get the control input at time t, using the last control input instead')
        if self.include_controls:
            if self.use_adjoint is True:
                u = self.current_controls[:,:,idx]
                u_lat_mu, u_lat_logvar = self.controls_encoder(u)
                u_lat = self.reparametrize_with_eps(u_lat_mu, u_lat_logvar, self.eps_lat_controls[:,:,idx], self.current_reparam_active)
            else: 
                u_lat = self.current_lat_controls[:,:,idx]
        else:
            u_lat = None

        # get latent parameters
        if self.include_parameters:
            lat_parameters = self.current_lat_parameters
        else:
            lat_parameters = None
        
        # swapaxes to convert from torchdiffeq convention to ours
        lat_states = lat_states.swapaxes(0,1)

        # put noise on states for the different lat_ode_types
        if self.lat_ode_type == 'variance_constant': 
            lat_states = self.reparametrize_with_eps(lat_states, self.lat_state_0_logvar, self.eps_lat_states[:,:,idx], self.current_reparam_active, self.alpha_mu)
        elif self.lat_ode_type == 'variance_dynamic':
            # split state vector
            lat_states_mu = lat_states[:, :self.lat_states_mu_dim]
            lat_states_logvar = lat_states[:, self.lat_states_mu_dim:]
            # get eps
            eps_mu = self.eps_lat_states[:, :self.lat_states_mu_dim, idx]
            eps_logvar = self.eps_lat_states[:, self.lat_states_mu_dim:, idx]
            # reparameterize
            lat_states_mu_w_noise = self.reparametrize_with_eps(lat_states_mu, lat_states_logvar, eps_mu, self.current_reparam_active, self.alpha_mu)
            lat_states_logvar_w_noise = self.reparametrize_with_eps(lat_states_logvar.mul(0.5).exp(), lat_states_logvar.mul(0.5).exp(),
                                                                     eps_logvar, self.current_reparam_active, self.alpha_sigma).log().mul(2) #TODO: check this
            # concatenate state vector
            lat_states = torch.concat([lat_states_mu_w_noise, lat_states_logvar_w_noise], dim=1)
        elif self.lat_ode_type == 'vanilla':
            lat_states = lat_states

        # call latent ode function with lat_states, lat_parameters, u_lat
        lat_states_dot = self.latent_ode_func(lat_states, lat_parameters, u_lat)

        # swapaxes back to torchdiffeq convention
        lat_states_dot = lat_states_dot.swapaxes(0,1)
        return lat_states_dot
    
    def reparametrize(self, mu, logvar, device, reparam_active = False):
        eps = torch.randn_like(mu, device=device)
        std = logvar.mul(0.5).exp()
        z_latent = eps.mul(std).add_(mu) if reparam_active is True else mu
        return z_latent
    
    def reparametrize_with_eps(self, mu, logvar, eps, reparam_active = False, alpha=1.0):
        if reparam_active is False or alpha == 0.0:
            z_latent = mu
        else:
            std = logvar.mul(0.5).exp() * alpha
            z_latent = eps.mul(std).add_(mu)
        return z_latent
    
    def model_and_loss_evaluation(self, data_batch, train_cfg, pre_train, device, return_model_outputs, test = False, last_batch = True, activate_deterministic_mode = False):
        # check if reparametrization is active
        deterministic_mode_active = self.deterministic_mode_active_masks_set
        if deterministic_mode_active:
            reparam_active = False
        else:
            reparam_active = not test
        
        # get data
        time = data_batch['time'].to(device)
        states = data_batch['states'].to(device)
        parameters = data_batch['parameters'].to(device) if 'parameters' in data_batch.keys() else None
        controls = data_batch['controls'].to(device) if 'controls' in data_batch.keys() else None
        outputs = data_batch['outputs'].to(device) if 'outputs' in data_batch.keys() else None

        # squeeze data if in pre_train
        if pre_train is True:
            time = time.squeeze(2)
            states = states.squeeze(2)
            controls = controls.squeeze(2) if controls is not None else None
            outputs = outputs.squeeze(2) if outputs is not None else None
        
        """pre-training"""
        if pre_train is True:
            raise NotImplementedError
        
        """main-training"""
        # for the following, the implementation might be a bit awkward:
        # when using odeint, we can use an elegant way, encode all the data first, save controls and parameters in the LatentODE class, then call odeint
        #      on the forward_ODE function, which then uses the saved controls and parameters. The decoder is then called on the latent state trajectory 
        #      to get the model outputs by using the encoded controls and parameters.
        # when using odeint_adjoint, this approach (can) lead to performance issues, as the adjoint method has problems to determine the influences of the data encoding.
        #      Therefore, we save the unencoded control signals to the LatentODE class, and encode them in the forward_ODE function. The parameters can be encoded once and saved then.
        #      The decoder is then called on the latent state trajectory to get the model outputs by using the encoded controls and parameters. But because the controls are 
        #      encoded in the forward_ODE, we need to generate them again. 
        if pre_train is False:
            _time_logging0 = pyTime.time() 

            '''Encode data'''
            # encode states and parameters
            lat_state_0_mu, lat_state_0_logvar = self.state_encoder(states[:,:,0], parameters if self.params_to_state_and_control_encoder else None)
            lat_parameters_mu, lat_parameters_logvar = self.parameter_encoder(parameters) if self.include_parameters else (None, None)

            # encode controls 
            if self.include_controls: # we need this anyways for the decoder, both if we use odeint or odeint_adjoint
                # lat_controls_mu = torch.empty(controls.shape[0], self.lat_controls_dim, controls.shape[2], device=device)
                # lat_controls_logvar = torch.empty(controls.shape[0], self.lat_controls_dim, controls.shape[2], device=device)
                # for i in range(controls.shape[2]):
                #     lat_controls_mu[:,:,i], lat_controls_logvar[:,:,i] = self.controls_encoder(controls[:,:,i], parameters if self.params_to_state_and_control_encoder else None)
                
                # do this with one call by using reshape: 
                #   my convention is [batch_size, feature_dim, time_steps] but to use the encoder efficiently, we need [batch_size*time_steps, feature_dim]. For that, we need to permute the tensor
                #   to [batch_size, time_steps, feature_dim] and then reshape it to [batch_size*time_steps, feature_dim] (batch_size and time_steps collapsed). Then, do the encoding and 
                #   reshape it back to [batch_size, time_steps, feature_dim] and permute it back to [batch_size, feature_dim, time_steps]
                _controls = controls.permute(0,2,1).reshape(controls.shape[0]*controls.shape[2], controls.shape[1])
                _parameters = parameters.unsqueeze(2).expand(-1, -1, controls.shape[2]).permute(0,2,1).reshape(parameters.shape[0]*controls.shape[2], parameters.shape[1]) if self.params_to_state_and_control_encoder else None
                _lat_controls_mu, _lat_controls_logvar = self.controls_encoder(_controls, _parameters)
                lat_controls_mu = _lat_controls_mu.reshape(controls.shape[0], controls.shape[2], self.lat_controls_dim).permute(0,2,1)
                lat_controls_logvar = _lat_controls_logvar.reshape(controls.shape[0], controls.shape[2], self.lat_controls_dim).permute(0,2,1)
            else:
                lat_controls_mu, lat_controls_logvar = None, None

            time = time[0,0,:] # as we used equidistant time steps in data generation, we can use the first time vector

            '''Reparametrize data'''
            # reparametrize parameters
            lat_parameters = self.reparametrize(lat_parameters_mu, lat_parameters_logvar, device, reparam_active) if self.include_parameters else None
            
            # prepare states depending on lat_ode_type
            # the meaning of lat_state is different for the different lat_ode_types
            if self.lat_ode_type == 'variance_constant':
                lat_state_0 = lat_state_0_mu 
                self.lat_state_0_logvar = lat_state_0_logvar
            elif self.lat_ode_type == 'variance_dynamic':
                lat_state_0 = torch.cat([lat_state_0_mu, lat_state_0_logvar], dim=1)
            elif self.lat_ode_type == 'vanilla':
                lat_state_0 = self.reparametrize(lat_state_0_mu, lat_state_0_logvar, device, reparam_active)
            # set alpha values from train_cfg
            
            # save eps for reparametrization of states
            if self.lat_ode_type in ['variance_constant', 'variance_dynamic']:
                self.eps_lat_states = torch.randn(lat_state_0_logvar.shape[0], self.latent_ode_func.lat_state_dim, len(time), device=time.device).detach()
            
            '''Set input (parameter, controls) for ODE integration --> this is different for odeint and odeint_adjoint'''
            # set inputs depending on use_adjoint differently, reparametrize controls if use_adjoint is True
            if train_cfg.use_adjoint is True: 
                self.set_input(None, controls, time, lat_parameters, reparam_active) # set physical controls and times, and latent parameters. controls are encoded in forward_ODE
                # during calling set_input, and eps is sampled. We can use the same eps here to reparametrize the controls
                lat_controls = self.reparametrize_with_eps(lat_controls_mu, lat_controls_logvar, self.eps_lat_controls, reparam_active) if self.include_controls else None
            else:
                lat_controls = self.reparametrize(lat_controls_mu, lat_controls_logvar, device, reparam_active) if self.include_controls else None
                self.set_input(lat_controls, None, time, lat_parameters, reparam_active) # set latent controls and times, and latent parameters
            
            '''Set alpha values'''
            if self.lat_ode_type == 'variance_constant':
                self.alpha_mu = train_cfg.alpha_mu
            if self.lat_ode_type == 'variance_dynamic':
                self.alpha_mu = train_cfg.alpha_mu
                self.alpha_sigma = train_cfg.alpha_sigma
            if self.lat_ode_type == 'vanilla':
                pass

            '''ODE integration'''
            lat_state_0 = lat_state_0.swapaxes(0,1) # lat_state_0 is shape [batch_size, states_dim], but for odeint it must be [states_dim, batch_size]
            # logging
            self.ode_fun_count = 0

            # prepare options for odeint
            _fixed_step_solvers = ['euler', 'midpoint', 'rk4', 'implicit_adams', 'explicit_adams']
            _base_options = {}
            if train_cfg.solver_norm == 'max':
                _base_options['norm'] = torchdiffeq._impl.misc._linf_norm
            elif train_cfg.solver_norm == 'mixed':
                _base_options['norm'] = _mixed_norm_tensor
            if self.include_controls or train_cfg.evaluate_at_control_times is True:
                if train_cfg.evaluate_at_control_times is True:
                    if train_cfg.solver in _fixed_step_solvers:
                        _base_options['perturb'] = True
                    else:
                        _base_options['jump_t'] = time
            options = _base_options.copy()
            
            if train_cfg.use_adjoint is True:           
                adjoint_options = _base_options.copy()
                adjoint_options['norm'] = 'seminorm'
                lat_states = odeint_adjoint(self.forward_ODE, lat_state_0, time, 
                                method=train_cfg.solver,
                                rtol = train_cfg.solver_rtol, 
                                atol = train_cfg.solver_atol,
                                adjoint_params=self.parameters(), # or should this be self.latent_ode_func.parameters()?
                                adjoint_options=adjoint_options,
                                options=options,)
            else:
                lat_states = odeint(self.forward_ODE, lat_state_0, time, # this is the preferred ways, as it is fast and exact
                                method=train_cfg.solver,
                                rtol = train_cfg.solver_rtol, 
                                atol = train_cfg.solver_atol,
                                options=options,)
            
            # logging stuff
            _ode_calls_forward = self.ode_fun_count
            self.ode_fun_count = 0
            time_odeint = pyTime.time() - _time_logging0
            _time_logging0 = pyTime.time()

            # swap axes back
            lat_states = lat_states.swapaxes(0,2) # lat_states is of shape [timeseries_length, states_dim, batch_size], but we need [batch_size, states_dim, timeseries_length]

            '''Unpack lat_states and reparameterize states depending on lat_ode_type'''
            # reuse eps for reparametrization of states, that was used for noise on states
            if self.lat_ode_type == 'variance_constant':
                lat_states_mu = lat_states.clone()
                lat_states = self.reparametrize_with_eps(lat_states_mu, self.lat_state_0_logvar.unsqueeze(2).repeat(1,1,lat_states.shape[2]), self.eps_lat_states, self.current_reparam_active)
            elif self.lat_ode_type == 'variance_dynamic': 
                lat_states_mu = lat_states[:, :self.lat_states_mu_dim, :]
                lat_states_logvar = lat_states[:, self.lat_states_mu_dim:, :]
                lat_states = self.reparametrize_with_eps(lat_states_mu, lat_states_logvar, self.eps_lat_states[:, :self.lat_states_mu_dim, :], self.current_reparam_active)
            elif self.lat_ode_type == 'vanilla':
                pass

            # decode data / get model outputs

            # states_hat = torch.empty_like(states)
            # states_hat_norm = torch.empty_like(states)
            # outputs_hat = torch.empty_like(outputs) if self.include_outputs else None
            # outputs_hat_norm = torch.empty_like(outputs) if self.include_outputs else None
            # for i in range(states.shape[2]): 
            #     _res = self.decoder(lat_states[:,:,i], lat_parameters, lat_controls[:,:,i] if self.controls_to_decoder else None)
            #     states_hat[:,:,i], outputs_hat[:,:,i], states_hat_norm[:,:,i], outputs_hat_norm[:,:,i] = self.decoder.split_return(_res)
            
            # do this with one call
            _lat_states = lat_states.permute(0,2,1).reshape(lat_states.shape[0]*lat_states.shape[2], lat_states.shape[1])
            _lat_parameters = lat_parameters.unsqueeze(2).expand(-1, -1, lat_states.shape[2]).permute(0,2,1).reshape(lat_parameters.shape[0]*lat_states.shape[2], lat_parameters.shape[1]) if self.include_parameters else None
            _lat_controls = lat_controls.permute(0,2,1).reshape(lat_controls.shape[0]*lat_controls.shape[2], lat_controls.shape[1]) if self.include_controls else None
            _res = self.decoder(_lat_states, _lat_parameters, _lat_controls)
            _res = _res.reshape(lat_states.shape[0], lat_states.shape[2], -1).permute(0,2,1)
            states_hat, outputs_hat, states_hat_norm, outputs_hat_norm = self.decoder.split_return(_res)
            
            # Normalize ground truth data (states, outputs), for use in loss calculation
            
            # states_norm = torch.empty_like(states)
            # outputs_norm = torch.empty_like(outputs) if self.include_outputs else None
            # with torch.no_grad():
            #     for i in range(states.shape[2]):
            #         states_norm[:,:,i] = self.decoder.state_normalization(states[:,:,i])
            #         if self.include_outputs:
            #             outputs_norm[:,:,i] = self.decoder.outputs_normalization(outputs[:,:,i])
            
            # do this with one call
            _states = states.permute(0,2,1).reshape(states.shape[0]*states.shape[2], states.shape[1])
            _outputs = outputs.permute(0,2,1).reshape(outputs.shape[0]*outputs.shape[2], outputs.shape[1]) if self.include_outputs else None
            _states_norm = self.decoder.state_normalization(_states)
            _outputs_norm = self.decoder.outputs_normalization(_outputs) if self.include_outputs else None
            states_norm = _states_norm.reshape(states.shape[0], states.shape[2], states.shape[1]).permute(0,2,1)
            outputs_norm = _outputs_norm.reshape(outputs.shape[0], outputs.shape[2], outputs.shape[1]).permute(0,2,1) if self.include_outputs else None

            # detach from computational graph (for safety)
            states_norm = states_norm.detach()
            outputs_norm = outputs_norm.detach() if self.include_outputs else None

            '''Calculate loss'''
            # calculate reconstruction loss
            reconstruction_loss_state_0 = torch.mean(torch.square(states_hat_norm[:,:,0] - states_norm[:,:,0]))
            reconstruction_loss_states = torch.mean(torch.square(states_hat_norm - states_norm))
            reconstruction_loss_outputs = torch.mean(torch.square(outputs_hat_norm - outputs_norm)) if self.include_outputs else 0
            
            reconstruction_loss_scaler = 1 / ( 1 + float(self.include_outputs) )
            reconstruction_loss_scaler += float(train_cfg.include_reconstruction_loss_state0)
            reconstruction_loss = reconstruction_loss_states + reconstruction_loss_outputs
            if train_cfg.include_reconstruction_loss_state0 is True:
                reconstruction_loss += reconstruction_loss_state_0

            # calculate RMSE
            rmse_state_0 = torch.sqrt(reconstruction_loss_state_0)
            rmse_states = torch.sqrt(reconstruction_loss_states)
            rmse_outputs = torch.sqrt(reconstruction_loss_outputs) if self.include_outputs else 0

            # calculate KL loss
            if deterministic_mode_active: # if deterministic mode is active, we set the logvars to 1 as we only need the means
                if self.lat_ode_type == 'variance_constant':
                    lat_state_0_logvar = torch.zeros_like(lat_state_0_logvar, device=device)
                elif self.lat_ode_type == 'variance_dynamic':
                    lat_states_logvar = torch.zeros_like(lat_states_logvar, device=device)
                lat_parameters_logvar = torch.zeros_like(lat_parameters_logvar, device=device) if self.include_parameters else None
                lat_controls_logvar = torch.zeros_like(lat_controls_logvar, device=device) if self.include_controls else None

            kl_lat_state_0 = kullback_leibler(lat_state_0_mu, lat_state_0_logvar)
            kl_lat_parameters = kullback_leibler(lat_parameters_mu, lat_parameters_logvar) if self.include_parameters else 0
            kl_lat_controls = kullback_leibler(lat_controls_mu, lat_controls_logvar, time_series_aggregation_mode=self.kl_timeseries_aggregation_mode) if self.include_controls else 0
            kl_loss_scaler = 1 / (self.states_dim + self.parameters_dim + self.controls_dim)
            if self.lat_ode_type == 'variance_constant':
                kl_lat_states = kullback_leibler(lat_states_mu, lat_state_0_logvar.unsqueeze(2).repeat(1,1,lat_states_mu.shape[2]), time_series_aggregation_mode=self.kl_timeseries_aggregation_mode)
            elif self.lat_ode_type == 'variance_dynamic':
                kl_lat_states = kullback_leibler(lat_states_mu, lat_states_logvar, time_series_aggregation_mode=self.kl_timeseries_aggregation_mode)
            else:
                kl_lat_states = 0

            # calculate loss
            kl_loss = kl_lat_parameters + kl_lat_controls
            if self.lat_ode_type == 'variance_constant' or self.lat_ode_type == 'variance_dynamic':
                kl_loss += kl_lat_states
            elif self.lat_ode_type == 'vanilla':
                kl_loss += kl_lat_state_0

            loss = reconstruction_loss_scaler * reconstruction_loss 
            if not deterministic_mode_active:
                loss += train_cfg.beta_start * kl_loss_scaler * kl_loss

            '''Count populated dimensions'''
            if last_batch or activate_deterministic_mode: # we only need to compute this for the last batch, as we only need the values for logging
                # for states depending on lat_ode_type
                if not deterministic_mode_active:
                    lat_dim_state_0_populated, idx_lat_dim_state_0 = count_populated_dimensions(lat_state_0_mu, lat_state_0_logvar, train_cfg.threshold_count_populated_dimensions, return_idx=activate_deterministic_mode)
                    if self.lat_ode_type == 'variance_constant':
                        lat_dim_states_populated, idx_lat_dim_states = count_populated_dimensions(lat_states_mu, lat_state_0_logvar.unsqueeze(2).repeat(1,1,lat_states_mu.shape[2]), train_cfg.threshold_count_populated_dimensions, kl_timeseries_aggregation_mode=self.kl_timeseries_aggregation_mode, return_idx=activate_deterministic_mode)
                    elif self.lat_ode_type == 'variance_dynamic':
                        lat_dim_states_populated, idx_lat_dim_states = count_populated_dimensions(lat_states_mu, lat_states_logvar, train_cfg.threshold_count_populated_dimensions, kl_timeseries_aggregation_mode=self.kl_timeseries_aggregation_mode, return_idx=activate_deterministic_mode)
                    elif self.lat_ode_type == 'vanilla':
                        lat_dim_states_populated = 0
                    # for parameters and controls
                    lat_dim_parameters_populated, idx_lat_dim_parameters = count_populated_dimensions(lat_parameters_mu, lat_parameters_logvar, train_cfg.threshold_count_populated_dimensions, return_idx=activate_deterministic_mode) if self.include_parameters else (0, None)
                    lat_dim_controls_populated, idx_lat_dim_controls = count_populated_dimensions(lat_controls_mu, lat_controls_logvar, train_cfg.threshold_count_populated_dimensions, kl_timeseries_aggregation_mode=self.kl_timeseries_aggregation_mode, return_idx=activate_deterministic_mode) if self.include_controls else (0, None)
                    # activate here deterministic mode, i.e. set masks and boolean flags
                    if activate_deterministic_mode:
                        if self.deterministic_mode_active_masks_set:
                            raise ValueError('Deterministic mode is already active, but we tried to activate it again')
                        if self.lat_ode_type == 'vanilla':
                            raise ValueError('Deterministic mode is not supported for lat_ode_type "vanilla"')
                        self.activate_deterministic_mode(idx_lat_dim_states, idx_lat_dim_controls, idx_lat_dim_parameters)
                else: # when deterministic mode is active, we only need to count the dimensions stored in the masks
                    lat_dim_state_0_populated = self.latent_ode_func.mask.sum()
                    lat_dim_states_populated = self.latent_ode_func.mask.sum()
                    lat_dim_parameters_populated = self.parameter_encoder.mask.sum() if self.include_parameters else 0
                    lat_dim_controls_populated = self.controls_encoder.mask.sum() if self.include_controls else 0
            else:
                lat_dim_state_0_populated, lat_dim_states_populated, lat_dim_parameters_populated, lat_dim_controls_populated = 0, 0, 0, 0
                

            time_outputs = pyTime.time() - _time_logging0
            # make ret_vals
            ret_val={
                'loss': loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss,
                'reconstruction_loss_state_0': reconstruction_loss_state_0,
                'reconstruction_loss_states': reconstruction_loss_states,
                'reconstruction_loss_outputs': reconstruction_loss_outputs,
                'kl_lat_state_0': kl_lat_state_0,
                'kl_lat_states': kl_lat_states,
                'kl_lat_parameters': kl_lat_parameters,
                'kl_lat_controls': kl_lat_controls,
                'lat_dim_state_0_populated': lat_dim_state_0_populated,
                'lat_dim_states_populated': lat_dim_states_populated,
                'lat_dim_parameters_populated': lat_dim_parameters_populated,
                'lat_dim_controls_populated': lat_dim_controls_populated,
                'time_odeint': time_odeint,
                'time_outputs': time_outputs,
                'ode_calls_forward': _ode_calls_forward,
                'rmse_state_0': rmse_state_0,
                'rmse_states': rmse_states,
                'rmse_outputs': rmse_outputs,
            }
            if return_model_outputs:
                # calculate KL loss per dimension
                kl_lat_state_0_per_dim = kullback_leibler(lat_state_0_mu, lat_state_0_logvar, per_dimension=True, reduce=False, time_series_aggregation_mode=None)
                kl_lat_parameters_per_dim = kullback_leibler(lat_parameters_mu, lat_parameters_logvar, per_dimension=True, reduce=False, time_series_aggregation_mode=None) if self.include_parameters else None
                kl_lat_controls_per_dim = kullback_leibler(lat_controls_mu, lat_controls_logvar, per_dimension=True, reduce=False, time_series_aggregation_mode=None) if self.include_controls else None
                if self.lat_ode_type == 'variance_constant':
                    kl_lat_states_per_dim = kullback_leibler(lat_states_mu, lat_state_0_logvar.unsqueeze(2).repeat(1,1,lat_states_mu.shape[2]), per_dimension=True, reduce=False, time_series_aggregation_mode=None)
                elif self.lat_ode_type == 'variance_dynamic':
                    kl_lat_states_per_dim = kullback_leibler(lat_states_mu, lat_states_logvar, per_dimension=True, reduce=False, time_series_aggregation_mode=None) 
                
                model_outputs={
                    'states_hat': states_hat,
                    'lat_state_0_mu': lat_state_0_mu,
                    'lat_state_0_logvar': lat_state_0_logvar,
                    'lat_parameters_mu': lat_parameters_mu,
                    'lat_parameters_logvar': lat_parameters_logvar,
                    'lat_controls_mu': lat_controls_mu,
                    'lat_controls_logvar': lat_controls_logvar,
                    'kl_lat_state_0_per_dim': kl_lat_state_0_per_dim,
                    'kl_lat_parameters_per_dim': kl_lat_parameters_per_dim,
                    'kl_lat_controls_per_dim': kl_lat_controls_per_dim,
                }
                if self.include_outputs:
                    model_outputs['outputs_hat'] = outputs_hat
                if self.lat_ode_type == 'variance_constant':
                    model_outputs['lat_states_mu'] = lat_states_mu
                    model_outputs['lat_state_0_logvar'] = lat_state_0_logvar
                    model_outputs['kl_lat_states_per_dim'] = kl_lat_states_per_dim
                elif self.lat_ode_type == 'variance_dynamic':
                    model_outputs['lat_states_mu'] = lat_states_mu
                    model_outputs['lat_states_logvar'] = lat_states_logvar
                    model_outputs['kl_lat_states_per_dim'] = kl_lat_states_per_dim
                elif self.lat_ode_type == 'vanilla':
                    model_outputs['lat_states'] = lat_states

            if test is True:
                # call value.item() for each value in return_value
                ret_val = dict({key: value.item() if type(value)==torch.Tensor else value for key, value in ret_val.items()})
            # detach model outputs from computational graph
            if return_model_outputs:
                _keys = list(model_outputs.keys())
                for key in _keys:
                    if model_outputs[key] is None:
                        model_outputs.pop(key)
                model_outputs = dict({key: value.cpu().detach().numpy() for key, value in model_outputs.items()})
            return ret_val if return_model_outputs is False else (ret_val, model_outputs)

    def get_progress_string(self, ret_vals_train, ret_vals_validation, ret_vals_test, pre_train):
        try:
            if pre_train is True:
                raise NotImplementedError
            else:
                _str = '\n \t[train/val/test] loss: {:.5f}/{:.5f}/{:.5f} | reconstruction_loss: {:.5f}/{:.5f}/{:.5f} | kl_loss: {:.5f}/{:.5f}/{:.5f}  \
                        \n \t| reconstruction_loss_state_0: {:.5f}/{:.5f}/{:.5f} | reconstruction_loss_states: {:.5f}/{:.5f}/{:.5f} | reconstruction_loss_outputs: {:.5f}/{:.5f}/{:.5f} | \
                        \n \
                        \n \t \t| rmse_state_0: {:.5f}/{:.5f}/{:.5f} | rmse_states: {:.5f}/{:.5f}/{:.5f} | rmse_outputs: {:.5f}/{:.5f}/{:.5f} \
                        \n \
                        \n \t| kl_lat_state_0: {:.5f}/{:.5f}/{:.5f} | kl_lat_states: {:.5f}/{:.5f}/{:.5f} \
                        \n \t \t \t| kl_lat_parameters: {:.5f}/{:.5f}/{:.5f} | kl_lat_controls: {:.5f}/{:.5f}/{:.5f} \
                        \n \t| pop. dims: state0: {}/{}/{}, states: {}/{}/{}, param: {}/{}/{}, ctrl: {}/{}/{}\
                        \n \t| ode_fun_count_forward: {}/{}/{} | ode_fun_count_backward: {}/-/-'.format(
                    ret_vals_train['loss'], ret_vals_validation['loss'], ret_vals_test['loss'],
                    ret_vals_train['reconstruction_loss'], ret_vals_validation['reconstruction_loss'], ret_vals_test['reconstruction_loss'],
                    ret_vals_train['kl_loss'], ret_vals_validation['kl_loss'], ret_vals_test['kl_loss'],
                    ret_vals_train['reconstruction_loss_state_0'], ret_vals_validation['reconstruction_loss_state_0'], ret_vals_test['reconstruction_loss_state_0'],
                    ret_vals_train['reconstruction_loss_states'], ret_vals_validation['reconstruction_loss_states'], ret_vals_test['reconstruction_loss_states'],
                    ret_vals_train['reconstruction_loss_outputs'], ret_vals_validation['reconstruction_loss_outputs'], ret_vals_test['reconstruction_loss_outputs'],
                    ret_vals_train['rmse_state_0'], ret_vals_validation['rmse_state_0'], ret_vals_test['rmse_state_0'],
                    ret_vals_train['rmse_states'], ret_vals_validation['rmse_states'], ret_vals_test['rmse_states'],
                    ret_vals_train['rmse_outputs'], ret_vals_validation['rmse_outputs'], ret_vals_test['rmse_outputs'],
                    ret_vals_train['kl_lat_state_0'], ret_vals_validation['kl_lat_state_0'], ret_vals_test['kl_lat_state_0'],
                    ret_vals_train['kl_lat_states'], ret_vals_validation['kl_lat_states'], ret_vals_test['kl_lat_states'],
                    ret_vals_train['kl_lat_parameters'], ret_vals_validation['kl_lat_parameters'], ret_vals_test['kl_lat_parameters'],
                    ret_vals_train['kl_lat_controls'], ret_vals_validation['kl_lat_controls'], ret_vals_test['kl_lat_controls'],
                    ret_vals_train['lat_dim_state_0_populated'], ret_vals_validation['lat_dim_state_0_populated'], ret_vals_test['lat_dim_state_0_populated'],
                    ret_vals_train['lat_dim_states_populated'], ret_vals_validation['lat_dim_states_populated'], ret_vals_test['lat_dim_states_populated'],
                    ret_vals_train['lat_dim_parameters_populated'], ret_vals_validation['lat_dim_parameters_populated'], ret_vals_test['lat_dim_parameters_populated'],
                    ret_vals_train['lat_dim_controls_populated'], ret_vals_validation['lat_dim_controls_populated'], ret_vals_test['lat_dim_controls_populated'],
                    ret_vals_train['ode_calls_forward'], ret_vals_validation['ode_calls_forward'], ret_vals_test['ode_calls_forward'],
                    ret_vals_train['ode_calls_backward'],
                )
        except Exception as e:
            logging.warning('error in get_progress_string: {}'.format(e))
            _str = 'error in get_progress_string'
        return _str

    def save(self, path: Path):
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.state_dict(), path)
        logging.debug('\t \t \tSaved model to {}'.format(path))
    
    def load(self, path: Path, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))
        logging.info('\tLoaded model from {}'.format(path))
