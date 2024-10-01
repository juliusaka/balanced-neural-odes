## Stratified Heat Flow

### BNODE variance constant beta sweep

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_shf nn_model=bnode_shf dataset_name=StratifiedHeatFlowModel_v3_c-RROCS__n-1024 nn_model.training.beta_start_override=0.1,0.08,0.06,0.04,0.01 -m;

 ### Latent ODE

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_shf nn_model=bnode_shf dataset_name=StratifiedHeatFlowModel_v3_c-RROCS__n-1024 nn_model.network.lat_ode_type=vanilla

### BNODE variance dynamic beta sweep

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_shf nn_model=bnode_shf dataset_name=StratifiedHeatFlowModel_v3_c-RROCS__n-1024 nn_model.network.lat_ode_type=variance_dynamic nn_model.training.beta_start_override=0.01,0.08,0.06,0.04,0.01 -m;

## Steam Cycle

### BNODE variance constant beta sweep

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_steam nn_model=bnode_steam dataset_name=SteamCycle_01_v3_c-RROCS__n-1024 nn_model.training.beta_start_override=0.1,0.05,0.01,0.05,0.001 -m; 

### BNODE variance dynamic beta sweep

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_steam nn_model=bnode_steam dataset_name=SteamCycle_01_v3_c-RROCS__n-1024 nn_model.network.lat_ode_type=variance_dynamic nn_model.training.beta_start_override=0.1,0.05,0.01,0.05,0.001 -m; 
    

### Koopman variance constant

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_steam_koopman nn_model=bnode_steam_koopman_robust dataset_name=SteamCycle_01_v3_c-RROCS__n-1024 nn_model.network.koopman_mpc_mode=true nn_model.training.beta_start_override=0.1,0.05,0.01 -m; 


## Koopman Tu

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_tu_koopman nn_model=bnode_tu dataset_name=KoopmannTu_v2_s-R__n-1024 nn_model.network.koopman_mpc_mode=true nn_model.training.beta_start_override=0.1,0.05,0.01, 0.005, 0.001 nn_model.training.lr_start_override=0.001 -m ;
    

# NODE Steam

    python .\networks\neural_ode\trainer.py raise_exception=false n_workers_train_loader=4 mlflow_experiment_name=iclr_steam_node nn_model=node_steam dataset_name=SteamCycle_01_v3_c-RROCS__n-1024;
