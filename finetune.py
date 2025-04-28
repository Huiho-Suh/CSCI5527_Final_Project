import subprocess
import optuna
import pandas as pd
import numpy as np
import os
from datetime import datetime
import shlex


"""_summary_

This script is used to optimize the hyperparameters of the model using Optuna.

"""

# Set parameters
num_trials = 40 # Number of trials with different hyperparameters
num_repeats = 1 # Number of times to repeat the for each trial
optimize_direction = "maximize" # maximize or minimize

EPOCH = 300

# Get date and time for the study directory
STUDY_DIR = f"ckpt/param_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{EPOCH}"

# Define the objective function for Optuna
def objective(trial):
    # Define the hyperparameter ranges
    # From "ACT Tuning Tips -- https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit"
    
    # kl_weight = trial.suggest_int('kl_weight', 10, 100, step=10) # 90
    kl_weight = 10
    # chunk_size = trial.suggest_int('chunk_size', 100, 160, step=10) # 160
    chunk_size = 50
    # batch_size = trial.suggest_int('batch_size', 16, 128, step=16) # 5
    batch_size = 16
    # hidden_dim = trial.suggest_int('hidden_dim', 256, 512, step=64)
    # dim_feedforward = trial.suggest_int('dim_feedforward', 1000, 4000, step=500)
    # learning_rate = 1e-5 * batch_size / 8
    learning_rate = 2e-5 
    # learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    
    mode = 'moe'
    # mode = trial.suggest_categorical('mode', ['latent', 'moe'])
    dim_feedforward_decoder = trial.suggest_int('dim_feedforward_decoder', 300, 1200, step=100)
    n_shared = trial.suggest_int('n_shared', 1, 2, step=1)
    n_experts = trial.suggest_int('n_experts', 5, 60, step=1)
    top_k = trial.suggest_int('top_k', 2, int(n_experts/2), step=1)

    # Set ckpt directory
    ckpt_dir = f"{STUDY_DIR}/{trial.number}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # ckpt_dir = "ckpt/test"
    
    # Create the command for training the model
    train_command = [
        "python3", "imitate_episodes.py",
        "--task_name", "sim_pick_place",
        "--ckpt_dir", str(ckpt_dir),  # Replace with your checkpoint directory
        "--policy_class", "ACT",
        "--kl_weight", str(kl_weight),
        "--chunk_size", str(chunk_size),
        "--hidden_dim", "512",
        "--batch_size", str(batch_size),
        "--dim_feedforward", "3200",
        "--num_epochs", str(EPOCH),
        "--lr", str(learning_rate),
        "--seed", "0",
        
        "--mode", str(mode),
        "--dim_feedforward_decoder", str(dim_feedforward_decoder),
        "--n_shared", str(n_shared),
        "--n_experts", str(n_experts),
        "--top_k", str(top_k),
    ]

    # Create the command for evaluating the model
    eval_command = train_command.copy()
    # eval_command.append("--temporal_agg") # ACT Tuning tips recommend removing temporal_agg and increase query frequency
    eval_command.append("--eval")

    evaluation_metrics = []
    for idx in range(num_repeats):
        # Run the training command
        subprocess.run(train_command)
        # Run the evaluation command
        result = subprocess.run(eval_command, capture_output=True, text=True)
        # Parse the evaluation result (success rate) - customized only for ACT model script
        output_lines = result.stdout.splitlines()
        success_rate = float(output_lines[-1])
        evaluation_metric = float(success_rate)
        
        # Get statistics and log them
        evaluation_metrics.append(evaluation_metric)
        mean = np.mean(evaluation_metrics)
        std  = np.std(evaluation_metrics)
        
        trial.set_user_attr('mean', mean)
        trial.set_user_attr('std', std)
        
        print(f"Trial {trial.number} | Repetition {idx} - Success Rate: {success_rate} | Mean: {mean} | Std: {std}")
        
    # Successive Halving
    trial.report(mean, step=1)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return mean # Optimizes using the mean of the success rates

def rank_trials(study): 
    # Create a DataFrame with the trial results
    trials_data = [{
        'Trial Number': trial.number,
        'kl_weight': trial.params['kl_weight'],
        'chunk_size': trial.params['chunk_size'],
        'hidden_dim': '512',
        'batch_size': trial.params['batch_size'],
        'dim_feedforward': '3200',
        'Success Rates (avg)': trial.value,
        'Standard Deviation': trial._user_attrs['std'],
        'Number of Repetitions': num_repeats,
        "mode": trial.params['mode'],
        "dim_feedforward_decoder": trial.params['dim_feedforward_decoder'],
        "n_shared": trial.params['n_shared'],
        "n_experts": trial.params['n_experts'],
        "top_k": trial.params['top_k'],
    } for trial in study.trials]

    df_trials = pd.DataFrame(trials_data)

    # Sort the trials by the Success Rates (descending order for maximizing)
    df_sorted = df_trials.sort_values(by='Success Rates (avg)', ascending=True)

    # Display the sorted DataFrame
    return df_sorted

# Set directory for Optuna study
os.makedirs(STUDY_DIR, exist_ok=True)

storage_name = f"sqlite:///{STUDY_DIR}/study.db"

# Run the optimization
study = optuna.create_study(direction=optimize_direction, 
                            study_name = 'finetune', 
                            storage=storage_name, 
                            load_if_exists=False,
                            pruner=optuna.pruners.SuccessiveHalvingPruner()
                            )  
study.optimize(objective, n_trials=num_trials)

# Rank the trials based on the Success Rates and display them
ranked_trials = rank_trials(study)

# Save the ranked trials to a CSV file
ranked_trials.to_csv(f"{STUDY_DIR}/ranked_trials.csv", index=False)
print(ranked_trials)
