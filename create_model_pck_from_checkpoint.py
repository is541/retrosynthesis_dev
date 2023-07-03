"""
Script to find minimum cost routes for PaRoutes benchmark
on a list of SMILES provided by the user.
"""
from __future__ import annotations
import argparse
import json
from embedding_model import GNNModel, FingerprintModel
import pickle
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="gnn_0629"

    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="epoch_51_checkpoint" 
    )
    
    args = parser.parse_args()
    
    model_input_folder = f'GraphRuns/{args.experiment_name}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    with open(f'{model_input_folder}/config.json', 'r') as f:
        config = json.load(f)
    
    
    if config["model_type"] == 'gnn':
        with open(f'{model_input_folder}/input_dim.pickle', 'rb') as f:
            input_dim_dict = pickle.load(f)
            gnn_input_dim = input_dim_dict['input_dim']
    #     gnn_input_dim = preprocessed_targets[0].node_features.shape[1]
        gnn_hidden_dim = config["hidden_dim"]
        gnn_output_dim = config["output_dim"]
    elif config["model_type"] == 'fingerprints':
        with open(f'{model_input_folder}/input_dim.pickle', 'rb') as f:
            input_dim_dict = pickle.load(f)
            fingerprint_input_dim = input_dim_dict['input_dim']
    #     fingerprint_input_dim = (preprocessed_targets[0].size()[0])
        fingerprint_hidden_dim = config["hidden_dim"]
        fingerprint_output_dim = config["output_dim"]
    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')

    if config["model_type"] == 'gnn':
        model = GNNModel(
            input_dim=gnn_input_dim, 
            hidden_dim=gnn_hidden_dim, 
            output_dim=gnn_output_dim).to(device)
        
    elif config["model_type"] == 'fingerprints':
        model = FingerprintModel(
            input_dim=fingerprint_input_dim, 
            hidden_dim=fingerprint_hidden_dim, 
            output_dim=fingerprint_output_dim).to(device)
    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')
        
        
    checkpoint_path = f'{model_input_folder}/{args.checkpoint_name}.pth'

    # Load the model state dict from the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.double()

    # Save model as pickle
    with open(f"{model_input_folder}/{args.checkpoint_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    
