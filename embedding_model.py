"""Basic code for embeddings computation."""
from __future__ import annotations

import torch.nn as nn
import pickle
import json

import torch

from enum import Enum

import numpy as np
# from rdkit.Chem import DataStructs, AllChem

# from syntheseus.search.graph.and_or import OrNode
# from syntheseus.search.node_evaluation.base import NoCacheNodeEvaluator
# from syntheseus.search.mol_inventory import ExplicitMolInventory

import torch
import torch.nn.functional as F


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x
    
class FingerprintModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FingerprintModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.double)
        self.fc2 = nn.Linear(hidden_dim, output_dim, dtype=torch.double)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def load_embedding_model(experiment_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    emb_model_input_folder = f'GraphRuns/{experiment_name}'
    
    model_path = f'{emb_model_input_folder}/model_min_val.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    with open(f'{emb_model_input_folder}/config.json', 'r') as f:
        config = json.load(f)
        
    return device, model, config




    




