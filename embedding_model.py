"""Basic code for embeddings computation."""
from __future__ import annotations

import torch.nn as nn
import pickle
import json
from tqdm.auto import tqdm

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from enum import Enum

import numpy as np

from torch_geometric.nn import GCNConv


import torch
import torch.nn.functional as F


def custom_global_max_pool(x):
    return torch.max(x, dim=0)[0]


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv1 = GCNConv(input_dim, hidden_dim, dtype=torch.double)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, dtype=torch.double)
        self.fc = nn.Linear(hidden_dim, output_dim, dtype=torch.double)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global max pooling (from node level to graph level embeddings)
        x = custom_global_max_pool(x)

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


def gnn_preprocess_input(input_data, featurizer, purch_featurizer_dict):
    targets = []
    positive_samples = []
    negative_samples = []

    for target_smiles, samples in tqdm(input_data.items()):
        target_feats = featurizer.featurize(Chem.MolFromSmiles(target_smiles))
        pos_feats = [
            purch_featurizer_dict[positive_smiles]
            for positive_smiles in samples["positive_samples"]
        ]
        neg_feats = [
            purch_featurizer_dict[negative_smiles]
            for negative_smiles in samples["negative_samples"]
        ]
        targets.append(target_feats[0])
        positive_samples.append(pos_feats)
        negative_samples.append(neg_feats)
        return targets, positive_samples, negative_samples


def fingerprint_vect_from_smiles(mol_smiles):
    return AllChem.GetMorganFingerprintAsBitVect(
        AllChem.MolFromSmiles(mol_smiles), radius=3
    )


def fingerprint_preprocess_input(input_data, purch_fingerprints_dict):
    targets = []
    positive_samples = []
    negative_samples = []

    for target_smiles, samples in tqdm(input_data.items()):
        #         target_feats = fingerprint_from_smiles(Chem.MolFromSmiles(target_smiles))
        #         pos_mols = [Chem.MolFromSmiles(positive_smiles) for positive_smiles in samples['positive_samples']]
        #         neg_mols = [Chem.MolFromSmiles(negative_smiles) for negative_smiles in samples['negative_samples']]
        target_feats = fingerprint_vect_from_smiles(target_smiles)
        #         pos_feats = list(map(fingerprint_vect_from_smiles, samples['positive_samples']))
        #         neg_feats = list(map(fingerprint_vect_from_smiles, samples['negative_samples']))
        pos_feats = [
            purch_fingerprints_dict[positive_smiles]
            for positive_smiles in samples["positive_samples"]
        ]
        neg_feats = [
            purch_fingerprints_dict[negative_smiles]
            for negative_smiles in samples["negative_samples"]
        ]

        #         targets.append(target_feats[0])
        #         positive_samples.append(pos_feats)
        #         negative_samples.append(neg_feats)
        targets.append(torch.tensor(target_feats, dtype=torch.double))
        positive_samples.append(torch.tensor(pos_feats, dtype=torch.double))
        negative_samples.append(torch.tensor(neg_feats, dtype=torch.double))

    return targets, positive_samples, negative_samples


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, targets, positive_samples, negative_samples, pos_weights=None):
        self.targets = targets
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.pos_weights = pos_weights

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target = self.targets[idx]
        positive = self.positive_samples[idx]
        negative = self.negative_samples[idx]
        if self.pos_weights is None:
            pos_weight = None
        else:
            pos_weight = self.pos_weights[idx]

        return target, positive, negative, pos_weight


def collate_fn(data):
    targets, positive_samples, negative_samples, pos_weights = zip(*data)

    return targets, positive_samples, negative_samples, pos_weights


# class SampleData:
#     def __init__(self, target, positive_samples, negative_samples, pos_weights):
#         self.target = target
#         self.positive_samples = positive_samples
#         self.negative_samples = negative_samples
#         self.pos_weights = pos_weights


# class NTXentLoss(nn.Module):
#     def __init__(self, temperature):
#         super(NTXentLoss, self).__init__()
#         self.temperature = temperature
#         self.cos_sim = nn.CosineSimilarity(dim=-1)

#     def forward(self, embeddings):
#         sample_losses = []
#         # for single_sample_embeddings in embeddings:
#         for target_emb, positive_embs, negative_embs, pos_weights in embeddings:
#             # target_emb = single_sample_embeddings.target
#             # positive_embs = single_sample_embeddings.positive_samples
#             # negative_embs = single_sample_embeddings.negative_samples
#             # pos_weights = single_sample_embeddings.pos_weights

#             # Positive similarity
#             nr_positives = positive_embs.size(0)
#             if nr_positives == 0:
#                 positive_similarity = torch.tensor(0.0)
#             else:
#                 # Sample one podsitive
#                 if pos_weights is None:
#                     # Randomly select a positive sample
#                     row_index = torch.randint(0, nr_positives, (1,))
#                     positive_emb = torch.index_select(
#                         positive_embs, dim=0, index=row_index
#                     )

#                 else:
#                     assert len(pos_weights) == nr_positives
#                     row_index = torch.multinomial(pos_weights, 1)
#                     positive_emb = torch.index_select(
#                         positive_embs, dim=0, index=row_index
#                     )

#                 positive_similarity = self.cos_sim(target_emb, positive_emb)
#                 positive_similarity /= self.temperature

#             # Negative similarity
#             negative_similarity = self.cos_sim(target_emb, negative_embs)
#             negative_similarity /= self.temperature

#             # Old implementation
#             numerator = torch.exp(positive_similarity)
#             denominator = torch.sum(torch.exp(negative_similarity))
#             sample_loss = -torch.log(numerator / (numerator + denominator))
#             # End Old implementation
#             #             # New implementation
#             #             all_similarities = torch.cat([positive_similarity, negative_similarity], dim=0)
#             #             sample_loss = -positive_similarity + torch.logsumexp(all_similarities, dim=0, keepdims=True)
#             #             # End New implementation

#             sample_losses = sample_losses + [sample_loss]

#         return sum(sample_losses) / len(sample_losses)

class NTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.device = device

    def forward(self, embeddings):
        # Initialize the tensor to store sample losses
        sample_losses = torch.zeros(len(embeddings), device=self.device)

        for i, (target_emb, positive_embs, negative_embs, pos_weights) in enumerate(embeddings):
            # Compute positive similarity
            nr_positives = positive_embs.size(0)
            if nr_positives == 0:
                positive_similarity = torch.tensor(0.0, device=self.device)
            else:
                # Sample one positive
                if pos_weights is None:
                    row_index = torch.randint(0, nr_positives, (1,), device=self.device)
                    positive_emb = positive_embs[row_index]
                else:
                    assert len(pos_weights) == nr_positives
                    row_index = torch.multinomial(pos_weights, 1, device=self.device)
                    positive_emb = positive_embs[row_index]

                positive_similarity = self.cos_sim(target_emb, positive_emb)
                positive_similarity /= self.temperature

            # Compute negative similarity
            negative_similarity = self.cos_sim(target_emb, negative_embs)
            negative_similarity /= self.temperature

            # Compute sample loss
            # OLD implementation
            numerator = torch.exp(positive_similarity)
            denominator = torch.sum(torch.exp(negative_similarity))
            sample_loss = -torch.log(numerator / (numerator + denominator))
            # # NEW implementation
            # all_similarities = torch.cat([positive_similarity, negative_similarity], dim=0)
            # sample_loss = -positive_similarity + torch.logsumexp(all_similarities, dim=0, keepdims=True)

            # Store sample loss in the tensor
            sample_losses[i] = sample_loss

        return torch.mean(sample_losses)



def compute_embeddings_and_loss(
    device,
    model_type,
    model,
    batch_targets,
    batch_positive_samples,
    batch_negative_samples,
    loss_fn,
    pos_sampling,
    fingerprint_num_atoms_dict=None,
):
    # embeddings = []
    targets = []
    positive_samples = []
    negative_samples = []
    positive_weights = []
    for i in range(len(batch_targets)):
        target = batch_targets[i]
        positives = batch_positive_samples[i]
        negatives = batch_negative_samples[i]

        if model_type == "gnn":
            target_node_features = torch.tensor(
                target.node_features, dtype=torch.double
            ).to(device)
            target_edge_index = torch.tensor(target.edge_index, dtype=torch.long).to(device)
            target_embedding = model(target_node_features, target_edge_index)

            if len(positives) == 0:
                positive_samples_embeddings = torch.empty((0, target_embedding.size(0)))
            else:
                positive_samples_embeddings = torch.stack(
                    [
                        model(
                            torch.tensor(example.node_features, dtype=torch.double).to(device),
                            torch.tensor(example.edge_index, dtype=torch.long).to(device),
                        )
                        for example in positives
                    ],
                    dim=0,
                )

            negative_samples_embeddings = torch.stack(
                [
                    model(
                        torch.tensor(example.node_features, dtype=torch.double).to(device),
                        torch.tensor(example.edge_index, dtype=torch.long).to(device),
                    )
                    for example in negatives
                ],
                dim=0,
            )
            
            if pos_sampling == "uniform":
                pos_weights = None
            elif pos_sampling == "prop_num_atoms":
                pos_weights = []
                for positive in positives:
                    pos_weights.append(positive.node_features.shape[0])
                    # pos_weights.append(positive.node_features.size(0))
                pos_weights = torch.tensor(pos_weights, dtype=torch.double)
                # Normalize the tensor to sum up to 1
                pos_weights = pos_weights / pos_weights.sum()
            else:
                raise NotImplementedError(f'{pos_sampling}')
            
            targets.append(target_embedding)
            positive_samples.append(positive_samples_embeddings)
            negative_samples.append(negative_samples_embeddings)
            positive_weights.append(pos_weights)

            # embeddings.append(
            #     SampleData(
            #         target=target_embedding,
            #         positive_samples=positive_samples_embeddings,
            #         negative_samples=negative_samples_embeddings,
            #         pos_weights=pos_weights
            #     )
            # )


        elif model_type == "fingerprints":
            target_embedding = model(target.to(device))
            if len(positives) == 0:
                positive_samples_embeddings = torch.empty((0, target_embedding.size(0))).to(device)
            else:
                positive_samples_embeddings = model(positives.to(device))
            negative_samples_embeddings = model(negatives.to(device))

            if pos_sampling == "uniform":
                pos_weights = None
            elif pos_sampling == "prop_num_atoms":
                print(positives[0])
                print(list(fingerprint_num_atoms_dict.keys())[0])
                pos_weights = [
                    fingerprint_num_atoms_dict[positive] for positive in positives
                ]
                pos_weights = torch.tensor(pos_weights, dtype=torch.double)
                # Normalize the tensor to sum up to 1
                pos_weights = pos_weights / pos_weights.sum()
            else:
                raise NotImplementedError(f'{pos_sampling}')
            
            targets.append(target_embedding)
            positive_samples.append(positive_samples_embeddings)
            negative_samples.append(negative_samples_embeddings)
            positive_weights.append(pos_weights)
            
            # embeddings.append(
            #     SampleData(
            #         target=target_embedding,
            #         positive_samples=positive_samples_embeddings,
            #         negative_samples=negative_samples_embeddings,
            #         pos_weights=pos_weights
            #     )
            # )
        #             embeddings = embeddings + [SampleData(target=target_embedding, positive_samples=positive_samples_embeddings, negative_samples=negative_samples_embeddings)]
        else:
            raise NotImplementedError(f'Model type {model_type}')

    embeddings_dataset = CustomDataset(
        targets,
        positive_samples,
        negative_samples,
        positive_weights
    )
    # Compute loss for the batch
    # loss = loss_fn(embeddings)
    loss = loss_fn(embeddings_dataset)

    # return embeddings, loss
    return embeddings_dataset, loss


def load_embedding_model(experiment_name, model_name="model_min_val"):
    emb_model_input_folder = f"GraphRuns/{experiment_name}"

    model_path = f"{emb_model_input_folder}/{model_name}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(f"{emb_model_input_folder}/config.json", "r") as f:
        config = json.load(f)

    return model, config
