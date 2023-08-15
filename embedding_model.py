"""Basic code for embeddings computation."""
from __future__ import annotations

import torch.nn as nn
import pickle
import json
from tqdm.auto import tqdm

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import math

from enum import Enum

import numpy as np

from torch_geometric.nn import GCNConv

import random
import torch
import torch.nn.functional as F


def num_heavy_atoms(mol):
    return Chem.rdchem.Mol.GetNumAtoms(mol, onlyExplicit=True)


def custom_global_max_pool(x):
    return torch.max(x, dim=0)[0]


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        targets,
        positive_samples,
        negative_samples,
        pos_weights=None,
    ):
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


class CustomDatasetReact(torch.utils.data.Dataset):
    def __init__(
        self,
        targets,
        positive_samples,
        negative_samples,
        num_mols_each_negative,
        cost_pos_react,
        cost_neg_react,
    ):
        self.targets = targets
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
        self.num_mols_each_negative = num_mols_each_negative
        self.cost_pos_react = cost_pos_react
        self.cost_neg_react = cost_neg_react

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target = self.targets[idx]
        positive = self.positive_samples[idx]
        negative = self.negative_samples[idx]
        num_mols_each_negative = self.num_mols_each_negative[idx]
        cost_pos_react = self.cost_pos_react[idx]
        cost_neg_react = self.cost_neg_react[idx]

        return (
            target,
            positive,
            negative,
            num_mols_each_negative,
            cost_pos_react,
            cost_neg_react,
        )


def collate_fn(data):
    (
        targets,
        positive_samples,
        negative_samples,
        pos_weights,
    ) = zip(*data)

    return (
        targets,
        positive_samples,
        negative_samples,
        pos_weights,
    )


def collate_fn_react(data):
    (
        targets,
        positive_samples,
        negative_samples,
        num_mols_each_negative,
        cost_pos_react,
        cost_neg_react,
    ) = zip(*data)

    return (
        targets,
        positive_samples,
        negative_samples,
        num_mols_each_negative,
        cost_pos_react,
        cost_neg_react,
    )


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0):
        super(GNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.conv1 = GCNConv(input_dim, hidden_dim, dtype=torch.double)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, dtype=torch.double)
        self.fc = nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Global max pooling (from node level to graph level embeddings)
        x = custom_global_max_pool(x)

        x = self.fc(x)
        return x


class FingerprintModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0):
        super(FingerprintModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=torch.double)
        self.fc2 = nn.Linear(hidden_dim, output_dim, dtype=torch.double)
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def gnn_preprocess_target_pos_negs(
    target_smiles,
    samples,
    featurizer,
    featurizer_dict,
):
    if featurizer_dict is None:
        compute_featurizer_dict = True
    else:
        compute_featurizer_dict = False
        
    target_feats = featurizer.featurize(Chem.MolFromSmiles(target_smiles))
    if compute_featurizer_dict:
        all_samples_set = set(samples["positive_samples"] + samples["negative_samples"])
        all_sample_list = list(all_samples_set)
        mol_to_featurize = [Chem.MolFromSmiles(smiles) for smiles in all_sample_list]
        mol_featurized = featurizer.featurize(mol_to_featurize)
        featurizer_dict = dict(zip(list(all_sample_list), mol_featurized))
    # print("All samples set: ", all_samples_set)
    # print("Positive samples: ", samples["positive_samples"])
    pos_feats = [
        featurizer_dict[positive_smiles]
        for positive_smiles in samples["positive_samples"]
    ]
    neg_feats = [
        featurizer_dict[negative_smiles]
        for negative_smiles in samples["negative_samples"]
    ]
    return target_feats[0], pos_feats, neg_feats


def gnn_preprocess_input_react(
    input_data,
    featurizer,
    featurizer_dict=None,
):
    targets = []
    positive_samples = []
    negative_samples = []
    num_mols_each_negative = []
    cost_pos_react = []
    cost_neg_react = []

    for target_smiles, samples in tqdm(input_data.items()):
        target_feats, pos_feats, neg_feats = gnn_preprocess_target_pos_negs(
            target_smiles, samples, featurizer, featurizer_dict
        )
        targets.append(target_feats)
        positive_samples.append(pos_feats)
        negative_samples.append(neg_feats)

        # Deal with num_mols_each_negative
        num_mols_each_negative.append(samples["num_mols_each_negative"])
        cost_pos_react.append(samples["cost_pos_react"])
        cost_neg_react.append(samples["cost_neg_react"])

    return CustomDatasetReact(
        targets=targets,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        num_mols_each_negative=num_mols_each_negative,
        cost_pos_react=cost_pos_react,
        cost_neg_react=cost_neg_react,
    )
    
def preprocess_input_format(
    input_data,
    pos_sampling=None,
):
    targets = []
    positive_samples = []
    negative_samples = []
    pos_weights = []

    for target_smiles, samples in tqdm(input_data.items()):
        targets.append(target_smiles)
        positive_samples.append(samples["positive_samples"])
        negative_samples.append(samples["negative_samples"])

        # Deal with pos_sampling
        if pos_sampling == "uniform":
            positive_weights = None
        elif pos_sampling == "prop_num_atoms":
            positive_weights = [
                num_heavy_atoms(Chem.MolFromSmiles(positive_smiles))
                for positive_smiles in samples["positive_samples"]
            ]
            positive_weights = torch.tensor(positive_weights, dtype=torch.double)
            # Normalize the tensor to sum up to 1
            positive_weights = positive_weights / positive_weights.sum()
        else:
            raise NotImplementedError(f"{pos_sampling}")

        pos_weights.append(positive_weights)

    return CustomDataset(
        targets=targets,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        pos_weights=pos_weights,
    )


def gnn_preprocess_input_staticNegatives(
    input_data,
    featurizer,
    featurizer_dict=None,
    pos_sampling=None,
):
    targets = []
    positive_samples = []
    negative_samples = []
    pos_weights = []

    for target_smiles, samples in tqdm(input_data.items()):
        target_feats, pos_feats, neg_feats = gnn_preprocess_target_pos_negs(
            target_smiles, samples, featurizer, featurizer_dict
        )
        targets.append(target_feats)
        positive_samples.append(pos_feats)
        negative_samples.append(neg_feats)

        # Deal with pos_sampling
        if pos_sampling == "uniform":
            positive_weights = None
        elif pos_sampling == "prop_num_atoms":
            positive_weights = [
                num_heavy_atoms(Chem.MolFromSmiles(positive_smiles))
                for positive_smiles in samples["positive_samples"]
            ]
            positive_weights = torch.tensor(positive_weights, dtype=torch.double)
            # Normalize the tensor to sum up to 1
            positive_weights = positive_weights / positive_weights.sum()
        else:
            raise NotImplementedError(f"{pos_sampling}")

        pos_weights.append(positive_weights)

    return CustomDataset(
        targets=targets,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        pos_weights=pos_weights,
    )


def fingerprint_vect_from_smiles(mol_smiles):
    return AllChem.GetMorganFingerprintAsBitVect(
        AllChem.MolFromSmiles(mol_smiles), radius=3
    )


def fingerprint_preprocess_target_pos_negs(target_smiles, samples, fingerprints_dict):
    if fingerprints_dict is None:
        compute_fingerprints_dict = True
    else:
        compute_fingerprints_dict = False
    # print(target_smiles)
    target_feats = fingerprint_vect_from_smiles(target_smiles)
    # print(target_feats[0])
    if compute_fingerprints_dict:
        all_samples_set = set(samples["positive_samples"] + samples["negative_samples"])
        mol_to_compute_fnps = list(all_samples_set)
        # mol_to_compute_fnps = [Chem.MolFromSmiles(smiles) for smiles in all_samples_set]
        mol_fingerprints = list(map(fingerprint_vect_from_smiles, mol_to_compute_fnps))
        fingerprints_dict = dict(zip(mol_to_compute_fnps, mol_fingerprints))
    pos_feats = [
        fingerprints_dict[positive_smiles]
        for positive_smiles in samples["positive_samples"]
    ]
    neg_feats = [
        fingerprints_dict[negative_smiles]
        for negative_smiles in samples["negative_samples"]
    ]

    return target_feats, pos_feats, neg_feats


def fingerprint_preprocess_input_react(input_data, fingerprints_dict=None):
    targets = []
    positive_samples = []
    negative_samples = []
    num_mols_each_negative = []
    cost_pos_react = []
    cost_neg_react = []

    for target_smiles, samples in tqdm(input_data.items()):
        target_feats, pos_feats, neg_feats = fingerprint_preprocess_target_pos_negs(
            target_smiles=target_smiles,
            samples=samples,
            fingerprints_dict=fingerprints_dict,
        )
        targets.append(torch.tensor(target_feats, dtype=torch.double))
        positive_samples.append(torch.tensor(pos_feats, dtype=torch.double))
        negative_samples.append(torch.tensor(neg_feats, dtype=torch.double))

        # Deal with num_mols_each_negative
        num_mols_each_negative.append(samples["num_mols_each_negative"])
        cost_pos_react.append(samples["cost_pos_react"])
        cost_neg_react.append(samples["cost_neg_react"])

    return CustomDatasetReact(
        targets=targets,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        num_mols_each_negative=num_mols_each_negative,
        cost_pos_react=cost_pos_react,
        cost_neg_react=cost_neg_react,
    )


def fingerprint_preprocess_input_staticNegatives(input_data, fingerprints_dict=None, pos_sampling=None):
    targets = []
    positive_samples = []
    negative_samples = []
    pos_weights = []

    for target_smiles, samples in tqdm(input_data.items()):
        target_feats, pos_feats, neg_feats = fingerprint_preprocess_target_pos_negs(
            target_smiles=target_smiles,
            samples=samples,
            fingerprints_dict=fingerprints_dict,
        )

        # Deal with positive weights
        if pos_sampling == "uniform":
            positive_weights = None
        elif pos_sampling == "prop_num_atoms":
            positive_weights = [
                num_heavy_atoms(Chem.MolFromSmiles(positive_smiles))
                for positive_smiles in samples["positive_samples"]
            ]
            positive_weights = torch.tensor(positive_weights, dtype=torch.double)
            # Normalize the tensor to sum up to 1
            positive_weights = positive_weights / positive_weights.sum()
        else:
            raise NotImplementedError(f"{pos_sampling}")

        targets.append(torch.tensor(target_feats, dtype=torch.double))
        positive_samples.append(torch.tensor(pos_feats, dtype=torch.double))
        negative_samples.append(torch.tensor(neg_feats, dtype=torch.double))
        pos_weights.append(positive_weights)

    return CustomDataset(
        targets=targets,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        pos_weights=pos_weights,
    )


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


class ContrastiveReact(nn.Module):
    def __init__(self, temperature, device, multiplicative_factor):
        super(ContrastiveReact, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.device = device
        self.multiplicative_factor = multiplicative_factor

    def cos_dist(self, a, b):
        return 1.0 - self.cos_sim(a, b)

    def forward(self, embeddings, purch_embeddings):
        # Initialize the tensor to store sample losses
        sample_losses = torch.zeros(len(embeddings), device=self.device)

        for i, (
            target_emb,
            positive_embs,
            negative_embs,
            target_num_mols_each_negative,
            target_cost_pos_react,
            target_cost_neg_react
        ) in enumerate(embeddings):
            # target_num_mols_each_negative = num_mols_each_negative[i]

            # Compute positive value
            nr_positives = positive_embs.size(0)
            if nr_positives == 0:
                positive_value = torch.tensor(
                    0.0, device=self.device, dtype=torch.double
                )
            else:
                # Compute the minimum cosine distance between each row of positive_embs and all rows of purch_embeddings
                positive_values = torch.min(
                    self.cos_dist(
                        positive_embs.unsqueeze(1), purch_embeddings.unsqueeze(0)
                    ),
                    dim=1,
                ).values * self.multiplicative_factor

                # Compute the sum of all elements in positive_values
                positive_value = positive_values.sum() 
                positive_value += target_cost_pos_react
                positive_value /= self.temperature

            # Split negative_embs into sub-tensors based on target_num_mols_each_negative
            target_num_mols_each_negative = [
                int(num) for num in target_num_mols_each_negative
            ]
            negative_subtensors = torch.split(
                negative_embs, target_num_mols_each_negative
            )
            
            assert len(negative_subtensors) == len(target_cost_neg_react), \
                f"Dimensions of negative_subtensors ({len(negative_subtensors)}) and target_cost_neg_react ({len(target_cost_neg_react)}) are inconsistent."

            # Compute negative_values using each tensor in negative_subtensors
            negative_values = []
            for j, tensor in enumerate(negative_subtensors):
                negative_values.append(
                    torch.min(
                        self.cos_dist(
                            tensor.unsqueeze(1), purch_embeddings.unsqueeze(0)
                        ),
                        dim=1,
                    ).values.sum() * self.multiplicative_factor
                    + target_cost_neg_react[j]  # Add target_cost_neg_react
                )

                # with open("react_loss_values.csv", mode='a', newline='') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([i, positive_values.sum().item(), target_cost_pos_react.item(),
                #                     positive_values.sum().item() + target_cost_pos_react.item(),
                #                     positive_value.item(), j, negative_values[j].item(),
                #                     target_cost_neg_react[j].item(), 
                #                     (target_cost_neg_react[j].item() + negative_values[j].item()),
                #                     (target_cost_neg_react[j].item() + negative_values[j].item()) / self.temperature,
                #                     math.exp(-positive_value), 
                #                     math.exp(-(target_cost_neg_react[j].item() + negative_values[j].item()) / self.temperature), 
                #                     ])

                
                
            negative_values = torch.tensor(negative_values)

            negative_values /= self.temperature

            # Compute sample loss
            # # OLD implementation
            numerator = torch.exp(-positive_value) 
            denominator = torch.sum(torch.exp(-negative_values))
            sample_loss = -torch.log(numerator / (numerator + denominator))
            # # NEW implementation
            # print(positive_value)
            # all_similarities = torch.cat([positive_value.unsqueeze(0), negative_values], dim=0)
            # sample_loss = -positive_value + torch.logsumexp(
            #     all_similarities, dim=0, keepdims=True
            # )
            # breakpoint()

            # Store sample loss in the tensor
            # breakpoint()
            sample_losses[i] = sample_loss
            
            # with open("react_loss_values_agg.csv", mode='a', newline='') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([i, numerator, denominator, sample_loss
            #                     ])

        return torch.mean(sample_losses)


class NTXentLoss(nn.Module):
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.device = device

    def forward(self, embeddings):
        # Initialize the tensor to store sample losses
        sample_losses = torch.zeros(len(embeddings), device=self.device)

        for i, (target_emb, positive_embs, negative_embs, pos_weights) in enumerate(
            embeddings
        ):
            # Compute positive similarity
            nr_positives = positive_embs.size(0)
            if nr_positives == 0:
                positive_similarity = torch.tensor(
                    [0.0], device=self.device, dtype=torch.double
                )
            else:
                # Sample one positive
                if pos_weights is None:
                    row_index = torch.randint(0, nr_positives, (1,), device=self.device)
                    positive_emb = positive_embs[row_index]
                else:
                    assert len(pos_weights) == nr_positives
                    row_index = torch.multinomial(pos_weights.to(self.device), 1)
                    positive_emb = positive_embs[row_index]

                positive_similarity = self.cos_sim(target_emb, positive_emb)
                positive_similarity /= self.temperature

            # Compute negative similarity
            negative_similarity = self.cos_sim(target_emb, negative_embs)
            negative_similarity /= self.temperature

            # Compute sample loss
            # # OLD implementation
            # numerator = torch.exp(positive_similarity)
            # denominator = torch.sum(torch.exp(negative_similarity))
            # sample_loss = -torch.log(numerator / (numerator + denominator))
            # NEW implementation
            all_similarities = torch.cat(
                [positive_similarity, negative_similarity], dim=0
            )
            sample_loss = -positive_similarity + torch.logsumexp(
                all_similarities, dim=0, keepdims=True
            )

            # Store sample loss in the tensor
            sample_losses[i] = sample_loss

        return torch.mean(sample_losses)


def compute_actual_embeddings(device, model_type, model, target, positives, negatives):
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
                        torch.tensor(example.node_features, dtype=torch.double).to(
                            device
                        ),
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

    elif model_type == "fingerprints":
        # target_embedding = model(torch.tensor(target, dtype=torch.double).to(device))
        target_embedding = model(torch.tensor(target, dtype=torch.double).to(device))
        if len(positives) == 0:
            positive_samples_embeddings = torch.empty((0, target_embedding.size(0))).to(
                device
            )
        else:
            positive_samples_embeddings = torch.stack(
                [
                    model(
                        torch.tensor(example, dtype=torch.double).to(device)
                    )
                    for example in positives
                ],
                dim=0,
            )
            
        negative_samples_embeddings = torch.stack(
                [
                    model(
                        torch.tensor(example, dtype=torch.double).to(device)
                    )
                    for example in negatives
                ],
                dim=0,
            )    
    else:
        raise NotImplementedError(f"Model type {model_type}")

    return target_embedding, positive_samples_embeddings, negative_samples_embeddings


def compute_embeddings_react(device, model_type, model, batch_data):
    targets = []
    positive_samples = []
    negative_samples = []
    num_mols_each_negative = []
    cost_pos_react = []
    cost_neg_react = []

    (
        batch_targets,
        batch_positive_samples,
        batch_negative_samples,
        batch_num_mols_each_negative,
        batch_cost_pos_react,
        batch_cost_neg_react,
    ) = batch_data

    for (
        target_i,
        positives_i,
        negatives_i,
        num_mols_each_negative_i,
        cost_pos_react_i,
        cost_neg_react_i,
    ) in zip(
        batch_targets,
        batch_positive_samples,
        batch_negative_samples,
        batch_num_mols_each_negative,
        batch_cost_pos_react,
        batch_cost_neg_react,
    ):
        (
            target_embedding,
            positive_samples_embeddings,
            negative_samples_embeddings,
        ) = compute_actual_embeddings(
            device=device,
            model_type=model_type,
            model=model,
            target=target_i,
            positives=positives_i,
            negatives=negatives_i,
        )
        targets.append(target_embedding)
        positive_samples.append(positive_samples_embeddings)
        negative_samples.append(negative_samples_embeddings)

        num_mols_each_negative.append(num_mols_each_negative_i)
        cost_pos_react.append(cost_pos_react_i)
        cost_neg_react.append(cost_neg_react_i)

    # return embeddings
    return CustomDatasetReact(
        targets=targets,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        num_mols_each_negative=num_mols_each_negative,
        cost_pos_react=cost_pos_react,
        cost_neg_react=cost_neg_react,
    )
    
def preprocess_and_compute_embeddings(
    device,
    model_type,
    model,
    batch_data,
    featur_creator, 
    not_in_route_sample_size
):
    targets = []
    positive_samples = []
    negative_samples = []
    pos_weights = []
    
    (
        batch_targets,
        batch_positive_samples,
        batch_negative_samples,
        batch_pos_weights,
    ) = batch_data

    for (
        target_i,
        positives_i,
        negatives_i,
        pos_weights_i,
    ) in zip(
        batch_targets,
        batch_positive_samples,
        batch_negative_samples,
        batch_pos_weights,
    ):
        # Prepare data
        # - Sample the negatives
        negatives_i_sample = random.sample(
            negatives_i, not_in_route_sample_size
        )
        samples = {
            "positive_samples": positives_i,
            "negative_samples": negatives_i_sample,
        }
        
        # - Featurize all
        if model_type == "gnn":
            featurizer = featur_creator["featurizer"]
            featurizer_dict = featur_creator["featurizer_dict"]
            
            target_feats_i, pos_feats_i, neg_feats_i = gnn_preprocess_target_pos_negs(
                target_i, samples, featurizer, featurizer_dict
            )
        elif model_type == "fingerprints":
            fingerprints_dict = featur_creator["fingerprints_dict"]
            
            target_feats_i, pos_feats_i, neg_feats_i = fingerprint_preprocess_target_pos_negs(
                target_smiles=target_i,
                samples=samples,
                fingerprints_dict=fingerprints_dict,
            )
        else:
            raise NotImplementedError(f"Model type {model_type}")
        
        
        (
            target_embedding,
            positive_samples_embeddings,
            negative_samples_embeddings,
        ) = compute_actual_embeddings(
            device=device,
            model_type=model_type,
            model=model,
            target=target_feats_i,
            positives=pos_feats_i,
            negatives=neg_feats_i,
        )
        targets.append(target_embedding)
        positive_samples.append(positive_samples_embeddings)
        negative_samples.append(negative_samples_embeddings)

        pos_weights.append(pos_weights_i)

    # return embeddings
    return CustomDataset(
        targets=targets,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        pos_weights=pos_weights,
    )




# def compute_embeddings(
#     device,
#     model_type,
#     model,
#     batch_data,
# ):
#     targets = []
#     positive_samples = []
#     negative_samples = []
#     pos_weights = []
    
#     (
#         batch_targets,
#         batch_positive_samples,
#         batch_negative_samples,
#         batch_pos_weights,
#     ) = batch_data

#     for (
#         target_i,
#         positives_i,
#         negatives_i,
#         pos_weights_i,
#     ) in zip(
#         batch_targets,
#         batch_positive_samples,
#         batch_negative_samples,
#         batch_pos_weights,
#     ):
#     # for (
#     #     target_i,
#     #     positives_i,
#     #     negatives_i,
#     #     pos_weights_i,
#     # ) in batch_data:
#         (
#             target_embedding,
#             positive_samples_embeddings,
#             negative_samples_embeddings,
#         ) = compute_actual_embeddings(
#             device=device,
#             model_type=model_type,
#             model=model,
#             target=target_i,
#             positives=positives_i,
#             negatives=negatives_i,
#         )
#         targets.append(target_embedding)
#         positive_samples.append(positive_samples_embeddings)
#         negative_samples.append(negative_samples_embeddings)

#         pos_weights.append(pos_weights_i)

#     # return embeddings
#     return CustomDataset(
#         targets=targets,
#         positive_samples=positive_samples,
#         negative_samples=negative_samples,
#         pos_weights=pos_weights,
#     )


def load_embedding_model_from_pickle(experiment_name, model_name="model_min_val"):
    emb_model_input_folder = f"GraphRuns/{experiment_name}"

    model_path = f"{emb_model_input_folder}/{model_name}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(f"{emb_model_input_folder}/config.json", "r") as f:
        config = json.load(f)

    return model, config

def load_embedding_model_from_checkpoint(experiment_name, device, checkpoint_name="model_min_val_checkpoint"):
    emb_model_input_folder = f"GraphRuns/{experiment_name}"
    
    with open(f"{emb_model_input_folder}/config.json", "r") as f:
        config = json.load(f)
    
    if config["model_type"] == 'gnn':
        with open(f'{emb_model_input_folder}/input_dim.pickle', 'rb') as f:
            input_dim_dict = pickle.load(f)
            gnn_input_dim = input_dim_dict['input_dim']
        gnn_hidden_dim = config["hidden_dim"]
        gnn_output_dim = config["output_dim"]
        gnn_dropout_prob = config.get('dropout_prob', 0)
    elif config["model_type"] == 'fingerprints':
        with open(f'{emb_model_input_folder}/input_dim.pickle', 'rb') as f:
            input_dim_dict = pickle.load(f)
            fingerprint_input_dim = input_dim_dict['input_dim']
        fingerprint_hidden_dim = config["hidden_dim"]
        fingerprint_output_dim = config["output_dim"]
        fingerprint_dropout_prob = config.get('dropout_prob', 0)
    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')

    if config["model_type"] == 'gnn':
        model = GNNModel(
            input_dim=gnn_input_dim, 
            hidden_dim=gnn_hidden_dim, 
            output_dim=gnn_output_dim,
            dropout_prob=gnn_dropout_prob).to(device)
        model.double()
        
    elif config["model_type"] == 'fingerprints':
        model = FingerprintModel(
            input_dim=fingerprint_input_dim, 
            hidden_dim=fingerprint_hidden_dim, 
            output_dim=fingerprint_output_dim,
            dropout_prob=fingerprint_dropout_prob).to(device)
    else:
        raise NotImplementedError(f'Model type {config["model_type"]}')
        
        
    checkpoint_path = f'{emb_model_input_folder}/{checkpoint_name}.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the model state dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model {config['model_type']} from epoch {checkpoint['epoch']}")

    # with open(f"{emb_model_input_folder}/config.json", "r") as f:
    #     config = json.load(f)

    return model, config