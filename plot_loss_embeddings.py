"""
Script to train neural network to get molecules embeddings
"""
from __future__ import annotations

import argparse
# import os
# import pickle
import json
import pandas as pd
# import random
# from torch.utils.data import DataLoader
# import torch
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# import pandas as pd
import time

# from tqdm.auto import tqdm
# from paroutes import PaRoutesInventory, get_target_smiles
# from embedding_model import (
#     preprocess_input_format,
#     preprocess_and_compute_embeddings,
#     CustomDataset,
#     collate_fn,
#     # SampleData,
#     fingerprint_vect_from_smiles,
#     GNNModel,
#     FingerprintModel,
#     NTXentLoss,
#     num_heavy_atoms
# )
# from paroutes import PaRoutesInventory
# from torch.utils.data import Subset
# from sklearn.model_selection import train_test_split
import plotly.express as px
# from rdkit import Chem
# import deepchem as dc



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, required=True, help="JSON file with configurations."
    )

    args = parser.parse_args()

    with open(f"{args.config_file}.json", "r") as f:
        config = json.load(f)

    experiment_name = config["experiment_name"]
    checkpoint_folder = f"GraphRuns/{experiment_name}/"

    # Read results data
    epoch_loss = pd.read_csv(f"{checkpoint_folder}/train_val_loss.csv")
    # input_file_distances = f'Runs/{config["run_id"]}/targ_to_purch_distances.pickle'

    
    # STEP 6: Plot

    # fig = px.line(x=epoch_loss['Epoch'], y=epoch_loss['TrainLoss'], title="Train loss")
    # fig.update_layout(width=1000, height=600, showlegend=False)
    # fig.write_image(f"{checkpoint_folder}/Train_loss.pdf")
    # fig.show()

    # Create a new figure with two lines    
    fig = px.line()

    # Add the TrainLoss line to the figure
    fig.add_scatter(x=epoch_loss["Epoch"], y=epoch_loss["TrainLoss"], name="Train Loss")

    # Add the ValLoss line to the figure
    fig.add_scatter(
        x=epoch_loss["Epoch"], y=epoch_loss["ValLoss"], name="Validation Loss"
    )

    # Set the title of the figure
    fig.update_layout(title="Train and Validation Loss")

    # Set the layout size and show the legend
    fig.update_layout(width=1000, height=600, showlegend=True)
    
    fig.update_yaxes(range=[0, 4])
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")

    # Save the figure as a PDF file
    fig.write_image(f"{checkpoint_folder}/Train_and_Val_loss.pdf")
    time.sleep(10)
    fig.write_image(f"{checkpoint_folder}/Train_and_Val_loss.pdf")
