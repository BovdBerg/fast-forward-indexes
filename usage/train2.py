import argparse
import os
from pathlib import Path

import lightning as L
import pandas as pd
import pyterrier as pt
import torch
import torch.nn as nn
from torch import nn, optim, utils
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.

    Arguments:
        Run the script with --help or -h to see the full list of arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create an OPQ index from an existing Fast-Forward index."
    )
    parser.add_argument(
        "--tct_index_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the TCT index.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of instances to process in a batch.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of queries to sample from the dataset.",
    )
    parser.add_argument(
        "--k_avg",
        type=int,
        default=10,
        help="Number of top-ranked documents to average.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="mem",
        choices=["disk", "mem"],
        help="The storage type of the index.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=20,
        help="Number of batches between each loss print.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to a model to load and continue training from.",
    )
    return parser.parse_args()


# TODO: training_step input should be a tuple of (x, y) where x is the input and y is the target.
### 2: Define a LightningModule
class LearnedAvgWeights(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10 * 768, 768))
        print(f"LearnedAvgWeights initialized as: {self}")

    def training_step(self, batch, batch_idx):
        print(f"batch shape: {batch[0].shape}")
        (x, y), _ = batch  # x should be combination of d_reps and q_rep_tct
        print(f"x shape: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"x shape 2: {x.shape}")
        z = self.encoder(x)
        print(f"z shape: {z.shape}")
        x_hat = self.decoder(z)
        print(f"x_hat shape: {x_hat.shape}")
        loss = nn.functional.mse_loss(x_hat, y)  # TODO: verify if this is a correct loss function
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # def predict_step(self, *args, **kwargs):
    # def validation_step(self, *args, **kwargs):
    # def test_step(self, *args, **kwargs):


def main(args: argparse.Namespace) -> None:
    """
    Train a model using PyTorch Lightning.
    """
    ### PyTerrier setup
    pt.init()

    # BM25
    sys_bm25 = pt.BatchRetrieve.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25"
    )
    sys_bm25.verbose = False
    sys_bm25_cut = ~sys_bm25 % 1000

    # FF with TCT-ColBERT encoding + Interpolation
    index_tct = OnDiskIndex.load(
        args.tct_index_path,
        TCTColBERTQueryEncoder(
            "castorini/tct_colbert-msmarco",
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
    )
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**15)


    ### 3: Define a dataset
    def dataset_to_dataloader(
        dataset_name: str,
        shuffle: bool,
    ) -> DataLoader:
        dataset = pt.get_dataset(dataset_name)
        topics = dataset.get_topics().sample(n=args.samples, random_state=42)

        set = []
        for query in tqdm(
            topics.itertuples(), desc="Processing queries", total=len(topics)
        ):
            query = pd.DataFrame([query._asdict()])

            ## Label: query encoded using TCT-ColBERT
            q_rep_tct = index_tct.encode_queries(query["query"])[0]

            ## Inputs: top-ranked document vectors for the query
            # TODO: would be cleaner to use an index with WeightedAvgEncoder and add a method there to get the d_reps
            df_sparse = sys_bm25_cut.transform(query)
            ranking_sparse = Ranking(
                df_sparse.rename(columns={"qid": "q_id", "docno": "id"})
            ).cut(args.k_avg)
            top_docs = ranking_sparse._df.query("query == @query['query'].iloc[0]")
            # Skip queries with too little top_docs
            if len(top_docs) == 0:
                print(
                    f"Skipping query {query['qid'].iloc[0]}: '{query['query'].iloc[0]}' (has no top_docs)"
                )
                continue
            top_docs_ids = top_docs["id"].values
            d_reps, d_idxs = index_tct._get_vectors(top_docs_ids)
            if index_tct.quantizer is not None:
                d_reps = index_tct.quantizer.decode(d_reps)
            order = [x[0] for x in d_idxs]  # [[0], [2], [1]] --> [0, 2, 1]
            d_reps = d_reps[order]  # sort d_reps on d_ids order

            set.append(((d_reps, q_rep_tct), [1]))  # (inputs, labels)

        dataloader = DataLoader(set, batch_size=args.batch_size, shuffle=shuffle, num_workers=11)
        print("{} set has {} instances".format(dataset_name, len(dataloader)))
        return dataloader


    # Create data loaders for our datasets; shuffle for training, not for validation
    train_loader = dataset_to_dataloader("irds:msmarco-passage/train", True)
    # val_loader = dataset_to_dataloader("irds:msmarco-passage/eval", False)

    ### 4: Train the model
    # TODO: inspect Trainer class in detail: https://lightning.ai/docs/pytorch/stable/common/trainer.html
    learned_avg_weights = LearnedAvgWeights()
    trainer = L.Trainer(limit_train_batches=500, max_epochs=args.max_epochs, log_every_n_steps=1)
    trainer.fit(model=learned_avg_weights, train_dataloaders=train_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
