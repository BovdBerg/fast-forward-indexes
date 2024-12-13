import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import lightning as L
import pyterrier as pt
import torch
import torch.nn as nn
from lightning.pytorch import callbacks
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_forward.encoder.avg import WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder, TransformerEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create an OPQ index from an existing Fast-Forward index."
    )
    parser.add_argument(
        "--dataset_cache_path",
        type=Path,
        default="data/msmarco-passage/train",
        help="Path to the dataloader file to save or load.",
    )
    parser.add_argument(
        "--tct_index_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the TCT index.",
    )
    parser.add_argument(
        "--k_avg",
        type=int,
        default=10,
        help="Number of top-ranked documents to average.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="""Number of queries to sample from the dataset.
        Traditional (too simplistic) rule of thumb: at least 10 * |features| = 10 * (k_avg * 768). 
        E.g. 76800 samples for k_avg=10.""",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="mem",
        choices=["disk", "mem"],
        help="""The storage type of the index. 
        'mem' takes some time to load Index into memory, which speeds up WeightedAvg._get_top_docs().
        Use 'disk' when using few samples, and 'mem' when using many samples.
        """,
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to the checkpoint file to load. If not provided, the model is trained from scratch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=11,
        help="Number of workers for the DataLoader.",
    )
    return parser.parse_args()


class LearnedAvgWeights(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(10 * 768, 768))
        print(f"LearnedAvgWeights initialized as: {self}")

    # TODO: should I use Contrastive loss with negatives such as bm25 hard-negatives & in-batch negatives?
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        loss = nn.functional.mse_loss(z, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def setup() -> Tuple[pt.Transformer, TransformerEncoder, WeightedAvgEncoder]:
    """Setup and initialize relevant objects.

    Returns:
        Tuple[pt.Transformer, TransformerEncoder, WeightedAvgEncoder]: Pyterrier BM25 transformer, TCT-ColBERT encoder, and WeightedAvg encoder.
    """
    pt.init()

    # BM25
    sys_bm25 = pt.BatchRetrieve.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25", memory=True, verbose=True
    )
    sys_bm25_cut = ~sys_bm25 % args.k_avg

    # TCT-ColBERT TransformerEncoder
    encoder_tct = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # WeightedAvgEncoder
    index_tct = OnDiskIndex.load(args.tct_index_path)
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**15)
    encoder_avg = WeightedAvgEncoder(index_tct, k_avg=args.k_avg)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        encoder_avg.load_state_dict(ckpt)

    return sys_bm25_cut, encoder_tct, encoder_avg


def dataset_to_dataloader(
    dataset_name: str,
    shuffle: bool,
    sys_bm25_cut: pt.Transformer,
    encoder_tct: TransformerEncoder,
    encoder_avg: WeightedAvgEncoder,
) -> DataLoader:
    """Create a DataLoader for the given dataset.

    Args:
        dataset_name (str): The name of the dataset.
        shuffle (bool): Whether to shuffle the dataset.
        sys_bm25_cut (pt.Transformer): The BM25 transformer.
        encoder_tct (TransformerEncoder): The TCT-ColBERT encoder.
        encoder_avg (WeightedAvgEncoder): The WeightedAvg encoder.

    Returns:
        DataLoader: A DataLoader for the given dataset.
    """
    dataset_cache_path = args.dataset_cache_path.with_name(
        f"{args.dataset_cache_path.stem}_k-avg={args.k_avg}"
    )
    if args.samples:
        dataset_cache_path = dataset_cache_path.with_stem(
            f"{dataset_cache_path.stem}_samples={args.samples}"
        )

    if dataset_cache_path.exists():
        dataset = torch.load(dataset_cache_path)
        print(f"Loaded dataloader from {dataset_cache_path}")
    else:
        s = f"Creating dataloader for {dataset_name} with "
        if args.samples:
            print(s + f"{args.samples} samples...")
        else:
            print(s + "all samples...")

        topics = pt.get_dataset(dataset_name).get_topics()
        if args.samples:
            topics = topics.sample(n=args.samples)

        top_ranking = Ranking(
            sys_bm25_cut.transform(topics).rename(
                columns={"qid": "q_id", "docno": "id"}
            )
        ).cut(args.k_avg)

        dataset = []
        for query in tqdm(
            topics["query"], desc="Processing queries", total=len(topics)
        ):
            # Label: query encoded by TCT-ColBERT
            q_rep_tct = encoder_tct([query])[0]  # [0]: only one query

            # Inputs: top-ranked document vectors for the query
            top_docs = encoder_avg._get_top_docs(query, top_ranking)
            if top_docs is None:
                continue  # skip sample: no top_docs
            d_reps, _ = top_docs
            if len(d_reps) < args.k_avg:
                continue  # skip sample: not enough top_docs

            dataset.append((d_reps, q_rep_tct))  # (inputs, labels)

        dataset_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, dataset_cache_path)
        print(f"Saved dataloader to {dataset_cache_path}")

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"Created dataloader with {len(dataloader)} instances from {dataset_name}")
    return dataloader


# TODO: rewrite to use ir_datasets instead of pyterrier. dataset.scoreddocs_iter() for bm25 % 1000.
def main() -> None:
    """
    Train a model using PyTorch Lightning.
    """
    start_time = time.time()
    sys_bm25_cut, encoder_tct, encoder_avg = setup()

    # Create data loaders for our datasets; shuffle for training, not for validation
    train_loader = dataset_to_dataloader(
        "irds:msmarco-passage/train", True, sys_bm25_cut, encoder_tct, encoder_avg
    )
    # val_loader = dataset_to_dataloader("irds:msmarco-passage/eval", False, sys_bm25_cut, encoder_tct, encoder_avg)

    # Train the model
    # TODO: inspect Trainer class in detail: https://lightning.ai/docs/pytorch/stable/common/trainer.html
    learned_avg_weights = LearnedAvgWeights()
    trainer = L.Trainer(
        deterministic="warn",
        max_epochs=50,
        callbacks=[
            callbacks.LearningRateMonitor(),
            callbacks.EarlyStopping(monitor="train_loss", patience=0, verbose=True),
            callbacks.ModelCheckpoint(monitor="train_loss", verbose=True),
        ],
    )
    trainer.fit(model=learned_avg_weights, train_dataloaders=train_loader)

    end_time = time.time()
    print(f"\nScript {__file__} took {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    args = parse_args()
    main()
