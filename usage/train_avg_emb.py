import argparse
import time
import warnings
from pathlib import Path
from typing import Sequence, Tuple

import lightning
import pandas as pd
import pyterrier as pt
import torch
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_forward.encoder.avg import AvgEmbQueryEstimator
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # AvgEmbQueryEstimator arguments
    parser.add_argument(
        "--index_tct_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the TCT index.",
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
        "--n_docs",
        type=int,
        default=30,
        help="Number of top-ranked documents to average.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training.",
    )

    # Training arguments
    parser.add_argument(
        "--dataset_cache_path",
        type=Path,
        default="data/q-to-rep/tct/",
        help="Path to the dataloader file to save or load.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10_000,
        help="Number of queries to sample from the dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=11,
        help="Number of workers for the DataLoader.",
    )

    return parser.parse_args()


def create_data(
    dataset_name: str,
    samples: int,
    shuffle: bool,
) -> Tuple[DataLoader, pd.DataFrame]:
    name = f"{samples}_samples" if samples else "all"
    dataset_file = args.dataset_cache_path / dataset_name / f"{name}.pt"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    topics = pt.get_dataset(dataset_name).get_topics()
    if samples:
        topics = topics.sample(n=samples, random_state=42)

    encoder_tct = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco",
        device=args.device,
    )

    dataset = []
    if (dataset_file).exists():
        print(f"Loading dataset from {dataset_file}")
        dataset = torch.load(dataset_file)
    else:
        for query in tqdm(topics["query"], desc="Creating dataset", total=len(topics)):
            q_rep_tct = encoder_tct([query])[0]
            dataset.append((query, q_rep_tct))

        torch.save(dataset, dataset_file)

    dataloader = DataLoader(
        dataset=dataset,  # type: ignore
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True,
    )

    return dataloader, topics


def setup() -> tuple[AvgEmbQueryEstimator, DataLoader, DataLoader]:
    print("\033[96m")  # Prints during setup are colored cyan
    pt.init()

    train_dataloader, train_topics = create_data(
        dataset_name="irds:msmarco-passage/train", samples=args.samples, shuffle=True
    )
    val_dataloader, val_topics = create_data(
        dataset_name="irds:msmarco-passage/eval", samples=1_000, shuffle=False
    )


    # Create model pre-requisites
    index = OnDiskIndex.load(args.index_tct_path)
    if args.storage == "mem":
        index = index.to_memory(2**15)

    sys_bm25 = pt.BatchRetrieve.from_dataset(
        "msmarco_passage", "terrier_stemmed", wmodel="BM25", memory=True, verbose=True
    )
    all_topics = train_topics + val_topics
    # TODO: why is all_topics NaN for qid and query
    print(f"all_topics:\n{all_topics}")
    lexical_ranking = Ranking(
        sys_bm25.transform(all_topics).rename(columns={"qid": "q_id", "docno": "id"})
    )

    # Create model instance
    query_estimator = AvgEmbQueryEstimator(
        index=index, n_docs=args.n_docs, device=args.device, ranking=lexical_ranking
    )

    print("\033[0m")  # Reset print color
    return query_estimator, train_dataloader, val_dataloader


def main() -> None:
    """
    Train a model using PyTorch Lightning.
    """
    start_time = time.time()

    model, train_dataloader, val_dataloader = setup()

    # Train model
    trainer = lightning.Trainer(
        deterministic="warn",
        max_epochs=50,
        limit_train_batches=args.samples,
        log_every_n_steps=1 if args.samples <= 1000 else args.samples // 100,
        val_check_interval=1.0 if args.samples <= 1000 else 0.1,
        callbacks=[
            callbacks.ModelCheckpoint(monitor="val_loss", verbose=True),
            callbacks.EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=5, verbose=True
            ),
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # TODO: save best model to transformers hub?

    end_time = time.time()
    print(f"\nScript took {end_time - start_time:.2f} seconds to complete.")
    return


if __name__ == "__main__":
    args = parse_args()
    main()
