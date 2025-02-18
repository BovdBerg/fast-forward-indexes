import argparse
import os
import pickle
import time
import warnings
from math import ceil
from pathlib import Path
from typing import Tuple

import lightning
import pandas as pd
import pyterrier as pt
import torch
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_forward.encoder.avg import WEIGHT_METHOD, AvgEmbQueryEstimator
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*`resume_download` is deprecated.*"
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # AvgEmbQueryEstimator arguments
    parser.add_argument(
        "--index_path",
        type=Path,
        # Note that training on non-OPQ index has better results.
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
        default=10,
        help="Number of top-ranked documents to average.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=None,
        help="Path to a checkpoint file to load the model from.",
    )
    parser.add_argument(
        "--q_only",
        action="store_true",
        help="Whether to only use the lightweight query estimation and not the top-ranked documents.",
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
        default=None,
        help="Number of queries to sample from the dataset. If not specified, use all samples.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of workers for the DataLoader.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=None,
        help="Validation check interval in epochs.",
    )

    return parser.parse_args()


def create_data(
    dataset_name: str,
    samples: int,
    shuffle: bool,
) -> Tuple[DataLoader, pd.DataFrame]:
    dataset_stem = args.dataset_cache_path / dataset_name
    dataset_stem.mkdir(parents=True, exist_ok=True)
    print(f"Creating/retrieving dataset for {samples} samples for {dataset_stem}")

    topics = pt.get_dataset(dataset_name).get_topics()
    if samples:
        topics = topics.sample(n=samples, random_state=42)

    encoder_tct = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco",
        device=args.device,
    )

    step = min(samples, 10_000)
    dataset = []
    for lb in tqdm(
        range(0, samples, step),
        desc="Processing dataset steps",
        total=ceil(samples / step),
    ):
        ub = lb + step
        step_dataset_file = dataset_stem / f"{lb}-{ub}.pt"
        step_dataset_file.parent.mkdir(parents=True, exist_ok=True)

        if (step_dataset_file).exists():  # Load dataset part
            new_data = torch.load(step_dataset_file, map_location=args.device)
        else:
            print(f"...Step {lb}-{ub}: Creating new data in {step_dataset_file}")
            step_topics = topics.iloc[lb:ub]

            queries = step_topics["query"].tolist()
            new_data = []
            batch_size = 32
            for i in tqdm(
                range(0, len(queries), batch_size),
                desc="Encoding queries",
                total=len(queries) // batch_size,
            ):
                batch_queries = queries[i : i + batch_size]
                q_reps_tct = encoder_tct(batch_queries)
                new_data.extend(zip(batch_queries, q_reps_tct))

            torch.save(new_data, step_dataset_file)

        dataset.extend(new_data)

    dataloader = DataLoader(
        dataset=dataset,  # type: ignore
        shuffle=shuffle,
        num_workers=args.num_workers,
        drop_last=True,
        batch_size=args.batch_size,
    )

    return dataloader, topics


def create_lexical_ranking(queries_path: Path):
    cache_n_docs = 50
    cache_dir = args.dataset_cache_path / f"ranking_cache_{cache_n_docs}docs"
    os.makedirs(cache_dir, exist_ok=True)
    chunk_size = 10_000

    res_df = pd.DataFrame()
    queries = pd.read_csv(queries_path)
    for i, chunk in enumerate(
        tqdm(
            pd.read_csv(queries_path, chunksize=chunk_size),
            desc="Loading/creating Ranking in chunks",
            total=ceil(len(queries) / chunk_size),
        )
    ):
        cache_file = cache_dir / f"{i * chunk_size}-{(i + 1) * chunk_size}.pt"

        if cache_file.exists():
            with open(cache_file, "rb") as f:
                chunk_ranking = pickle.load(f)
        else:
            print(f"Creating new ranking for {cache_file}")
            sys_bm25 = (
                pt.BatchRetrieve.from_dataset(
                    "msmarco_passage",
                    "terrier_stemmed",
                    wmodel="BM25",
                    memory=True,
                    verbose=True,
                    num_results=cache_n_docs,
                )
                % cache_n_docs
            )
            chunk["query"] = chunk["query"].astype(str)
            chunk_df = sys_bm25.transform(chunk)
            chunk_ranking = Ranking(
                chunk_df.rename(columns={"qid": "q_id", "docno": "id"})
            )

            with open(cache_file, "wb") as f:
                pickle.dump(chunk_ranking, f)

        chunk_ranking = chunk_ranking.cut(args.n_docs)
        res_df = pd.concat([res_df, chunk_ranking._df])

    return Ranking(res_df).cut(args.n_docs)


def setup() -> tuple[AvgEmbQueryEstimator, DataLoader, DataLoader]:
    print("\033[96m")  # Prints during setup are colored cyan
    pt.init()

    train_topics = pt.get_dataset("irds:msmarco-passage/train").get_topics()
    n_train_topics = len(train_topics)
    if args.samples is not None:
        n_train_topics = min(args.samples, n_train_topics)

    train_dataloader, train_topics = create_data(
        dataset_name="irds:msmarco-passage/train", samples=n_train_topics, shuffle=True
    )
    val_dataloader, val_topics = create_data(
        dataset_name="irds:msmarco-passage/eval",  # TODO: should be dev, but I have eval cached
        samples=min(1_000, n_train_topics),
        shuffle=False,
    )

    # Create model pre-requisites
    all_topics = pd.concat(
        [val_topics, train_topics]
    )  # Important that val_topics is first, because len(train_topics) may vary.
    queries_path = args.dataset_cache_path / f"{len(all_topics)}_topics.csv"
    all_topics.to_csv(queries_path, index=False)
    lexical_ranking = create_lexical_ranking(queries_path)

    index = OnDiskIndex.load(args.index_path)
    if args.storage == "mem":
        index = index.to_memory(2**15)

    # Create model instance
    query_estimator = AvgEmbQueryEstimator(
        index=index,
        n_docs=args.n_docs,
        device=args.device,
        ranking=lexical_ranking,
        ckpt_path=args.ckpt_path,
        q_only=args.q_only,
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
        log_every_n_steps=(
            1 if len(train_dataloader) <= 1000 else len(train_dataloader) // 100
        ),
        val_check_interval=(
            args.val_check_interval
            if args.val_check_interval
            else (
                1.0
                if len(train_dataloader) <= 1_000
                else 0.5 if len(train_dataloader) <= 10_000 else 0.1
            )
        ),
        callbacks=[
            callbacks.ModelCheckpoint(
                monitor="val_loss",
                verbose=True,
            ),
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
