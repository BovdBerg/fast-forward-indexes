import argparse
import os
import time
import warnings
from math import ceil
from pathlib import Path
from typing import Tuple

import lightning
import pyterrier as pt
import torch
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fast_forward.encoder.avg import LearnedAvgWeights, WeightedAvgEncoder
from fast_forward.encoder.transformer import TCTColBERTQueryEncoder, TransformerEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)
setup_done = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a LearnedAvgWeights model.")
    parser.add_argument(
        "--dataset_cache_path",
        type=Path,
        default="data/",
        help="Path to the dataloader file to save or load.",
    )
    parser.add_argument(
        "--with_queries",
        action="store_true",
        help="Include the query in the model input.",
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
        default=80_000,
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
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs to train the model (if not stopped by EarlyStopping).",
    )
    parser.add_argument(
        "--test_datasets",
        type=str,
        nargs="*",
        default=[
            "irds:msmarco-passage/trec-dl-2019/judged",
        ],
        help="List of test datasets to evaluate the model on.",
    )
    return parser.parse_args()


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
    encoder_avg = WeightedAvgEncoder(
        index_tct, k_avg=args.k_avg, ckpt_path=args.ckpt_path
    )

    return sys_bm25_cut, encoder_tct, encoder_avg


def dataset_to_dataloader(
    dataset_name: str,
    samples: int,
) -> DataLoader:
    """Create a DataLoader for the given dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        DataLoader: A DataLoader for the given dataset.
    """
    global setup_done, sys_bm25_cut, encoder_tct, encoder_avg
    print("\033[96m")  # Prints in this method are cyan
    suffix = "+query" if args.with_queries else ""
    dataset_stem = (
        args.dataset_cache_path / dataset_name / f"k_avg-{args.k_avg}{suffix}"
    )
    step = 1000
    samples_ub = ceil(samples / step) * step  # Ceil ub to nearest 10k

    dataset = []
    print(f"Creating/retrieving dataset for {samples_ub} samples from {dataset_name}")
    for lb in tqdm(
        range(0, samples_ub, step),
        desc="Processing dataset steps",
        total=samples_ub // step,
    ):
        ub = lb + step
        step_dataset_file = dataset_stem / f"{lb}-{ub}"
        step_dataset_file.parent.mkdir(parents=True, exist_ok=True)

        if (step_dataset_file).exists():  # Load dataset part
            new_data = torch.load(step_dataset_file)
        else:
            print(f"...Step {lb}-{ub}: Creating new data in {step_dataset_file}")
            if not setup_done:
                print(
                    f"Setting up BM25, TCT-ColBERT, and WeightedAvg for {dataset_name}"
                )
                sys_bm25_cut, encoder_tct, encoder_avg = setup()
                setup_done = True
            topics = pt.get_dataset(dataset_name).get_topics()

            step_topics = topics.iloc[lb:ub]
            top_ranking = Ranking(
                sys_bm25_cut.transform(step_topics).rename(
                    columns={"qid": "q_id", "docno": "id"}
                )
            )

            new_data = []
            for query in tqdm(
                step_topics["query"], desc="Processing queries", total=len(step_topics)
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

                if args.with_queries:
                    data = ((d_reps, query), q_rep_tct)
                else:
                    data = (d_reps, q_rep_tct)
                new_data.append(data)  # (inputs, labels, [query])

            torch.save(new_data, step_dataset_file)

        dataset.extend(new_data)

    # Convert list to TensorDataset
    inputs, labels = zip(*dataset)
    inputs_tensor = torch.stack([torch.tensor(i) for i in inputs])
    labels_tensor = torch.stack([torch.tensor(l) for l in labels])
    tensor_dataset = TensorDataset(inputs_tensor, labels_tensor)

    # Cut dataset to --samples and create DataLoader
    dataloader = DataLoader(
        tensor_dataset,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"Created dataloader with {len(dataloader)} instances from {dataset_name}.")
    print("\033[0m")  # Reset print color
    return dataloader


def main() -> None:
    """
    Train a model using PyTorch Lightning.
    """
    start_time = time.time()

    # Create data loaders for our datasets; shuffle for training, not for validation
    train_loader = dataset_to_dataloader("irds:msmarco-passage/train", args.samples)
    val_samples = 1000
    val_loader = dataset_to_dataloader("irds:msmarco-passage/eval", val_samples)

    # Train the model
    # TODO: inspect Trainer class in detail: https://lightning.ai/docs/pytorch/stable/common/trainer.html
    learned_avg_weights = LearnedAvgWeights(k_avg=args.k_avg)
    trainer = lightning.Trainer(
        deterministic="warn",
        max_epochs=1,
        limit_train_batches=args.samples,
        limit_val_batches=val_samples,
        log_every_n_steps=250,
        val_check_interval=0.05,
        callbacks=[
            callbacks.ModelCheckpoint(monitor="val_loss", verbose=True),
            # callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=5, verbose=True),
        ],
    )
    trainer.fit(
        model=learned_avg_weights,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    for dataset in args.test_datasets:
        print(f"Testing the trained model on {dataset}...")
        test_loader = dataset_to_dataloader(dataset, 43)
        trainer.test(model=learned_avg_weights, dataloaders=test_loader)

    # TODO: save best model to transformers hub?

    end_time = time.time()
    print(f"\nScript took {end_time - start_time:.2f} seconds to complete.")
    return


if __name__ == "__main__":
    args = parse_args()
    main()
