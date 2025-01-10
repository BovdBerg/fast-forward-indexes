import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Tuple, Sequence

import lightning
import torch
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_forward.index import Index
from fast_forward.index.disk import OnDiskIndex

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_cache_path",
        type=Path,
        default="data/rep-to-rep/tct-to-emb",
        help="Path to the dataloader file to save or load.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="mem",
        choices=["disk", "mem"],
        help="""The storage type of the index. 
        'mem' takes some time to load Index into memory, which speeds up Index._get_vectors().
        """,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1_000,
        help="""Number of queries to sample from the dataset.
        Traditional (too simplistic) rule of thumb: at least 10 * |features|""",
    )
    parser.add_argument(
        "--index_tct_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the TCT index.",
    )
    parser.add_argument(
        "--index_emb_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_emb_bert_opq.h5",
        help="Path to the TCT index.",
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
    return parser.parse_args()


def setup() -> Tuple[Index, Index]:
    """Setup and initialize relevant objects.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        Tuple[pd.DataFrame, Index, Index]: Dataset, TCT index, and embedding index.
    """
    index_tct = OnDiskIndex.load(args.index_tct_path)
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**15)

    index_emb = OnDiskIndex.load(args.index_emb_path)
    if args.storage == "mem":
        index_emb = index_emb.to_memory(2**15)

    return index_tct, index_emb


def create_data() -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
    print("\033[96m")  # Prints in this method are cyan

    dataset_name = "irds:msmarco-passage"
    dataset_file = args.dataset_cache_path / dataset_name / f"0-{args.samples}"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    index_tct, index_emb = setup()

    dataset = []
    if (dataset_file).exists():
        print(f"Loading dataset from {dataset_file}")
        dataset = torch.load(dataset_file)
    else:
        # TODO: Remove --samples and use all documents in the dataset -- index_tct.doc_ids()
        for docno in tqdm(range(args.samples), desc="Creating dataset", total=args.samples):
            input = index_tct._get_vectors([str(docno)])
            target = index_emb._get_vectors([str(docno)])
            dataset.append((input, target))
            
        torch.save(dataset, dataset_file)

    print("\033[0m")  # Reset print color
    return dataset


def main() -> None:
    """
    Train a model using PyTorch Lightning.
    """
    start_time = time.time()

    dataset = create_data()

    # TODO: Split dataset into train and val
    # TODO: Create DataLoaders for train and val. Shuffle for training, not for validation.
    # dataloader = DataLoader(
    #     dataset,  # type: ignore
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=True,
    # )

    return
    # Train the model
    k_avg = args.k_avg
    if args.with_queries:
        k_avg += 1  # +1 for emb-encoded query
    adapter = MODEL()
    trainer = lightning.Trainer(
        deterministic="warn",
        max_epochs=50,
        limit_train_batches=args.samples,
        limit_val_batches=val_samples,
        log_every_n_steps=250,
        val_check_interval=1.0 if args.samples <= 1000 else 0.1,
        callbacks=[
            callbacks.ModelCheckpoint(monitor="val_loss", verbose=True),
            callbacks.EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=5, verbose=True
            ),
        ],
    )
    trainer.fit(
        model=adapter,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    end_time = time.time()
    print(f"\nScript took {end_time - start_time:.2f} seconds to complete.")
    return


if __name__ == "__main__":
    args = parse_args()
    main()
