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

from fast_forward.adapter import Adapter
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
        default="data/rep-to-rep/tct-to-emb/",
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
        default=100_000,
        help="""Number of docs to sample from the dataset.
        Traditional (too simplistic) rule of thumb: at least 10 * |features|.
        If not provided, all docs are used.""",
    )
    parser.add_argument(
        "--index_input_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5",
        help="Path to the input index.",
    )
    parser.add_argument(
        "--index_target_path",
        type=Path,
        default="/home/bvdb9/indices/msm-psg/ff_index_msmpsg_emb_bert_opq.h5",
        help="Path to the target index.",
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
    index_tct = OnDiskIndex.load(args.index_input_path)
    if args.storage == "mem":
        index_tct = index_tct.to_memory(2**15)

    index_emb = OnDiskIndex.load(args.index_target_path)
    if args.storage == "mem":
        index_emb = index_emb.to_memory(2**15)

    return index_tct, index_emb


def create_data() -> Tuple[DataLoader, DataLoader]:
    print("\033[96m")  # Prints in this method are cyan

    dataset_name = "irds:msmarco-passage"
    name = f"{args.samples}_samples" if args.samples else "all"
    dataset_file = args.dataset_cache_path / dataset_name / f"{name}.pt"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = []
    if (dataset_file).exists():
        print(f"Loading dataset from {dataset_file}")
        dataset = torch.load(dataset_file)
    else:
        index_tct, index_emb = setup()

        doc_ids = range(args.samples) if args.samples else index_tct.doc_ids
        for docno in tqdm(doc_ids, desc="Creating dataset", total=len(doc_ids)):
            input, _ = index_tct._get_vectors([str(docno)])
            if index_tct.quantizer is not None:
                input = index_tct.quantizer.decode(input)
            target, _ = index_emb._get_vectors([str(docno)])
            if index_emb.quantizer is not None:
                target = index_emb.quantizer.decode(target)
            dataset.append((input, target))
            
        torch.save(dataset, dataset_file)

    print("Splitting dataset into train and validation sets...")
    val_samples = int(len(dataset) * 0.2)
    train_samples = len(dataset) - val_samples

    train_dataset = dataset[:train_samples]
    val_dataset = dataset[train_samples:]

    train_loader = DataLoader(
        train_dataset, # type: ignore
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, # type: ignore
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print("Created train and validation dataloaders.")

    print("\033[0m")  # Reset print color
    return train_loader, val_loader


def main() -> None:
    """
    Train a model using PyTorch Lightning.
    """
    start_time = time.time()

    train_loader, val_loader = create_data()

    # Train the model
    adapter = Adapter()
    trainer = lightning.Trainer(
        deterministic="warn",
        max_epochs=50,
        log_every_n_steps=1 if args.samples <= 1000 else args.samples // 100,
        val_check_interval=1.0 if args.samples <= 1000 else 0.1,
        callbacks=[
            callbacks.ModelCheckpoint(monitor="val_loss", verbose=True),
            callbacks.EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=3, verbose=True
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
