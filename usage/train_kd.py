import argparse
import os
from pydoc_data import topics
import time
import warnings
from pathlib import Path
from typing import Sequence, Tuple

import lightning
import pyterrier as pt
import torch
from lightning.pytorch import callbacks
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_forward.encoder.transformer import TCTColBERTQueryEncoder

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
        default="data/q-to-rep/tct/",
        help="Path to the dataloader file to save or load.",
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


def create_data(
    dataset_name: str,
    samples: int,
) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
    print("\033[96m")  # Prints in this method are cyan

    name = f"{samples}_samples" if samples else "all"
    dataset_file = args.dataset_cache_path / dataset_name / f"{name}.pt"
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    encoder_tct = TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    dataset = []
    if (dataset_file).exists():
        print(f"Loading dataset from {dataset_file}")
        dataset = torch.load(dataset_file)
    else:
        topics = pt.get_dataset(dataset_name).get_topics()
        if samples:
            topics = topics.sample(n=samples, random_state=42)

        for query in tqdm(topics["query"], desc="Creating dataset", total=len(topics)):
            q_rep_tct = encoder_tct([query])[0]
            dataset.append((query, q_rep_tct))
            
        torch.save(dataset, dataset_file)

    print("\033[0m")  # Reset print color
    return dataset


def main() -> None:
    """
    Train a model using PyTorch Lightning.
    """
    start_time = time.time()
    pt.init()

    train_loader = create_data("irds:msmarco-passage/train", args.samples)
    train_loader = create_data("irds:msmarco-passage/eval", 1_000)

    # TODO: Split dataset into train and val
    # TODO: Create DataLoaders for train and val.
    # dataloader = DataLoader(
    #     dataset,  # type: ignore
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=True,
    # )

    return
    # Train the model
    adapter = MODEL()
    trainer = lightning.Trainer(
        deterministic="warn",
        max_epochs=50,
        limit_val_batches=val_samples,
        log_every_n_steps=250,
        val_check_interval=0.1,
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
