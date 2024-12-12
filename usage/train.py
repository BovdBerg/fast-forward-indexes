from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyterrier as pt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util.pyterrier import FFInterpolate, FFScore

### PARAMETERS
BATCH_SIZE = 1
SAMPLES = 1000
K_AVG = 10
DIM = 768
IN_MEMORY = True
SAVE_INTERVAL = 1000


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
    Path("/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5"),
    TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ),
    verbose=False,
)
if IN_MEMORY:
    index_tct = index_tct.to_memory(2**15)
sys_tct = sys_bm25_cut >> FFScore(index_tct) >> FFInterpolate(alpha=0.1)


### Dataset and DataLoader
def dataset_to_dataloader(
    dataset_name: str,
    shuffle: bool,
) -> DataLoader:
    dataset = pt.get_dataset(dataset_name)
    topics = dataset.get_topics().sample(n=SAMPLES, random_state=42)

    set = []
    for query in tqdm(
        topics.itertuples(), desc="Processing queries", total=len(topics)
    ):
        query = pd.DataFrame([query._asdict()])

        # Encode query using TCT-ColBERT
        q_rep_tct = index_tct.encode_queries(query["query"])[0]

        # Get the top-ranked document vectors for the query
        # TODO: would be cleaner to use an index with WeightedAvgEncoder and add a method there to get the d_reps
        df_sparse = sys_bm25_cut.transform(query)
        ranking_sparse = Ranking(
            df_sparse.rename(columns={"qid": "q_id", "docno": "id"})
        ).cut(K_AVG)
        top_docs = ranking_sparse._df.query("query == @query['query'].iloc[0]")
        # Skip queries with too little top_docs
        if len(top_docs) == 0:
            print(f"Skipping query {query['qid'].iloc[0]}: '{query['query'].iloc[0]}' (has no top_docs)")
            continue
        top_docs_ids = top_docs["id"].values
        d_reps, d_idxs = index_tct._get_vectors(top_docs_ids)
        if index_tct.quantizer is not None:
            d_reps = index_tct.quantizer.decode(d_reps)
        order = [x[0] for x in d_idxs]  # [[0], [2], [1]] --> [0, 2, 1]
        d_reps = d_reps[order]  # sort d_reps on d_ids order

        set.append((d_reps, q_rep_tct))  # (inputs, labels)

    dataloader = DataLoader(set, batch_size=BATCH_SIZE, shuffle=shuffle)
    print("{} set has {} instances".format(dataset_name, len(dataloader)))
    return dataloader


# Create data loaders for our datasets; shuffle for training, not for validation
train_loader = dataset_to_dataloader("irds:msmarco-passage/train", True)
val_loader = dataset_to_dataloader("irds:msmarco-passage/dev", False)


### Model
class LearnedAvgWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(K_AVG) / K_AVG
        )  # shape (K_AVG,), init as uniform weights
        print("Model initialized as {}".format(self))

    def forward(
        self,
        d_reps: np.ndarray,  # shape (BATCH_SIZE, K_AVG, DIM) or (K_AVG, DIM)
    ) -> Sequence[float]:  # shape (BATCH_SIZE, DIM)
        q_rep = torch.einsum("k,bkd->bd", self.weights, d_reps)
        return q_rep


model = LearnedAvgWeights()
# model.load_state_dict(torch.load(model_path))


### Create target embedding


### Loss Function
loss_fn = nn.MSELoss()  # TODO: select loss function to fit my needs


### Optimizer
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


### Training Loop
def train_one_epoch(epoch_index, tb_writer):
    run_loss = 0.0
    last_loss = 0.0

    # Gets a batch of training data from the DataLoader
    for i, (inputs, labels) in enumerate(train_loader):
        print(f"inputs.shape: {inputs.shape}, labels.shape: {labels.shape}")
        # Zero the optimizer’s gradients
        optimizer.zero_grad()

        # Performs an inference - gets predictions from the model
        outputs = model(inputs)

        # Compute the loss and its gradients
        # Calculate loss for the predictions vs. dataset labels
        loss = loss_fn(outputs, labels)

        # Calculate backward gradients over learning weights
        loss.backward()

        # Adjust learning weights
        # Perform 1 optimizer learning step - adjust model’s learning weights based on the observed gradients, according to the optimization algorithm we chose
        optimizer.step()

        # Gather data and reports avg per-batch loss every 1000 batches. For comparison with a validation run
        run_loss += loss.item()
        if i % SAVE_INTERVAL == SAVE_INTERVAL - 1:
            last_loss = run_loss / SAVE_INTERVAL  # loss per batch
            print("\tbatch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            run_loss = 0.0

    return last_loss


### Per-Epoch Activity
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("runs/avg_weights_{}".format(timestamp))

EPOCHS = 3

best_vloss = float("inf")
for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer)

    run_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            run_vloss += vloss

    avg_vloss = run_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch + 1,
    )
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_dir = Path("outputs/models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model_{}_{}".format(timestamp, epoch)
        print("Saving model to {}".format(model_path))
        torch.save(model.state_dict(), model_path)
