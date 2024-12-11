from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyterrier as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fast_forward.encoder.transformer import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.ranking import Ranking
from fast_forward.util.pyterrier import FFInterpolate, FFScore

### PyTerrier setup
pt.init()
sys_bm25 = pt.BatchRetrieve.from_dataset(
    "msmarco_passage", "terrier_stemmed", wmodel="BM25", verbose=True
)
sys_bm25_cut = ~sys_bm25 % 1000

index_tct = OnDiskIndex.load(
    Path("/home/bvdb9/indices/msm-psg/ff_index_msmpsg_TCTColBERT_opq.h5"),
    TCTColBERTQueryEncoder(
        "castorini/tct_colbert-msmarco",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ),
    verbose=True,
)
# index_tct = index_tct.to_memory(2**15)
sys_tct = sys_bm25_cut >> FFScore(index_tct) >> FFInterpolate(alpha=0.1)


### Dataset and DataLoader
BATCH_SIZE = 1
SAMPLES = 1
K_AVG = 10


# TODO: Load data (inputs, labels) per batch
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
        q_rep_tct = index_tct.encode_queries(query["query"])
        print(f"q_rep_tct.shape: {q_rep_tct.shape}")

        # Get the top-ranked document vectors for the query
        # TODO: would be cleaner to use an index with WeightedAvgEncoder and add a method there to get the d_reps
        df_sparse = sys_bm25_cut.transform(query)
        ranking_sparse = Ranking(
            df_sparse.rename(columns={"qid": "q_id", "docno": "id"})
        ).cut(K_AVG)
        top_docs = ranking_sparse._df.query("query == @query['query'].iloc[0]")
        top_docs_ids = top_docs["id"].values
        d_reps, d_idxs = index_tct._get_vectors(top_docs_ids)
        if index_tct.quantizer is not None:
            d_reps = index_tct.quantizer.decode(d_reps)
        order = [x[0] for x in d_idxs]  # [[0], [2], [1]] --> [0, 2, 1]
        d_reps = d_reps[order]  # sort d_reps on d_ids order
        print(f"d_reps.shape: {d_reps.shape}")
        # print(f'avg shape: {np.average(d_reps, axis=0).shape}')

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
        self.weights = nn.Parameter(torch.ones(10))
        print("Model initialized as {}".format(self))

    def forward(self, d_reps: np.ndarray) -> Sequence[float]:
        # Softmax: Normalize weights to sum to 1
        weights = F.softmax(self.weights, dim=0)
        # Compute the weighted average of the input embeddings
        q_rep = torch.sum(weights.unsqueeze(1) * d_reps, dim=0)
        return q_rep


model = LearnedAvgWeights()
# model.load_state_dict(torch.load(model_path))


### Create target embedding


### Loss Function
# TODO: select loss function to fit my needs
loss_fn = nn.MSELoss()

# Example usage:
# Assuming `input_embeddings` is a tensor of shape (10, 768)
# and `target_embedding` is a tensor of shape (768,)
input_embeddings = torch.randn(10, 768)  # Example input
target_embedding = torch.randn(768)  # Example target
print(
    "input_embeddings.shape: {}".format(input_embeddings.shape)
)  # Should be (10, 768,)
print("target_embedding.shape: {}".format(target_embedding.shape))  # Should be (768,)

# Forward pass
output_embedding = model(input_embeddings)

# Compute loss
loss = loss_fn(output_embedding, target_embedding)
print("Loss:", loss.item())

exit()

# TODO: modify classes, inputs, and test loss function
# classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
# print(f"Classes: {classes}")

# # NB: Loss functions expect data in batches, so we're creating batches of 4
# n_batches = 4
# n_classes = len(classes)

# # Represents the model's confidence in each of the n_classes classes for a given input
# dummy_outputs = torch.rand(n_batches, n_classes)
# print('dummy_outputs:\n{}'.format(dummy_outputs))
# print('-> selected: {}'.format(torch.argmax(dummy_outputs, dim=1)))

# # Represents the correct class among the n_classes being tested
# dummy_labels = torch.randint(0, n_classes, (n_batches,))
# print('dummy_labels: {}'.format(dummy_labels))

# loss = loss_fn(dummy_outputs, dummy_labels)
# print('Total loss for this batch: {}'.format(loss.item()))


### Optimizer
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


### Training Loop
def train_one_epoch(epoch_index, tb_writer):
    run_loss = 0.0
    last_loss = 0.0

    # Gets a batch of training data from the DataLoader
    for i, (inputs, labels) in enumerate(train_loader):
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
        if i % 1000 == 999:
            last_loss = run_loss / 1000  # loss per batch
            print("\tbatch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            run_loss = 0.0

    return last_loss


### Per-Epoch Activity
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(
    "runs/fashion_trainer_{}".format(timestamp)
)  # TODO: rename to fit my needs

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
        model_path = "outputs/models/model_{}_{}".format(timestamp, epoch)
        print("Saving model to {}".format(model_path))
        torch.save(model.state_dict(), model_path)
