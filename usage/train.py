from datetime import datetime
from typing import Sequence
import ir_datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


### Dataset and DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# # Create datasets for training & validation
# training_dataset = ir_datasets.load('msmarco-passage/train')
# validation_dataset = ir_datasets.load('msmarco-passage/dev')

# training_set = []
# for query in training_dataset.queries_iter():
#     training_set.append(query)

# validation_set = []
# for query in validation_dataset.queries_iter():
#     validation_set.append(query)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))



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



### Loss Function
# TODO: select loss function to fit my needs
loss_fn = torch.nn.CrossEntropyLoss()

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
    run_loss = 0.
    last_loss = 0.

    # Gets a batch of training data from the DataLoader
    for i, (inputs, labels) in enumerate(training_loader):
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
            last_loss = run_loss / 1000 # loss per batch
            print('\tbatch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            run_loss = 0.

    return last_loss



### Per-Epoch Activity
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp)) # TODO: rename to fit my needs

EPOCHS = 3

best_vloss = float('inf')
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer)


    run_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            run_vloss += vloss

    avg_vloss = run_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    {'Training': avg_loss, 'Validation': avg_vloss},
                    epoch + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'outputs/models/model_{}_{}'.format(timestamp, epoch)
        print('Saving model to {}'.format(model_path))
        torch.save(model.state_dict(), model_path)
