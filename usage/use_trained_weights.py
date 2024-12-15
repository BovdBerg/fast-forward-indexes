"""
Watch this short video on PyTorch for this class to make sense: https://youtu.be/ORMx45xqWkA?si=Bvkm9SWi8Hz1n2Sh&t=147
"""

import torch
from fast_forward.encoder.avg import LearnedAvgWeights

ckpt_path = "/home/bvdb9/fast-forward-indexes/lightning_logs/version_8/checkpoints/epoch=0-step=995.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LearnedAvgWeights.load_from_checkpoint(ckpt_path, k_avg=10).to(device)
model.eval()  # disable randomness, dropout, etc.

# embed some fake top docs (= d_reps)! model(X) calls model.forward(X). Output is weighted average of d_reps.
X = torch.rand(10 * 768, device=model.device)
embeddings = model(X)
print("⚡" * 20, "\nPredictions:\n", embeddings.shape, "\n", "⚡" * 20)
