import torch
from fast_forward.encoder.avg import LearnedAvgWeights

ckpt_path = "/home/bvdb9/fast-forward-indexes/lightning_logs/version_8/checkpoints/epoch=0-step=995.ckpt"
learned_weights = LearnedAvgWeights.load_from_checkpoint(ckpt_path)
encoder = learned_weights.encoder
encoder.eval()
print(f"Encoder loaded as: {encoder}")

ckpt = torch.load(ckpt_path)
for k, v in ckpt["state_dict"].items():
    print(f"k: {k}, v.shape: {v.shape}")

# embed 10 fake d_reps!
fake_image_batch = torch.rand(10 * 768, device=learned_weights.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions:\n", embeddings.shape, "\n", "⚡" * 20)
