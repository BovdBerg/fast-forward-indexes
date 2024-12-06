from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from fast_forward.encoder import Encoder


class TransformerEncoder(Encoder):
    """Uses a pre-trained transformer model for encoding. Returns the pooler output."""

    def __init__(
        self, model: Union[str, Path], device: str = "cpu", **tokenizer_args
    ) -> None:
        """Create a transformer encoder.

        Args:
            model (Union[str, Path]): Pre-trained transformer model (name or path).
            device (str, optional): PyTorch device. Defaults to "cpu".
            **tokenizer_args: Additional tokenizer arguments.
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.tokenizer_args = tokenizer_args

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", **self.tokenizer_args)
        inputs.to(self.device)
        with torch.no_grad():
            return self.model(**inputs).pooler_output.detach().cpu().numpy()


class TCTColBERTQueryEncoder(TransformerEncoder):
    """Query encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L72
    """

    def __call__(self, queries: Sequence[str]) -> np.ndarray:
        max_length = 36
        inputs = self.tokenizer(
            ["[CLS] [Q] " + q + "[MASK]" * max_length for q in queries],
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self.tokenizer_args,
        )
        inputs.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.detach().cpu().numpy()
            return np.average(embeddings[:, 4:, :], axis=-2)


class TCTColBERTDocumentEncoder(TransformerEncoder):
    """Document encoder for pre-trained TCT-ColBERT models.

    Adapted from Pyserini:
    https://github.com/castorini/pyserini/blob/310c828211bb3b9528cfd59695184c80825684a2/pyserini/encode/_tct_colbert.py#L27
    """

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        max_length = 512
        inputs = self.tokenizer(
            ["[CLS] [D] " + text for text in texts],
            max_length=max_length,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
            **self.tokenizer_args,
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs["last_hidden_state"][:, 4:, :]
            input_mask_expanded = (
                inputs["attention_mask"][:, 4:]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            return embeddings.detach().cpu().numpy()


class TransformerEmbeddingEncoder(TransformerEncoder):
    """Encodes a string using the average of the embedded tokens.
    Static token embeddings are obtained from a pre-trained Transformer model.
    """

    def __init__(
        self,
        model: Union[str, Path],
        ckpt_path: Path,
        device: str = "cpu",
        **tokenizer_args,
    ) -> None:
        super().__init__(model, device, **tokenizer_args)
        self.dense = None
        self.embeddings = self.model.get_input_embeddings()

        sd_enc, sd_proj = {}, {}
        ckpt = torch.load(ckpt_path, map_location=device)
        for k, v in ckpt["state_dict"].items():
            if k.endswith("embeddings.position_ids"):
                continue

            # remove prefix and dot
            # if k.startswith("query_encoder"):
            #     sd_enc[k[len("query_encoder") + 1 :]] = v
            if k.startswith("doc_encoder.model"):
                sd_enc[k[len("doc_encoder.model") + 1 :]] = v

        self.model.load_state_dict(sd_enc)
        if ckpt["hyper_parameters"].get("projection_size") is not None:
            self.projection = torch.nn.Linear(
                self.model.embedding_dimension,
                ckpt["hyper_parameters"]["projection_size"],
            )
            self.projection.load_state_dict(sd_proj)
            self.projection.eval()
        else:
            self.projection = None
        self.model.eval()

    def forward(self, batch: Dict[str, torch.LongTensor]) -> torch.Tensor:
        inputs = batch.input_ids
        lengths = (inputs != 0).sum(dim=1)
        sequences_emb = self.embeddings(inputs)

        # create a mask corresponding to sequence lengths
        _, max_len, emb_dim = sequences_emb.shape
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(-1, -1, emb_dim)

        # compute the mean for each sequence
        rep = torch.sum(mask * sequences_emb, dim=1) / lengths.unsqueeze(-1)
        return rep

    @property
    def embedding_dimension(self) -> int:
        return self.embeddings.embedding_dim

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **self.tokenizer_args,
        )
        inputs.to(self.device)

        with torch.no_grad():
            input_ids = inputs["input_ids"]
            lengths = (input_ids != 0).sum(dim=1)
            sequences_emb = self.embeddings(input_ids)

            # create a mask corresponding to sequence lengths
            _, max_len, emb_dim = sequences_emb.shape
            mask = torch.arange(max_len, device=lengths.device).unsqueeze(
                0
            ) < lengths.unsqueeze(-1)
            mask = mask.unsqueeze(-1).expand(-1, -1, emb_dim)

            # compute the mean for each sequence
            rep = torch.sum(mask * sequences_emb, dim=1) / lengths.unsqueeze(-1)
            return rep.detach().cpu().numpy()

        # with torch.no_grad():
        #     # Mean pooling: average the embeddings of all non-padding tokens.
        #     outputs = self.model(**inputs)
        #     embeddings = outputs.last_hidden_state
        #     attention_mask = inputs.attention_mask.unsqueeze(-1)
        #     avg_embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        #     return avg_embeddings.cpu().numpy()

# TODO [with Martijn]: Find the best perfoming up-to-date encoders and add them; also create new OPQ indixes.
