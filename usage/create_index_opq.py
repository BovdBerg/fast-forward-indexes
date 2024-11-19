import logging
from pathlib import Path

import torch

from fast_forward import Indexer, OnDiskIndex
from fast_forward.encoder.transformer import CoCondenserQueryEncoder
from fast_forward.quantizer.nanopq import NanoOPQ

logging.basicConfig(level=logging.DEBUG)


ff_index_source = OnDiskIndex.load(
    Path("/home/bvdb9/indices/msmarco_passage/ff_index_CoCondenser.h5"),
)

device = "cuda" if torch.cuda.is_available() else "cpu"
ff_index_target = OnDiskIndex(
    Path("/home/bvdb9/indices/msmarco_passage/ff_index_CoCondenser.h5"),
    CoCondenserQueryEncoder("Luyu/co-condenser-marco-retriever", device=device),
    overwrite=True,
    init_size=len(ff_index_source),
)

Indexer(
    ff_index_target,
    quantizer=NanoOPQ(96, 2048),
    batch_size=100000,
    quantizer_fit_batches=2,
).from_index(ff_index_source)
