from pathlib import Path

import torch
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.index.disk import OnDiskIndex
from fast_forward.quantizer.nanopq import NanoPQ


# q_encoder = TCTColBERTQueryEncoder(
#     "castorini/tct_colbert-msmarco",
#     device="gpu" if torch.cuda.is_available() else "cpu"
# )
# ff_index = OnDiskIndex.load(
#     Path("/home/bvdb9/indices/ff_msmarco-v1-passage.tct_colbert.h5"),
#     query_encoder=q_encoder
# )

# in practice, a subset of the encoded corpus should be used as training vectors
ff_index = OnDiskIndex.load(
    Path("/home/bvdb9/indices/ff_msmarco-v1-passage.tct_colbert.h5")
)
training_vectors = ff_index.cut(300)


quantizer = NanoPQ(M=8, Ks=256)
quantizer.fit(training_vectors)

ff_index = OnDiskIndex(Path("ff_index_pq.h5"), quantizer=quantizer, overwrite=True)
