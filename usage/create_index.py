from pathlib import Path
import torch
from fast_forward.encoder.transformer import CoCondenserDocumentEncoder, CoCondenserQueryEncoder
from fast_forward import Indexer
from fast_forward.index.disk import OnDiskIndex
import pyterrier as pt


### PARAMETERS
dataset_name = "msmarco_passage"


if not pt.started():
    pt.init()

dataset = pt.get_dataset(dataset_name)
out_dir = Path(f"/home/bvdb9/indices/{dataset_name}")
out_dir.mkdir(parents=True, exist_ok=True)
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("dataset info:", dataset.info_url())
print("out_dir:", out_dir)
print("device_name:", device_name)

q_encoder = CoCondenserQueryEncoder(
    "Luyu/co-condenser-marco-retriever",
    device=device_name
)
d_encoder = CoCondenserDocumentEncoder(
    "Luyu/co-condenser-marco-retriever",
    device=device_name,
)

ff_index = OnDiskIndex(
    Path(f"{out_dir}/ff_index_CoCondenser.h5"),
    query_encoder=q_encoder,
    overwrite=True
)
print("ff_index created")

def docs_iter():
    for d in dataset.get_corpus_iter():
        yield {"doc_id": d["docno"], "text": d["text"]}

ff_indexer = Indexer(ff_index, d_encoder, batch_size=4)
ff_indexer.from_dicts(docs_iter())

print("Done.")
