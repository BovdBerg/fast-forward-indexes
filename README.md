# Re-Ranking Using Fast-Forward Indexes

## Pre-requisites
This implementation extends upon "Efficient Neural Ranking using Forward Indexes" by J. Leonhardt, et al: [Paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511955), [Slides](https://mrjleo.github.io/slides/2023-phd/), [GitHub](https://github.com/mrjleo/fast-forward-indexes).


## Installation
Install the package via `pip`:
```bash
conda create -y -n ff python=3.12.3
conda activate ff

git clone https://github.com/BovdBerg/fast-forward-indexes.git
cd fast-forward-indexes
pip install -e .
```


## Instructions
For a detailed description of the program arguments, run ```python path/to/file.py -h``` or inspect the `parse_args` method of that file.

### Training AvgEmb Query Estimator
```bash
python usage/train_avg_emb.py \
    --index_path path/to/index_path.h5
    --n_docs 10
```
The main other options are:
- `--cache_n_docs X` determines how large the cache files will be (n_docs per topic). X also decides the maximum n_docs setting for the trained AvgEmb estimator.
- `--ckpt_path` can be added to continue training from an earlier checkpoint.
- `--samples X` to train on only the first X samples.

### Interpolated Dual-Encoder Re-ranking w/ AvgEmb Query Estimator
```bash
python usage/retrieve_and_re_rank.py \
    --storage mem \
    --device cpu \
    --index_path path/to/index_path.h5 \
    --ckpt_path path/to/ckpt_path.ckpt \
    --n_docs 10 \
    --val_pipelines all \
    --test_datasets "irds:msmarco-passage/trec-dl-2019/judged" "irds:msmarco-passage/trec-dl-2020/judged"
    --eval_metrics "nDCG@10" "RR(rel=2)" "AP(rel=2)"
```
