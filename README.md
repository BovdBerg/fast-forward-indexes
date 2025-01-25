# Re-Ranking Using Fast-Forward Indexes

## Pre-requisites
This implementation extends upon Fast-Forward Indexes by J. Leonhardt, et al.

This original research is described by:

- Original Paper "Efficient Neural Ranking using Forward Indexes"  
by J. Leonhardt, et al.  
[Link](https://dl.acm.org/doi/abs/10.1145/3485447.3511955)

- Extended Paper "Efficient Neural Ranking using Forward Indexes and Lightweight Encoders" (pre-print)  
by J. Leonhardt, et al.  
[Link](https://arxiv.org/abs/2311.01263)

- Accompanying slide deck by by J. Leonhardt  
[Link](https://mrjleo.github.io/slides/2023-phd/).

- [GitHub repository](https://github.com/mrjleo/fast-forward-indexes)

    - [Docs](https://mrjleo.github.io/fast-forward-indexes/docs)

> [!IMPORTANT]
> As this library is still in its early stages, the API is subject to change!

## Installation

Install the package via `pip`:

<!-- TODO [final]: improve installation instructions -->
```bash
conda create -y -n ff python=3.12.3
conda activate ff

git clone https://github.com/BovdBerg/fast-forward-indexes.git
cd fast-forward-indexes
pip install -e .
```


## Instructions
Feel free to email me at `bvdb98@gmail.com` for an example index (for `index_path`) and checkpoint file (for `ckpt_path`).

For a detailed description of the program arguments, run ```python path/to/file.py -h``` or inspect the `parse_args` method of that file.

### Training AvgEmbQueryEstimator
```bash
python usage/train_avg_emb.py \
    --index_path path/to/index_path.h5
    --n_docs 10
```
The main important options are:
- `--ckpt_path` can be added to continue training from an earlier checkpoint.
- `--samples X` to train on only the first X (integer) samples.
- `--num_workers X` to use X (int) cpu cores for dataloading. More workers speeds up training.

### Re-ranking
<!-- TODO [final]: update run script in readme -->
```bash
python usage/retrieve_and_re_rank.py \
    --index_path path/to/index_path.h5 \
    --ckpt_path path/to/ckpt_path.ckpt \
    --device cpu \
    --n_docs 10 \
    --val_pipelines all \
    --test_datasets "irds:msmarco-passage/trec-dl-2019/judged"
```

### Profiling
```bash
python usage/retrieve_and_re_rank.py \
    [...]
    --storage mem \
    --profiling \
    --device=cpu
```
