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

### Training AvgEmbQueryEstimator
```bash
python usage/train_avg_emb.py \
    --index_path path/to/index_path.h5
    --n_docs 10
```
The main options are:
- `--ckpt_path` can be added to continue training from an earlier checkpoint.
- `--samples X` to train on only the first X samples.

### Re-ranking
```bash
python usage/retrieve_and_re_rank.py \
    --index_path path/to/index_path.h5 \
    --ckpt_path path/to/ckpt_path.ckpt \
    --device cpu \
    --n_docs 10 \
    --val_pipelines all \
    --test_datasets "irds:msmarco-passage/trec-dl-2019/judged"
```
