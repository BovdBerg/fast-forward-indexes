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

<!-- TODO: improve installation instructions -->
```bash
pip install fast-forward-indexes
```


## Instructions
<!-- TODO [final]: update run script in readme -->
Re-ranking can be done by running this code:
```bash
python usage/rerank.py \
    --index_path path/to/index_path.h5 \
    --in_memory \
    --test_dataset irds:msmarco-passage/trec-dl-2019/judged \
    --test_sparse_ranking_path path/to/sparse_ranking.tsv \
```

For a detailed description of the program arguments:
- Look in the `usage/rerank.py::parse_args` method.
- Or run: ```python usage/rerank.py -h```
