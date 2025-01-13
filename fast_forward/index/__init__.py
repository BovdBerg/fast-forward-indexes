"""
.. include:: ../docs/index.md
"""

import abc
import logging
import warnings
from enum import Enum
from time import perf_counter
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from fast_forward.encoder import Encoder
from fast_forward.quantizer import Quantizer
from fast_forward.ranking import Ranking

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Mode(Enum):
    """Enum used to set the ranking mode of an index."""

    PASSAGE = 1
    MAXP = 2
    FIRSTP = 3
    AVEP = 4


IDSequence = Sequence[Optional[str]]


class Index(abc.ABC):
    """Abstract base class for Fast-Forward indexes."""

    _query_encoder: Encoder = None
    _quantizer: Quantizer = None
    _verbose: bool = False

    def __init__(
        self,
        query_encoder: Encoder = None,
        quantizer: Quantizer = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
        verbose: bool = False,
        profiling: bool = False,
    ) -> None:
        """Create an index.

        Args:
            query_encoder (Encoder, optional): The query encoder to use. Defaults to None.
            quantizer (Quantizer, optional): The quantizer to use. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.MAXP.
            encoder_batch_size (int, optional): Encoder batch size. Defaults to 32.
            verbose (bool, optional): Print progress. Defaults to True.
            profiling (bool, optional): Log performance metrics. Defaults to False.
        """
        super().__init__()
        if query_encoder is not None:
            self.query_encoder = query_encoder
        self.mode = mode
        if quantizer is not None:
            self.quantizer = quantizer
        self._encoder_batch_size = encoder_batch_size
        self._verbose = verbose
        self._profiling = profiling
        warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

    def encode_queries(
        self,
        queries: Sequence[str],
        ranking: Ranking = None
    ) -> np.ndarray:
        """Encode queries.

        Args:
            queries (Sequence[str]): The queries to encode.

        Raises:
            RuntimeError: When no query encoder exists.

        Returns:
            np.ndarray: The query representations.
        """
        t0 = perf_counter()
        if self.query_encoder is None:
            raise RuntimeError("Index does not have a query encoder.")

        query_encoder = self._query_encoder
        if hasattr(query_encoder, 'ranking_in'):
            query_encoder.ranking_in = ranking

        result = []
        total_batches = (len(queries) + self._encoder_batch_size - 1) // self._encoder_batch_size
        for i in tqdm(
            range(0, len(queries), self._encoder_batch_size),
            desc="Encoding queries per batch",
            total=total_batches,
            disable=total_batches == 1 or not self._verbose,
        ):
            batch = queries[i : i + self._encoder_batch_size]
            result.append(self.query_encoder(batch))

        if self._profiling:
            LOGGER.info(f"encode_queries took {perf_counter() - t0:.5f} seconds for {len(queries)} queries")
        return np.concatenate(result)

    @property
    def query_encoder(self) -> Optional[Encoder]:
        """Return the query encoder if it exists.

        Returns:
            Optional[Encoder]: The query encoder (if any).
        """
        return self._query_encoder

    @query_encoder.setter
    def query_encoder(self, encoder: Encoder) -> None:
        """Set the query encoder.

        Args:
            encoder (Encoder): The new query encoder.
        """
        assert isinstance(encoder, Encoder)
        self._query_encoder = encoder

    @property
    def quantizer(self) -> Optional[Quantizer]:
        """Return the quantizer if it exists.

        Returns:
            Optional[Quantizer]: The quantizer (if any).
        """
        return self._quantizer

    @quantizer.setter
    def quantizer(self, quantizer: Quantizer) -> None:
        """Set the quantizer. This is only possible before any vectors are added to the index.

        Raises:
            RuntimeError: When the index is not empty.

        Args:
            quantizer (Quantizer): The new quantizer.
        """
        assert isinstance(quantizer, Quantizer)

        if len(self) > 0:
            raise RuntimeError("Quantizers can only be attached to empty indexes.")
        self._quantizer = quantizer
        quantizer.set_attached()

    @property
    def mode(self) -> Mode:
        """Return the ranking mode.

        Returns:
            Mode: The ranking mode.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: Mode) -> None:
        """Set the ranking mode.

        Args:
            mode (Mode): The new ranking mode.
        """
        assert isinstance(mode, Mode)
        self._mode = mode

    @abc.abstractmethod
    def _get_internal_dim(self) -> Optional[int]:
        """Return the dimensionality of the vectors (or codes) in the index (internal method).

        If no vectors exist, return None. If a quantizer is used, return the dimension of the codes.

        Returns:
            Optional[int]: The dimensionality (if any).
        """
        pass

    @property
    def dim(self) -> Optional[int]:
        """Return the dimensionality of the vector index.

        May return None if there are no vectors.

        If a quantizer is used, the dimension before quantization is returned.

        Returns:
            Optional[int]: The dimensionality (if any).
        """
        if self._quantizer is not None:
            return self._quantizer.dims[0]
        return self._get_internal_dim()

    @property
    @abc.abstractmethod
    def doc_ids(self) -> Set[str]:
        """Return all unique document IDs.

        Returns:
            Set[str]: The document IDs.
        """
        pass

    @property
    @abc.abstractmethod
    def psg_ids(self) -> Set[str]:
        """Return all unique passage IDs.

        Returns:
            Set[str]: The passage IDs.
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """The number of vectors in the index.

        Returns:
            int: The index size.
        """
        pass

    @abc.abstractmethod
    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence,
        psg_ids: IDSequence,
    ) -> None:
        """Add vector representations and corresponding IDs to the index.

        Document IDs may have duplicates, passage IDs are assumed to be unique. Vectors may be quantized.

        Args:
            vectors (np.ndarray): The representations, shape `(num_vectors, dim)` or `(num_vectors, quantized_dim)`.
            doc_ids (IDSequence): The corresponding document IDs.
            psg_ids (IDSequence): The corresponding passage IDs.
        """
        pass

    def add(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence = None,
        psg_ids: IDSequence = None,
    ) -> None:
        """Add vector representations and corresponding IDs to the index.

        Only one of `doc_ids` and `psg_ids` may be None. Individual IDs in the sequence may also be None,
        but each vector must have at least one associated ID.

        Document IDs may have duplicates, passage IDs must be unique.

        Args:
            vectors (np.ndarray): The representations, shape `(num_vectors, dim)`.
            doc_id (IDSequence, optional): The corresponding document IDs (may be duplicate). Defaults to None.
            psg_id (IDSequence, optional): The corresponding passage IDs (must be unique). Defaults to None.

        Raises:
            ValueError: When there are no document IDs and no passage IDs.
            ValueError: When the number of IDs does not match the number of vectors.
            ValueError: When the input vector and index dimensionalities don't match.
            ValueError: When a vector has neither a document nor a passage ID.
            RuntimeError: When items can't be added to the index for any reason.
        """
        if doc_ids is None and psg_ids is None:
            raise ValueError("At least one of doc_ids and psg_ids must be provided.")

        num_vectors, dim = vectors.shape

        if doc_ids is None:
            doc_ids = [None] * num_vectors
        if psg_ids is None:
            psg_ids = [None] * num_vectors
        if not len(doc_ids) == len(psg_ids) == num_vectors:
            raise ValueError("Number of IDs does not match number of vectors.")

        if self.dim is not None and dim != self.dim:
            raise ValueError(
                f"Input vector dimensionality ({dim}) does not match index dimensionality ({self.dim})."
            )

        for doc_id, psg_id in zip(doc_ids, psg_ids):
            if doc_id is None and psg_id is None:
                raise ValueError("Vector has neither document nor passage ID.")

        self._add(
            vectors if self.quantizer is None else self.quantizer.encode(vectors),
            doc_ids,
            psg_ids,
        )

    @abc.abstractmethod
    def _get_vectors(self, ids: Iterable[str]) -> Tuple[np.ndarray, List[List[int]]]:
        """Return:
            * A single array containing all vectors necessary to compute the scores for each document/passage.
            * For each document/passage (in the same order as the IDs), a list of integers (depending on the mode).

        The integers will be used to get the corresponding representations from the array.
        The output of this function depends on the current mode.
        If a quantizer is used, this function returns quantized vectors.

        Args:
            ids (Iterable[str]): The document/passage IDs to get the representations for.

        Returns:
            Tuple[np.ndarray, List[List[int]]]: The vectors and corresponding indices.
        """
        pass

    def _compute_scores(
        self, df: pd.DataFrame, query_vectors: np.ndarray
    ) -> pd.DataFrame:
        """Computes scores for a data frame.
        The input data frame needs a "q_no" column with unique query numbers.

        Args:
            df (pd.DataFrame): Input data frame.
            query_vectors (np.ndarray): All query vectors indexed by "q_no".

        Returns:
            pd.DataFrame: Data frame with computed scores.
        """
        t0 = perf_counter()
        # map doc/passage IDs to unique numbers (0 to n)
        id_df = df[["id"]].drop_duplicates().reset_index(drop=True)
        id_df["id_no"] = id_df.index

        # attach doc/passage numbers to data frame
        df = df.merge(id_df, on="id", suffixes=[None, "_"]).reset_index(drop=True)

        # get all required vectors from the FF index
        t1 = perf_counter()
        vectors, id_to_vec_idxs = self._get_vectors(id_df["id"].to_list())
        t2 = perf_counter()
        if self._profiling:
            LOGGER.info(f"_get_vectors took {t2 - t1:.5f} seconds for {len(vectors)} vectors")
        if self.quantizer is not None:
            vectors = self.quantizer.decode(vectors)
        t3 = perf_counter()
        if self._profiling:
            LOGGER.info(f"decoding took {t3 - t2:.5f} seconds for {len(vectors)} vectors")

        # compute indices for query vectors and doc/passage vectors in current arrays
        select_query_vectors = []
        select_vectors = []
        select_scores = []
        c = 0
        for id_no, q_no in zip(df["id_no"], df["q_no"]):
            vec_idxs = id_to_vec_idxs[id_no]
            select_vectors.extend(vec_idxs)
            select_scores.append(list(range(c, c + len(vec_idxs))))
            c += len(vec_idxs)
            select_query_vectors.extend([q_no] * len(vec_idxs))

        # compute all dot products (scores)
        q_reps = query_vectors[select_query_vectors]
        d_reps = vectors[select_vectors]
        scores = np.sum(q_reps * d_reps, axis=1)

        # select aggregation operation based on current mode
        if self.mode == Mode.MAXP:
            op = np.max
        elif self.mode == Mode.AVEP:
            op = np.average
        else:
            op = lambda x: x[0]

        def _mapfunc(i):
            scores_i = select_scores[i]
            if len(scores_i) == 0:
                return np.nan
            return op(scores[scores_i])

        # insert FF scores in the correct rows
        df["ff_score"] = df.index.map(_mapfunc)
        if self._profiling:
            LOGGER.info(f"_compute_scores took {perf_counter() - t3:.5f} seconds")
        return df

    def _early_stopping(
        self,
        df: pd.DataFrame,
        query_vectors: np.ndarray,
        cutoff: int,
        alpha: float,
        depths: Iterable[int],
    ) -> pd.DataFrame:
        """Compute scores with early stopping for a data frame.
        The input data frame needs a "q_no" column with unique query numbers.

        Args:
            df (pd.DataFrame): Input data frame.
            query_vectors (np.ndarray): All query vectors indexed by "q_no".
            cutoff (int): Cut-off depth for early stopping.
            alpha (float): Interpolation parameter.
            depths (Iterable[int]): Depths to compute scores at.

        Returns:
            pd.DataFrame: Data frame with computed scores.
        """
        # data frame for computed scores
        scores_so_far = None

        # [a, b] is the interval for which the scores are computed in each step
        a = 0
        for b in sorted(depths):
            if b < cutoff:
                continue

            # identify queries which do not meet the early stopping criterion
            if a == 0:
                # first iteration: take all queries
                q_ids_left = pd.unique(df["q_id"])
            else:
                # subsequent iterations: compute ES criterion
                q_ids_left = (
                    scores_so_far.groupby("q_id")
                    .filter(
                        lambda g: g["int_score"].nlargest(cutoff).iat[-1]
                        < alpha * g["score"].iat[-1] + (1 - alpha) * g["ff_score"].max()
                    )["q_id"]
                    .drop_duplicates()
                    .to_list()
                )
            LOGGER.info("depth %s: %s queries left", b, len(q_ids_left))

            # take the next chunk with b-a docs/passages for each query
            chunk = df.loc[df["q_id"].isin(q_ids_left)].groupby("q_id").nth(range(a, b))

            # stop if no pairs are left
            if len(chunk) == 0:
                break

            # compute scores for the chunk and merge
            out = self._compute_scores(chunk, query_vectors)[["orig_index", "ff_score"]]
            chunk_scores = chunk.merge(out, on="orig_index", suffixes=[None, "_"])

            # compute interpolated scores
            chunk_scores["int_score"] = (
                alpha * chunk_scores["score"] + (1 - alpha) * chunk_scores["ff_score"]
            )

            if scores_so_far is None:
                scores_so_far = chunk_scores
            else:
                scores_so_far = pd.concat(
                    [scores_so_far, chunk_scores],
                    axis=0,
                )

            a = b
        return scores_so_far.join(df, on="orig_index", lsuffix=None, rsuffix="_")

    def __call__(
        self,
        ranking: Ranking,
        early_stopping: int = None,
        early_stopping_alpha: float = None,
        early_stopping_depths: Iterable[int] = None,
        batch_size: int = None,
    ) -> Ranking:
        """Compute scores for a ranking.

        Args:
            ranking (Ranking): The ranking to compute scores for. Must have queries attached.
            early_stopping (int, optional): Perform early stopping at this cut-off depth. Defaults to None.
            early_stopping_alpha (float, optional): Interpolation parameter for early stopping. Defaults to None.
            early_stopping_depths (Iterable[int], optional): Depths for early stopping. Defaults to None.
            batch_size (int, optional): How many queries to process at once. Defaults to None.

        Returns:
            Ranking: Ranking with the computed scores.

        Raises:
            ValueError: When the ranking has no queries attached.
            ValueError: When early stopping is enabled but arguments are missing.
        """
        if not ranking.has_queries:
            raise ValueError("Input ranking has no queries attached.")
        if early_stopping is not None and (
            early_stopping_alpha is None or early_stopping_depths is None
        ):
            raise ValueError("Early stopping requires alpha and depths.")
        t0 = perf_counter()

        # get all unique queries and query IDs and map to unique numbers (0 to m)
        query_df = (
            ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
        )
        query_df["q_no"] = query_df.index
        df_with_q_no = ranking._df.merge(query_df, on="q_id", suffixes=[None, "_"])

        # early stopping splits the data frame, hence we need to keep track of the original index
        df_with_q_no["orig_index"] = df_with_q_no.index

        # batch encode queries
        query_vectors = self.encode_queries(list(query_df["query"]), ranking)

        def _get_result(df):
            if early_stopping is None:
                return self._compute_scores(df, query_vectors)
            return self._early_stopping(
                df,
                query_vectors,
                early_stopping,
                early_stopping_alpha,
                early_stopping_depths,
            )

        num_queries = len(query_df)
        if batch_size is None or batch_size >= num_queries:
            result = _get_result(df_with_q_no)
        else:
            # assign batch indices to query IDs
            df_with_q_no["batch_idx"] = (
                df_with_q_no.groupby("q_id").ngroup() / batch_size
            ).astype(int)

            chunks = []
            num_batches = int(num_queries / batch_size) + 1
            for batch_idx in tqdm(range(num_batches)):
                batch = df_with_q_no[df_with_q_no["batch_idx"] == batch_idx]
                chunks.append(_get_result(batch))
            result = pd.concat(chunks)

        result["score"] = result["ff_score"]
        ranking = Ranking(
            result,
            name="fast-forward",
            dtype=ranking._df.dtypes["score"],
            copy=False,
            is_sorted=False,
        )
        if self._profiling:
            LOGGER.info(f"__call__ in {perf_counter() - t0:.5f} seconds")
        return ranking

    @abc.abstractmethod
    def _batch_iter(
        self, batch_size: int
    ) -> Iterator[Tuple[np.ndarray, IDSequence, IDSequence]]:
        """Iterate over the index in batches (internal method).

        If a quantizer is used, the vectors are the quantized codes.
        When an ID does not exist, it must be set to None.

        Args:
            batch_size (int): Batch size.

        Yields:
            Tuple[np.ndarray, IDSequence, IDSequence]: Vectors, document IDs, passage IDs in batches.
        """
        pass

    def batch_iter(
        self, batch_size: int
    ) -> Iterator[Tuple[np.ndarray, IDSequence, IDSequence]]:
        """Iterate over all vectors, document IDs, and passage IDs in batches.
        IDs may be either strings or None.

        Args:
            batch_size (int): Batch size.

        Yields:
            Tuple[np.ndarray, IDSequence, IDSequence]: Batches of vectors, document IDs (if any), passage IDs (if any).
        """
        if self._quantizer is None:
            yield from self._batch_iter(batch_size)

        else:
            for batch in self._batch_iter(batch_size):
                yield self._quantizer.decode(batch[0]), batch[1], batch[2]

    def __iter__(
        self,
    ) -> Iterator[Tuple[np.ndarray, Optional[str], Optional[str]]]:
        """Iterate over all vectors, document IDs, and passage IDs.

        Yields:
            Tuple[np.ndarray, Optional[str], Optional[str]]: Vector, document ID (if any), passage ID (if any).
        """
        for vectors, doc_ids, psg_ids in self.batch_iter(2**9):
            yield from zip(vectors, doc_ids, psg_ids)
