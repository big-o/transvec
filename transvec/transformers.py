import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import Ridge
from typing import Optional, Union


__all__ = ["TranslationWordVectorizer"]


class TranslationWordVectorizer(TransformerMixin, BaseEstimator):
    """
    A word -> word vector transformer. Any words that do not exist in the target
    model's vocabulary can still be converted to a comparable word embedding as long as
    they exist in one of the source models' vocabularies.

    If a word does not exist in the target vocabulary, it will be will be searched for
    in each of the source vocabularies in the order they were provided in the
    constructor; as soon as a match is found, the word's embedding in that vector
    space will be converted into an embedding in the target space.

    Conversions are done via a translation matrix, which is calculated from the least
    squares method (with optional L2 regularisation) using known words with their
    translations as training data.

    This class also implements the scikit-learn Transformer API, so you may use it
    to convert arrays of words and sentences into word embeddings (in the target
    model's vector space).

    Parameters
    ==========

    target : :class:`gensim.models.keyedvectors.Word2VecKeyedVectors`
        The word embedding model for the target language (language to translate *to*).
        For best results, this should be the model with the largest vocabulary.

    sources : :class:`gensim.models.keyedvectors.Word2VecKeyedVectors`
        The word embedding model(s) for the source language(s) (language to translate
        *from*). Models will be prioritised in the order they are provided to the
        constructor: any word provided to the model will be searched for in the
        vocabulary of the target, followed by the first source, then the second source
        and so on until all source models are exhausted.

    alpha : float (default = 1.0)
        Regularisation strength for calculating translation matrices. An alpha of zero
        is equivalent to an ordinary least squares fit. See
        :class:`sklearn.linear_model.Ridge` for more details.

    max_iter : int, optional
        Max iterations for ridge regression. See :class:`sklearn.linear_model.Ridge`
        for more details.

    tol : float (default = 1e-3)
        Precision of the least squares fit. See :class:`sklearn.linear_model.Ridge`
        for more details.

    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'} (default = 'auto')
        Solver to use for ridge regression. See :class:`sklearn.linear_model.Ridge`
        for more details.

    missing : str (default = "raise")
        Action to perform when words in training data are not found in their
        respective models. Can be one of {"ignore", or "raise"}. If "ignore", the word
        pair will not be used in the fitting.

    random_state : int or RandomState instance, optional
        Random seed for ridge regression. See :class:`sklearn.linear_model.Ridge` for
        more details.

    Examples
    ========

    >>> import gensim.downloader
    >>> ru_model = gensim.downloader.load("word2vec-ruscorpora-300")
    >>> en_model = gensim.downloader.load("glove-wiki-gigaword-100")
    >>> word_pairs = [
    ...     ("king", "царь_NOUN"), ("tsar", "царь_NOUN"),
    ...     ("man", "мужчина_NOUN"), ("woman", "женщина_NOUN")
    ... ]
    >>> model = TranslationWordVectorizer(en_model, ru_model).fit(word_pairs)
    >>> [word for word, score in model.similar_by_word("царь_NOUN")][:3]
    ['tsar', 'king', 'emperor']

    References
    ==========

    .. [1] `Tomas Mikolov, Quoc V Le, Ilya Sutskever. 2013.
            Exploiting Similarities among Languages for Machine Translation
            <https://arxiv.org/pdf/1309.4168.pdf>`_
    """

    def __init__(
        self,
        target: "gensim.models.keyedvectors.Word2VecKeyedVectors",
        *sources: "gensim.models.keyedvectors.Word2VecKeyedVectors",
        alpha: float = 1.0,
        max_iter: Optional[int] = None,
        tol: float = 0.001,
        solver: str = "auto",
        missing: str = "raise",
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        if len(sources) == 0:
            raise ValueError(
                "__init__() missing 1 required positional argument: 'sources'"
            )

        self.target = target

        sample = target[next(iter(target.vocab.keys()))]
        self.dim = len(sample)
        self.dtype = sample.dtype

        self.sources = sources
        self.missing = missing

        self._reg = Ridge(
            fit_intercept=False,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state,
        )

    def fit(self, X) -> "TranslationWordVectorizer":
        """
        Train this object to translate word vectors from the source model's
        vector space into the destination model's vector space.

        A translation matrix will be generated by least-squares fitting to vectors
        for words that have the same meaning in each model's space.

        Parameters
        ==========

        X : list-like
            List of (target_language, source1_language, ...) tuples of words with
            their translations. Translations must be provided for every source, and
            all words and translations must exist in their respective models to be
            used for training.
        """

        srcmats = [list()] * len(self.sources)
        dstmat = []
        for dst, *srcs in np.asarray(X):
            if dst not in self.target:
                if self.missing == "raise":
                    raise KeyError(dst)
                elif self.missing == "ignore":
                    continue

            skip = False
            for src, source in zip(srcs, self.sources):
                if src not in source:
                    if self.missing == "raise":
                        raise KeyError(src)
                    elif self.missing == "ignore":
                        skip = True
                        break

            if skip:
                continue

            for src, source, srcmat in zip(srcs, self.sources, srcmats):
                srcmat.append(source.get_vector(src))
                dstmat.append(self.target.get_vector(dst))

        if len(srcmats[0]) == 0:
            raise ValueError("Unable to fit: no valid training words were provided.")

        self.trans_ = []
        for i in range(len(self.sources)):
            trans = clone(self._reg).fit(dstmat, srcmats[i]).coef_
            self.trans_.append(trans)

        return self

    def get_vector(self, word: str) -> np.ndarray:
        """
        Get the entity's representations in vector space, as a 1D numpy array.

        Parameters
        ==========

        entity : str
            Identifier of the entity to return the vector for.

        Returns
        =======

        numpy.ndarray
            Vector for the specified entity.

        Raises
        ======

        KeyError
            If the given entity identifier doesn't exist.
        """

        if word in self.target:
            return self.target.get_vector(word)
        else:
            for source, trans in zip(self.sources, self.trans_):
                if word in source:
                    vec = source.get_vector(word)
                    return np.dot(vec, trans)

        raise KeyError(word)

    def __getitem__(self, word: str) -> np.ndarray:
        return self.get_vector(word)

    def __contains__(self, word: str) -> bool:
        if word in self.target:
            return True

        for src in self.sources:
            if word in src:
                return True

        return False

    def similar_by_vector(
        self, vector: np.ndarray, topn: int = 10, restrict_vocab: Optional[int] = None
    ) -> np.ndarray:
        return self.target.similar_by_vector(
            vector, topn=topn, restrict_vocab=restrict_vocab
        )

    def similar_by_word(
        self, word: str, topn: int = 10, restrict_vocab: Optional[int] = None
    ) -> np.ndarray:
        if word in self.target:
            return self.target.similar_by_word(
                word, topn=topn, restrict_vocab=restrict_vocab
            )
        else:
            vector = self.get_vector(word)
            return self.target.similar_by_vector(
                vector, topn=topn, restrict_vocab=restrict_vocab
            )

    def transform(self, X) -> np.ndarray:
        """
        Convert words into embeddings. If a 2D array (sentences) is provided,
        a mean embedding for the whole sentence will be provided.
        """

        X = np.asarray(X)

        if X.ndim == 0:
            vectors = self.get_vector(X.item())
            return vectors
        elif X.ndim > 2:
            raise ValueError("X cannot have more than 2 dimensions.")

        if len(X) == 0:
            return np.array([], dtype=np.float32)

        vectors = []
        for words in X:
            if isinstance(words, str):
                words = [words]

            words = [word for word in words if word in self]
            if len(words) == 0:
                vectors.append(np.array([np.nan] * self.dim, dtype=self.dtype))
            else:
                vector = np.mean(
                    [self.get_vector(word) for word in words if word in self], axis=0
                )
                vectors.append(vector)

        return np.array(vectors)

    def fit_transform(self, X, y=None, **fit_params):
        raise NotImplementedError()
