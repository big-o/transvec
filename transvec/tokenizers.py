import nltk
import re
import string


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


__all__ = ["EnRuTokenizer"]


PUNCTUATION_REGEX = re.compile("[{0}]".format(re.escape(string.punctuation)))


def strip_punc(s, all=False):
    """Removes punctuation from a string.

    :param s: The string.
    :param all: Remove all punctuation. If False, only removes punctuation from
        the ends of the string.
    """
    if all:
        return PUNCTUATION_REGEX.sub("", s.strip())
    else:
        return s.strip().strip(string.punctuation)


class EnRuTokenizer(nltk.tokenize.NLTKWordTokenizer):
    """
    Tokenises text (including ngrams), lower-cases, removes punctuation and
    defeats both English and Russian stopwords.

    Since the gensim word embedding model for Russian includes POS tags, this tokenizer
    also append tags to the end of any words that contain cyrillic characters.
    """

    _STOPWORDS = nltk.corpus.stopwords.words("english") + nltk.corpus.stopwords.words(
        "russian"
    )

    # From https://github.com/akutuzov/universal-pos-tags/blob
    #      /4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map
    _TAGMAP = {
        "A": "ADJ",
        "ADV": "ADV",
        "ADVPRO": "ADV",
        "ANUM": "ADJ",
        "APRO": "DET",
        "COM": "ADJ",
        "CONJ": "SCONJ",
        "INTJ": "INTJ",
        "NONLEX": "X",
        "NUM": "NUM",
        "PART": "PART",
        "PR": "ADP",
        "S": "NOUN",
        "SPRO": "PRON",
        "UNKN": "X",
        "V": "VERB",
    }

    _CYRILLIC = set(
        "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
    )

    def is_cyrillic(self, text):
        return len(set(text) & self._CYRILLIC) > 0

    def tokenize(self, text, max_ngram_size=1):
        tokens = [
            tok
            for tok in super().tokenize(strip_punc(text.lower(), all=True))
            if tok not in self._STOPWORDS
        ]
        tokens = [
            (tok, self._TAGMAP.get(tag, "UNKN"))
            if self.is_cyrillic(tok)
            else (tok, "UNKN")
            for tok, tag in nltk.pos_tag(tokens, lang="rus")
        ]
        tokens = [
            f"{tok}_{tag}" if tag not in ("NONLEX", "UNKN") else tok
            for tok, tag in tokens
        ]

        ngrams = []
        if max_ngram_size > 1:
            for n in range(2, max_ngram_size + 1):
                ngrams.append(
                    list(["_".join(ngram) for ngram in nltk.ngrams(tokens, n)])
                )

        for ngs in ngrams:
            tokens += ngs

        return tokens
