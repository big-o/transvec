========
Transvec
========

Translate word embeddings across models.

This package includes a python implementation of the the method outlined in `MLS2013`_,
which allows for word embeddings from one model to be translated to the vector space of
another model.

This allows you to compare word embeddings for different languages, avoiding the
expense and complexity of training bilingual models. With ``transvec``, you can simply
use pre-trained `Word2Vec <https://radimrehurek.com/gensim/models/word2vec.html>`_
models for different languages to measure the similarity of words in different
languages.

Installation
------------

.. code-block:: bash

    pip install transvec

Example
-------

Let's say we want to study a corpus of text that contains a mix of Russian and English.
``gensim`` has pre-trained models for both languages:

.. code-block:: python

    >>> import gensim.downloader
    >>> ru_model = gensim.downloader.load("word2vec-ruscorpora-300")
    >>> en_model = gensim.downloader.load("glove-wiki-gigaword-300")

Let's say you don't have the resources to train a model that understands both languages
well (and you probably don't). It would be nice to take advantage of the knowledge we
have in these two pre-trained models instead. Let's use the Russian model to compare
Russian words and the English model to compare English words:

.. code-block:: python

    >>> en_model.similar_by_word("king", 1)
    [('queen', 0.6336469054222107)]

    >>> ru_model.similar_by_word("царь_NOUN", 1) # "king"
    [('царица_NOUN', 0.7304918766021729)] # "queen"

As advertised, the word models kind correctly find words with a similar meaning. What if
we now wish to compare words from different languages?

.. code-block:: python

    >>> ru_model.similar_by_word("king", 1)
    Traceback (most recent call last):
        ...
    KeyError: "word 'king' not in vocabulary"

It doesn't work, because the Russian model was not trained on English words. We could
of course convert our word to a vector in the English model, and then look for the most
similar vector in our corpus:

.. code-block:: python

    >>> king_vector = en_model.get_vector("king")
    >>> ru_model.similar_by_vector(king_vector, 1)
    [('непроизводительный_ADJ', 0.21217751502990723)]

Our result (which appropriately means `"unproductive"`) makes no sense at all. The
meaning is nothing like our input word. Why did this happen? Because the "king" vector
is defined by the vector space of the English model, which has nothing to do with the
vector space of the Russian model. Output from the two models is completely
uncomparable. To remedy this, we must translate the vector from the `source` space
(English in the above case) into the `target` space (Russian).

This is where ``transvec`` can help you. By providing pairs of words in the source
language along with their translation into the target language, ``transvec`` can train a
model that will translate the vector for a word in the source language to a vector in
the target language:

.. code-block:: python

    >>> from transvec.transformers import TranslationWordVectorizer

    >>> train = [
    ...     ("king", "царь_NOUN"), ("tsar", "царь_NOUN"),
    ...     ("man", "мужчина_NOUN"), ("woman", "женщина_NOUN")
    ... ]

    >>> bilingual_model = TranslationWordVectorizer(en_model, ru_model).fit(train)

For the convenience of English speakers, we have defined English to be our target
language in this case. This will create a model that can take inputs in both languages,
but produce output in English.

.. note::
    The models in our example both produce vectors with the same number of dimensions:
    this is not required by the TranslationWordVectorizer, and models with different
    dimensionality may be mixed. The output of the TranslationWordVectorizer will
    always have the same dimensionality as the target model.

Now we can make comparisons across both languages:

.. code-block:: python

    >>> bilingual_model.similar_by_word("царь_NOUN", 1) # "tsar"
    [('king', 0.8043200969696045)]

If the provided word does not exist in the source corpus, but does exist in the target
corpus, the model will fall back to using the target language's vector:

.. code-block:: python

    >>> bilingual_model.similar_by_word("king", 1)
    [('queen', 0.6336469054222107)]

We can also get sensible results for words that weren't in our training set (the
accuracy will depend on how representative your training data is):

.. code-block:: python

    >>> bilingual_model.similar_by_word("царица_NOUN", 1) # "queen"
    [('king', 0.7763221263885498)]
    
Note that you can provide regularisation parameters to the `TranslationWordVectorizer`
to help improve these results if you need to.


Extra features
--------------

Bulk vectorisation
++++++++++++++++++

For convenience, ``TranslationWordVectorizer`` also implements the `scikit-learn`
``Transformer`` API, allowing you to vectorise large sets of data in a pipeline easily.
If you provide a 2D matrix of words, it will assume each row represents a single
document and produce a single vector for each row, which is just the mean of all of the
word vectors in the document (this is a simple, cheap way of approximating document
vectors when your documents contain multiple languages).

Multilingual models
+++++++++++++++++++

The example above converts a single source language into a target language. You can
however train a model that recognises multiple source languages instead. Simply provide
more than one source language when you initialise the model. Source languages will be
prioritised in the order you define them. Note that your training data must now contain
word tuples rather than word pairs; the order of the languages matching the order of
your models.

How does it work?
-----------------

The full details are outlined in `MLS2013`_, but basically it's just Ordinary Least
Squares. The paper notes that a linear relationship exists between the vector spaces of
monolingual models, meaning that a simple translation matrix can be used to convert a
vector from its native vector space to a similar point in a target vector space, placing
it close to words in the target language with similar meanings.

Unlike the original paper, ``transvec`` uses ridge regression rather than OLS to derive
this translation matrix: this is to help prevent overfitting if you only have a small
set of training word pairs. If you want to use OLS instead, simply set the
regularization parameter (``alpha``) to zero in the ``TranslationWordVectorizer``
constructor.

References
----------

.. [MLS2013] `Tomas Mikolov, Quoc V Le, Ilya Sutskever. 2013.
        Exploiting Similarities among Languages for Machine Translation
        <https://arxiv.org/pdf/1309.4168.pdf>`_
