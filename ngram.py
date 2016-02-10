from utils import iterlen, lines_iter, subsequences

from itertools import chain, repeat
from numpy import random, fromiter, float32

def ngrams_from_texts(texts, N):
    line_to_ngrams = lambda line: subsequences(line.split(), N)
    return chain.from_iterable(map(line_to_ngrams, texts))

def random_corpus_ngrams(corpus_path, N, number):
    with open(corpus_path) as corpus:
        ngrams = ngrams_from_texts(lines_iter(corpus), N)
        ngram_num = iterlen(ngrams)
        corpus.seek(0)

        if number > ngram_num:
            raise RuntimeError('Corpus not long enough')

        includes = repeat(True, number)
        excludes = repeat(False, ngram_num - number)
        flags = fromiter(chain(includes, excludes), dtype=bool)
        random.shuffle(flags)

        ngrams = ngrams_from_texts(lines_iter(corpus), N)
        for include, ngram in zip(flags, ngrams):
            if include:
                yield ngram
