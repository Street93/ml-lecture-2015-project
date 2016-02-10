from utils import iterlen, lines_iter, subsequences

from itertools import chain, repeat
from numpy import random, fromiter

def random_corpus_ngrams(corpus_path, N, number, predicate=None):
    with open(corpus_path) as corpus:
        def iter_ngrams():
            corpus.seek(0)
            line_to_ngrams = lambda line: subsequences(line.split(), N)
            lines = lines_iter(corpus)
            ngrams = chain.from_iterable(map(line_to_ngrams, lines))
            if predicate:
                ngrams = map(list, ngrams)
                ngrams = filter(predicate, ngrams)
            return ngrams
            
        ngram_num = iterlen(iter_ngrams())

        if number > ngram_num:
            raise RuntimeError('Corpus not long enough')

        includes = repeat(True, number)
        excludes = repeat(False, ngram_num - number)
        flags = fromiter(chain(includes, excludes), dtype=bool)
        random.shuffle(flags)

        ngrams = [list(ngram) for (include, ngram) in zip(flags, iter_ngrams()) if include]
        return ngrams
