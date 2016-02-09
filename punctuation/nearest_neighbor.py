from utils import subsequences
from numpy import fromiter, float32
from numpy.linalg import norm
from itertools import chain

def nearest_neighbor(before, after, train, embedding, debug=False):
    tovec = embedding.__getitem__

    def ngram_to_vec(ngram):
        vecs = map(tovec, ngram)
        return fromiter(chain.from_iterable(vecs), dtype=float32)


    best_comma_ngram = None
    best_comma_dist = float('inf')
    best_nocomma_ngram = None
    best_nocomma_dist = float('inf')

    comma_ngram = list(chain(before, [','], after))
    nocomma_ngram = list(chain(before, after))

    comma_vec = ngram_to_vec(comma_ngram)
    nocomma_vec = ngram_to_vec(nocomma_ngram)

    for i, text in enumerate(train):
        words = text.split()

        for ngram in subsequences(words, length=len(comma_ngram)):
            try:
                if ngram[len(before)] == ',':
                    dist = norm(comma_vec - ngram_to_vec(ngram))

                    if dist < best_comma_dist:
                        best_comma_ngram = ngram
                        best_comma_dist = dist
            except KeyError:
                pass

        for ngram in subsequences(words, length=len(nocomma_ngram)):
            try:
                dist = norm(nocomma_vec - ngram_to_vec(ngram))

                if dist < best_nocomma_dist:
                    best_nocomma_ngram = ngram
                    best_nocomma_dist = dist
            except KeyError as exc:
                pass

    return { 'best_comma_ngram' : best_comma_ngram \
           , 'best_comma_dist' : best_comma_dist \
           , 'best_nocomma_ngram' : best_nocomma_ngram \
           , 'best_nocomma_dist' : best_nocomma_dist }


