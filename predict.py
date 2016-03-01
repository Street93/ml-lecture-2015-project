from utils import random_round_robin
from texttools import WordEmbedding
from learn import splitsave_XY

from itertools import repeat, starmap, islice
import gzip
from copy import copy

def iter_ngrams(dataconfig):
    paths = dataconfig.ngrampaths()
    def path_to_ngrams(path):
        f = gzip.open(path, mode='rt')
        for line in f:
            yield line.split()

    return tuple(map(path_to_ngrams, paths))

def iter_XY(runconfig):
    dataconfig = runconfig.dataconfig

    embedding = WordEmbedding(dataconfig.embeddingpath())
    (ngrams, n1grams) = iter_ngrams(dataconfig)

    data = random_round_robin(zip(ngrams, repeat(True)), zip(n1grams, repeat(False)))

    def prepare_gram(gram, isngram):
        if isngram:
            return (embedding[gram], 0)

        l = len(gram)
        assert l % 2 == 1

        middle = l // 2

        if gram[middle] not in runconfig.punctuation:
            return None

        label = runconfig.punctuation.index(gram[middle]) + 1

        skipgram = copy(gram)
        del skipgram[middle]

        return (embedding[skipgram], label)
    
    XY = starmap(prepare_gram, data)
    XY = filter(None, XY)

    return XY

def run_analysis(runconfig):
    dataconfig = runconfig.dataconfig

    XY = iter_XY(runconfig)

    train_XY = islice(XY, runconfig.train_size)
    classifier = runconfig.classify_trainer()(train_XY)

    test_XY = islice(XY, 2000)
    test_X, test_Y = splitsave_XY(test_XY)

    response_Y = classifier(test_X)
    for test_y, response_y in zip(test_Y, response_Y):
        print(test_y, response_y)
