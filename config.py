from learn import nn_classifier, qda_classifier
from utils import hyphen_format

from collections import namedtuple
from functools import partial
from pathlib import Path
from copy import copy

class DataConfig(namedtuple('DataConfig', \
    'corpus dimension estimator negative downsampling ngram_size')):

    def corpuspath(self):
        return 'data/corpora/{}.gz'.format(self.corpus)
    def embeddingpath(self):
        return 'data/embeddings/{}.gz'.format( \
                hyphen_format((self.corpus, self.dimension, self.estimator, \
                              self.negative, self.downsampling)))
    def vocabularypath(self):
        return 'data/vocabularies/{}.gz'.format(self.corpus)
    def ngrampaths(self):
        template = 'data/ngrams/{}-{}.gz'
        ngrams = template.format(self.corpus, self.ngram_size)
        n1grams = template.format(self.corpus, self.ngram_size + 1)
        return (ngrams, n1grams)
    
    def to_jso(self):
        return dict(self._asdict())
    @staticmethod
    def from_jso(jso):
        return DataConfig(**jso)
    
class RunConfig(namedtuple('RunConfig', 'algorithm punctuation train_size')):
    def classify_trainer(self):
        m = { 'nn': nn_classifier \
            , 'nn-5': partial(nn_classifier, k=5) \
            , 'qda': qda_classifier }
        return m[self.algorithm]
    
    def to_jso(self):
        return dict(self._asdict())
    @staticmethod
    def from_jso(jso):
        return RunConfig(**jso)
    
def resultpath(dataconfig, runconfig):
    data_str = hyphen_format(dataconfig)
    run_str = hyphen_format(runconfig)

    return 'data/results/{}.gz'.format(hyphen_format((data_str, run_str)))

dataconfigs = [DataConfig(corpus, dimension, estimator, negative, downsampling, ngram_size) \
                  for corpus in ['spiegel-full'] \
                  for dimension in [10, 50, 100, 500, 1000] \
                  for estimator in ['skipgram', 'cbow'] \
                  for negative in [5, 15] \
                  for downsampling in [True, False] \
                  for ngram_size in [4, 10]]

runconfigs = [RunConfig(algorithm, punctuation, train_size) \
                 for algorithm in ['nn', 'nn-5', 'qda'] \
                 for punctuation in [',', '.', '!', '?', '.,!?'] \
                 for train_size in [1000]] # , 5000, 10000, 50000, 10000]]
