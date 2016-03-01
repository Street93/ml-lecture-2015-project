from learn import nn_classifier

from collections import namedtuple
from functools import partial

# from learn import nn_classifer, qda_classifier, gaussian_kde_classifier, \
#                   tophat_kde_classifier, random_forest_classifier


class DataConfig(namedtuple('DataConfig', \
    'corpus dimension estimator negative downsampling ngram_size')):

    def corpuspath(self):
        return 'data/corpora/{}.gz'.format(self.corpus)
    def embeddingpath(self):
        return 'data/embeddings/{}-{}-{}-{}-{}.gz'.format( \
                self.corpus, self.dimension, self.estimator, self.negative, self.downsampling)
    def vocabularypath(self):
        return 'data/vocabularies/{}.gz'.format(self.corpus)
    def ngrampaths(self):
        template = 'data/ngrams/{}-{}.gz'
        ngrams = template.format(self.corpus, self.ngram_size)
        n1grams = template.format(self.corpus, self.ngram_size + 1)
        return (ngrams, n1grams)
    
class RunConfig(namedtuple('RunConfig', 'dataconfig algorithm punctuation train_size')):

    def resultpath(self):
        ngrampath = Path(self.dataconfig.ngrampaths()[0])
        return 'data/ngram/{}-{}-{}-{}.gz'.format( ngrampath.stem, self.algorithm \
                                                 , self.punctuation, self.train_size)

    def classify_trainer(self):
        m = { 'nn': nn_classifier \
            , 'nn-5': partial(nn_classifier, k=5) }
        return m[self.algorithm]

dataconfigs = [DataConfig(corpus, dimension, estimator, negative, downsampling, ngram_size) \
                  for corpus in ['spiegel-full'] \
                  for dimension in [10, 50, 100, 500, 1000] \
                  for estimator in ['skipgram', 'cbow'] \
                  for negative in [5, 15] \
                  for downsampling in [True, False] \
                  for ngram_size in [4, 10]]

runconfigs = [RunConfig(dataconfig, algorithm, punctuation, train_size) \
                 for dataconfig in dataconfigs \
                 for algorithm in ['nn'] \
                 for punctuation in [',', '.', '!', '?', '.,!?'] \
                 for train_size in [1000]] # , 5000, 10000, 50000, 10000]]
