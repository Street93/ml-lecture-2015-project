from collections import namedtuple

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
    
class RunConfig(namedtuple('RunConfig', 'dataconfig algorithm punctuation')):

    def resultpath(self):
        ngrampath = Path(self.dataconfig.ngrampaths()[0])
        return 'data/ngram/{}-{}-{}.gz'.format(ngrampath.stem, self.algorithm, self.punctuation)

dataconfigs = [DataConfig(corpus, dimension, estimator, negative, downsampling, ngram_size) \
                  for corpus in ['spiegel-full'] \
                  for dimension in [10, 50, 100, 500, 1000] \
                  for estimator in ['skipgram', 'cbow'] \
                  for negative in [5, 15] \
                  for downsampling in [True, False] \
                  for ngram_size in [4, 10]]

runconfigs = [RunConfig(dataconfig, algorithm, punctuation) \
                 for dataconfig in dataconfigs \
                 for algorithm in [] \
                 for punctuation in [',', '.', '!', '?', '.,!?']]
