from utils import compress, decompress, iterlen, lines_iter, subsequences, to_ndarray

from numpy import array
import os
import gzip
import subprocess
from pathlib import PurePath
import random
import string
from itertools import chain, repeat
from numpy import random, fromiter, zeros

def create_vocabulary(infile, outfile, mincount=5):
    vocab = dict()
    with gzip.open(infile, mode='rt') as f:
        for line in f:
            for word in line.split():
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

    vocab = sorted(vocab.items(), key=(lambda item: item[1]), reverse=True)
    with gzip.open(outfile, mode='xt') as f:
        for word, count in vocab:
            if count >= 5:
                print(word, count, file=f)
            else:
                break

def load_vocabulary(infile):
    vocab = dict()
    with gzip.open(infile, mode='rt') as f:
        for line in f:
            line = line.split()
            assert len(line) == 2
            word = line[0]
            count = int(line[1])

            vocab[word] = count
    
    return vocab
        

def create_word_embedding(infile, outfile, min_count=5, size=100, downsample=True, estimator='skipgram', negative=5):
    def run_on(actual_infile):
        assert estimator in ['skipgram', 'cbow']
        use_cbow = estimator == 'cbow'
        sample = 10e-3
        if not downsample:
            sample = 1
        subprocess.run([ 'tools/word2vec/word2vec' \
                       , '-train', actual_infile \
                       , '-output', outfile \
                       , '-min-count', str(min_count) \
                       , '-size', str(size) \
                       , '-sample', str(downsample) \
                       , '-negative', str(negative) \
                       , '-cbow', str(int(use_cbow)) ])

    compress_flag = False
    outfilep = PurePath(outfile)
    if outfilep.suffix == '.gz':
        outfile = str(outfilep.parent / outfilep.stem)
        compress_flag = True
    
    infilep = PurePath(infile)
    if infilep.suffix == '.gz':
        rndstr = ''.join(random.choice(list(string.ascii_uppercase + string.digits)) for _ in range(10))
        tmp_filename = '/tmp/' + infilep.stem + rndstr
        try:
            decompress(infile, keep=True, outfile=tmp_filename)
            run_on(tmp_filename)
        finally:
            if os.path.exists(tmp_filename):
                os.remove(tmp_filename)
    else:
        run_on(infile)

    if compress_flag:
        compress(outfile)

class WordEmbedding:
    def __init__(self, file_name):
        f = gzip.open(file_name, mode='rt')

        firstline = f.readline()
        [word_num_str, dimension_str] = firstline.split()
        word_num = int(word_num_str)
        dimension = int(dimension_str)

        self.word_indices = {}
        self.values = zeros((word_num, dimension))

        word_index = 0
        for line in f:
            items = line.split()

            word = items[0]
            self.word_indices[word] = word_index

            str_vec = items[1 : ]
            assert len(str_vec) == dimension
            for i in range(0, dimension):
                self.values[word_index, i] = float(str_vec[i])

            word_index += 1
        
        assert len(self.word_indices) == word_num

    def __getitem__(self, param):
        if type(param) is str:
            return self.values[self.word_indices[param]]
        else:
            return to_ndarray(chain.from_iterable(map(self.__getitem__, param)))

def random_corpus_ngrams(corpus_path, N, predicate=None):
    ngrams = []
    with gzip.open(corpus_path, mode='rt') as corpus:
        lines = lines_iter(corpus)
        line_to_ngrams = lambda line: map(list, subsequences(line.split(), N))
        ngrams = chain.from_iterable(map(line_to_ngrams, lines))
        
        if predicate is not None:
            ngrams = filter(predicate, ngrams)

        ngrams = list(ngrams)

    random.shuffle(ngrams)
    return ngrams
