import numpy as np
import os
from tempfile import NamedTemporaryFile
import subprocess

def prepare_for_word2vec(string):
    chars_with_padding = '.!?:,;"()'
    for c in chars_with_padding:
        string = string.replace(c, ' {} '.format(c))
    return string

def create_word_embedding(texts, outfile_name):
    tmpf = NamedTemporaryFile(mode='w', delete=False)
    try:
        for text in texts:
            print(prepare_for_word2vec(text), file=tmpf)
        tmpf.close()
        subprocess.run([ 'tools/word2vec/word2vec'
                       , '-train', tmpf.name
                       , '-output', outfile_name ])
    finally:
        tmpf.close()
        # os.remove(tmpf.name)

class WordEmbedding:
    def __init__(self, file_name):
        f = open(file_name)

        firstline = f.readline()
        [word_num_str, dimension_str] = firstline.split()
        word_num = int(word_num_str)
        dimension = int(dimension_str)

        self.word_indices = {}
        self.values = np.zeros((word_num, dimension))

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

    def wordOfIndex(self, index):
        for w, i in self.word_indices.items():
            if i == index:
                return w

    def __getitem__(self, param):
        if type(param) is str:
            return self.values[self.word_indices[param]]
        if type(param) is np.ndarray:
            assert param.shape == (self.values.shape[1],)
            best_index = None
            best_distance = float('inf')
            for i in range(0, self.values.shape[0]):
                distance = np.linalg.norm(self.values[i] - param)
                if distance < best_distance:
                    best_distance = distance
                    best_index = i

            return self.wordOfIndex(best_index)
    
    def nearest(self, word):
        value = self[word]

        distances = np.zeros(self.values.shape[0])
        for i in range(0, len(distances)):
            distances[i] = np.linalg.norm(self.values[i] - value)

        for i in np.argsort(distances):
            yield self.wordOfIndex(i)
