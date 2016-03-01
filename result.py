from collections import namedtuple
from copy import copy

class ClassificationHistogramBin(namedtuple('ClassificationHistogramBin', \
                                            'truth prediction number')):
    def to_jso(self):
        return dict(self._asdict())
    @staticmethod
    def from_jso(jso):
        return Bin(**jso)

class ClassificationHistogram(list):
    @staticmethod
    def from_labels(labels):
        B = ClassificationHistogramBin
        labels = list(labels)
        h = ClassificationHistogram()
        for l1 in labels:
            for l2 in labels:
                h.append(B(truth=l1, prediction=l2, number=0))

        return h

    def labels(self):
        return set((b.truth for b in self))

    def to_jso(self):
        return [b.to_jso() for b in self]

    @staticmethod
    def from_jso(jso):
        return ClassificationHistogram((ClassificationHistogramBin.from_jso(jsobin) for jsobin in jso))


class ClassificationResult(namedtuple('ClassificationResult', \
                                      'traintime testtime priors histogram')):

    def to_jso(self):
        self = copy(self)
        jso = dict(self._asdict())
        jso['histogram'] = self.histogram.to_jso()
        return jso

    @staticmethod
    def from_jso(jso):
        jso = copy(jso)
        jso['histogram'] = ClassificationHistogram.from_jso(jso['histogram'])
        return Classificationresult(jso)

    def correctness(self):
        accum = 0
        for label in self.histogram.labels():
            correct_num = next((num for b in histogram if b.truth == label if b.prediction ==label))
            total_num = sum((num for b in histogram if b.truth == label))
            accum += self.priors[label] * (correct_num / total_num)
        return accum

    def error(self):
        return 1 - self.correctness()

    def prior_correctness(self):
        return max(self.priors)

    def prior_error(self):
        return 1 - self.prior_correctness()

    def error_quotient_to_prior(self):
        return self.error() / self.prior_error()
