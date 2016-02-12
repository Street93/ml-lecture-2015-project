from utils import to_ndarray

from numpy import array, ones, zeros, concatenate, dot, fromiter, float32
from numpy.linalg import norm, lstsq
from scipy import stats

def train_nn_score(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples, dtype=float32)
    no_samples = to_ndarray(no_samples, dtype=float32)
    
    def score(sample):
        sample = to_ndarray(sample, dtype=float32)
        dist = lambda s: norm(s - sample)
        yes_dists = map(dist, yes_samples)
        no_dists = map(dist, no_samples)

        return min(yes_dists) - min(no_dists)

    return score

def train_lstsq_score(yes_samples, no_samples):
    yes_a = to_ndarray(yes_samples, dtype=float32)
    yes_b = ones(yes_a.shape[0])

    no_a = to_ndarray(no_samples, dtype=float32)
    no_b = -ones(no_a.shape[0])

    a = concatenate((yes_a, no_a))
    b = concatenate((yes_b, no_b))

    (coeff_vec, _, _, _) = lstsq(a, b)

    def score(sample):
        return dot(coeff_vec, to_ndarray(sample, dtype=float32))

    return score

def train_lda_classifier(yes_samples, no_samples):
    yes_samples = to_ndarray(yes_samples, dtype=float32)
    no_samples = to_ndarray(no_samples, dtype=float32)

    (yes_loc, yes_scale) = stats.norm.fit(yes_samples)
    (no_loc, no_scale) = stats.norm.fit(no_samples)

    yes_prior = len(yes_samples) / (len(yes_samples) + len(no_samples))
    no_prior = 1 - yes_prior

    def classifier(sample):
        sample = to_ndarray(sample)
        yes_score = stats.norm.pdf(sample, loc=yes_loc, scale=yes_scale) * yes_prior
        no_score = stats.norm.pdf(sample, loc=no_loc, scale=no_scale) * no_prior

        if yes_score > no_score:
            return True
        else:
            return False

    return classifier

# def train_density_tree(yes_samples, no_samples):
