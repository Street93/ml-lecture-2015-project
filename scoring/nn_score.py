from numpy import array
from numpy.linalg import norm

def make_nn_score(yes_samples, no_samples):
    yes_samples = array(yes_samples)
    no_samples = array(no_samples)
    
    def score(sample):
        dist = lambda s: norm(s - sample)
        yes_dists = map(dist, yes_samples)
        no_dists = map(dist, no_samples)

        return min(yes_dists) - min(no_dists)

    return score
