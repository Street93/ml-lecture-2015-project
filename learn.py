from utils import to_ndarray

from sklearn.neighbors import NearestNeighbors
from numpy import array


def splitsave_XY(XY):
    Y = []
    X = []
    
    for x, y in XY:
        X.append(to_ndarray(x))
        Y.append(y)

    return array(X), array(Y)


def nn_classifier(train_XY, k=1):
    train_X, train_Y = splitsave_XY(train_XY)

    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')

    if len(train_X.shape) == 1:
        train_X = train_X.reshape((-1, 1))

    nn = nn.fit(train_X)

    def classify(test_X):
        test_X = to_ndarray(test_X)

        response_Y = []
        for neighbor_indices in nn.kneighbors(test_X, return_distance=False):
            neighbor_labels = [train_Y[index] for index in neighbor_indices]
            best_label = max(set(train_Y), key=lambda label: neighbor_labels.count(label))
            response_Y.append(best_label)
        return to_ndarray(response_Y, dtype=int)

    return classify
