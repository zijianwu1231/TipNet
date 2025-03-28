import numpy as np

def mean_euclidean_distance(preds, gts):
    """
    Compute the mean euclidean distance between the predicted keypoints and the ground truth keypoints
    """
    dist = np.sqrt(np.sum((preds - gts) ** 2, axis=1))
    return np.mean(dist)