"""main file for testing and final integrating."""
from util import get_prior

TRAIN_LABEL = 'Data/traininglabels'

prior = get_prior(TRAIN_LABEL)
print(prior)
