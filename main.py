"""main file for testing and final integrating."""
from util import get_prior

TRAIN_LABEL = 'Data/traininglabels'
TRAIN_DATA = 'Data/trainingimages'
TEST_LABEL =  'Data/testlabels'
TEST_DATA = 'Data/testimages'
WIDTH = 28
HEIGHT = 28
TOTAL_PIXEL = WIDTH*HEIGHT
TOTAL_DIG = 10

prior = get_prior(TRAIN_LABEL)
print(prior)
# read test data
testdata = open(TEST_DATA, 'r')
samples = []
curr_list = []
for line in testdata.readlines():
    for char in line:
        if char == ' ':
            curr_list.append(0)
        elif char == '+':
            curr_list.append(0.5)
        elif char == '#':
            curr_list.append(1)

    if len(curr_list) == TOTAL_PIXEL:
        samples.append(curr_list)
        curr_list = []

print(samples[0])