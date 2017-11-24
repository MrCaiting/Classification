"""The function that contains all the necessary method for Bayes."""
from math import log
import random

TRAIN_LABEL = 'Data/traininglabels'
TRAIN_DATA = 'Data/trainingimages'
TEST_LABEL =  'Data/testlabels'
TEST_DATA = 'Data/testimages'
WIDTH = 28
HEIGHT = 28
TOTAL_PIXEL = WIDTH*HEIGHT
TOTAL_DIG = 10
TOTAL_IMG = 4000

K = 0.1
V = 2

def get_prior(training_label):
    """get_prior.

    DESCRIPTION: Calculating the prior by using the given training
        data. Specifically, the training label file.

        P(class) here is essentailly the occurance of a class
        (or digit) over the total training samples. In this
        way, we are able to fulfill the probability distribution
        requirement since all prior will be added up to 1.

    INPUT:
        training_label: The training label file that is given for
            this MP

    OUTPUT:
        prior: A dictionary that contains all the calculated P(class)
    """
    # A dictionary to hold all the caculated prior
    prior = dict()
    # A dictionary to keep track of the occurance of each class
    num_label = dict()

    with open(training_label, 'r') as train:
        # The file contains all string type numbers, need to be converted

        t_labels = [int(x.strip('\n')) for x in train.readlines()]

    # Iterating through all the labels to fill up the two dicts
    for label in t_labels:

        # If we have the key already, increment the count
        if label in num_label:
            num_label[label] += 1
        # If not, we create one with 1
        else:
            num_label[label] = 1

    for i in range(TOTAL_DIG):
        prior[i] = float(num_label[i]) / TOTAL_IMG

    return prior


def train(training_data, training_label):
    """train.

    DESCRIPTION: The function is used to read all the training data and calculate
        probability value corresponds to each pixel of each number

    INPUTS:
        1. training_data: the file path of the training data
        2. training_label: the file path of the training label

    OUTPUT:
        p_prob: a dictionary that holds all the probability value of each
            pixel of each number
    """
    # dictionary that holds the count of any pixel
    p_counts = dict()
    # dictionary that holds all the probability
    p_prob = dict()
    # dictionary that holds all the count of label
    label_counts = dict()

    with open(training_label, 'r') as train_l:
        t_labels = [int(x.strip('\n')) for x in train_l.readlines()]

    with open(training_data, 'r') as train_d:
        image = [y.strip('\n') for y in train_d.readlines()]

    for index in range(TOTAL_IMG):
        # starting reading every single picture
        this_image = []
        for i in range(index*HEIGHT, (index+1)*HEIGHT):
            this_image.append(image[i])
        # this_image[] now has all the lines of the current image

        # if this is the first time that we see this number, update the count
        if t_labels[index] not in label_counts:
            label_counts[t_labels[index]] = 1
        else:
            label_counts[t_labels[index]] += 1

        # if this number we have not seen before, we need to initialize its dict term
        if t_labels[index] not in p_counts:
            p_counts[t_labels[index]] = [0] * TOTAL_PIXEL

        # create a seperate counter to help us iterate through the image
        temp_count = 0
        for line in this_image:
            for char in line:
                # if it is a hashtag, we add one to it
                if char == "#":
                    p_counts[t_labels[index]][temp_count] += 1
                # if it is a plus sign, which denote the border, we assign 0.5
                elif char == "+":
                    p_counts[t_labels[index]][temp_count] += 1
                # otherwise, we do move on without doing anything
                else:
                    p_counts[t_labels[index]][temp_count] += 0
                temp_count += 1

    # have all the information, we start calculating the probability
    for num in range(TOTAL_DIG):

        this_l_count = label_counts[num]
        # print("label count", label_counts[0])
        # initialize the probability term in the dict
        p_prob[num] = [0] * TOTAL_PIXEL
        for pix in range(TOTAL_PIXEL):
            n_pix = p_counts[num][pix]
            p_pix = (n_pix + K) / (this_l_count + K*V)
            p_prob[num][pix] = p_pix

    return p_prob


def estimate(samples, p_prob, prior):
    """estimate.
    Predict label of the input samples using MAP decision rule.
    :param sample: samples in test data
            p_prob: pixel probabilities matrix
            prior: prior values
    :return: y_: predicted labels of the sample data
    """
    y_ = ['*'] * len(samples)

    for index, sample in enumerate(samples):
        curr_max_likelihood = None
        for number in range(0,TOTAL_DIG):
            log_likelihood = log(prior[number])
            for pixel in range(TOTAL_PIXEL):
                curr_pixel = sample[pixel]
                curr_prob = p_prob[number][pixel]
                # print(curr_prob, number, pixel)
                log_likelihood += curr_pixel * log(curr_prob) + (1-curr_pixel) * log(1-curr_prob)
                # update the max log likelihood and the predicted label for current sample
            if curr_max_likelihood is None or log_likelihood > curr_max_likelihood:
                curr_max_likelihood = log_likelihood
                y_[index] = number

    return y_


def get_accuracy(y, y_, length):
    correct = 0
    for i in range(length):
        if y[i] == y_[i]:
            correct += 1
    return correct/length


# read test labels
testlabels = open(TEST_LABEL, 'r')
y = [int(x.strip('\n')) for x in testlabels.readlines()]
# read test data
testdata = open(TEST_DATA, 'r')
samples = []
curr_list = []
for line in testdata.readlines():
    for char in line:
        if char == ' ':
            curr_list.append(0)
        elif char == '+':
            curr_list.append(1)
        elif char == '#':
            curr_list.append(1)

    if len(curr_list) == TOTAL_PIXEL:
        samples.append(curr_list)
        curr_list = []

prior = get_prior(TRAIN_LABEL)
p_prob = train(TRAIN_DATA, TRAIN_LABEL)
y_ = estimate(samples, p_prob, prior)
accuracy = get_accuracy(y, y_, len(samples))

print('Test Labels: ', y)
print('Predicted Labels: ', y_)
print('Accuracy: ', accuracy * 100, '%')