"""The function that contains all the necessary method for Bayes."""
from math import log
import numpy as np
import matplotlib.pyplot as plt


TRAIN_LABEL = 'Data/traininglabels'
TRAIN_DATA = 'Data/trainingimages'
TEST_LABEL = 'Data/testlabels'
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
    big_dic = dict()

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
        big_dic[index] = (y_[index], curr_max_likelihood)

    return y_, big_dic


def get_accuracy(y, y_, length):
    correct = 0
    for i in range(length):
        if y[i] == y_[i]:
            correct += 1
    return correct/length


def confusion_matrix(y, y_, length):
    conf_m = np.zeros((TOTAL_DIG, TOTAL_DIG))
    for number in range(TOTAL_DIG):
        number_counter = 0
        for index in range(length):
            if y[index] == number:
                number_counter += 1
            if y[index] == number and y_[index] == number:
                conf_m[number][number] += 1
            elif y[index] == number and y_[index] != number:
                conf_m[number][y_[index]] += 1
        conf_m[number] = conf_m[number] * 100 / number_counter

    return conf_m


# function to calculate odds ratio
def odds_ratio(pair, p_prob):
    a, b = pair
    F_a = p_prob[a]
    F_b = p_prob[b]
    odd_ratio = []
    for index in range(TOTAL_PIXEL):
        odd_ratio.append(log(F_b[index]/F_a[index]))
    return odd_ratio


def displayHeatMap(firstProb, secondProb, oddRatio):
    """displayHeatMap.

    DESCRIPTION: The function that we use to display the three different
        heatmap based on the given three lists

    INPUTS:
        1.firstProb: A list that holds all the pixel probabilities of the first number
        2.secondProb: A list that holds all the pixel probabilities of the second number
        3.oddRatio: A list that holds all the calculated odd ratio on each pixel

    OUTPUT:
        None, simply display the heatmap
    """
    firstRevised = []
    secondRevised = []
    oddRevised = []

    for row in range(HEIGHT):
        tempFirst = []
        tempSecond = []
        tempOdd = []
        for unit in range(WIDTH):
            tempFirst.append(firstProb[row * WIDTH + unit])
            tempSecond.append(secondProb[row * WIDTH + unit])
            tempOdd.append(oddRatio[row * WIDTH + unit])
        firstRevised.append(tempFirst)
        secondRevised.append(tempSecond)
        oddRevised.append(tempOdd)

    # So far, we have reformatted all the list into 2D list
    # Plot the first graph

    figure = plt.figure()
    # get a new subplot image
    first = figure.add_subplot(1, 3, 1)
    plt.imshow(firstRevised, cmap='jet', interpolation='nearest', vmin=-4, vmax=0)
    first.set_title("First Digit")
    plt.colorbar()

    second = figure.add_subplot(1, 3, 2)
    plt.imshow(secondRevised, cmap='jet', interpolation='nearest', vmin=-4, vmax=0)
    second.set_title("Second Digit")
    plt.colorbar()

    odd = figure.add_subplot(1, 3, 3)
    plt.imshow(oddRevised, cmap='jet', interpolation='nearest', vmin=-3, vmax=1.7)
    odd.set_title("Odd Ratio")
    plt.colorbar()
    plt.show()

    """
    f, sub = plt.subplots(1, 3)

    plot1 = sub[0].imshow(firstRevised)
    plot1.colorbar()
    # Plot the second graph
    sub[1].imshow(secondRevised)


    # Plot the Odd Ratio Graph
    sub[2].imshow(oddRevised)
    plt.show()
    """


def reformat_disjoint(size0, size1, inlist):
    if WIDTH % size1 == 0 and HEIGHT % size0 == 0:
        G_all = ['*'] * int(TOTAL_PIXEL/(size0*size1))
    else:
        print("Invalid Patch Size")
        return False

    for row, line in enumerate(inlist):
        for col, char in enumerate(line):
            row_index = int(row/size0)
            col_index = int(col/size1)
            curr_tuple = (char,)
            if G_all[int(row_index*WIDTH/size1) + col_index] == '*':
                G_all[int(row_index*WIDTH/size1) + col_index] = curr_tuple
            else:
                G_all[int(row_index*WIDTH/size1) + col_index] += curr_tuple

    return G_all

def disjoint_helper(size0, size1, inlist, wid, hei):
    if wid % size1 == 0 and hei % size0 == 0:
        G_all = ['*'] * int((wid*hei)/(size0*size1))
    else:
        print("Invalid Patch Size")
        return False

    for row, line in enumerate(inlist):
        for col, char in enumerate(line):
            row_index = int(row/size0)
            col_index = int(col/size1)
            curr_tuple = (char,)
            if G_all[int(row_index*wid/size1) + col_index] == '*':
                G_all[int(row_index*wid/size1) + col_index] = curr_tuple
            else:
                G_all[int(row_index*wid/size1) + col_index] += curr_tuple

    return G_all
def row_expand_helper(line, size1, row, inlist):
    curr_expand_row = []
    for col, char in enumerate(line):
        if col + size1 <= len(inlist[0]):
            for i in range(size1):
                curr_expand_row.append(inlist[row][col + i])
    return curr_expand_row


def reformat_overlap(size0, size1, inlist):
    expanded_list = []

    for row in range(len(inlist)):
        if row+size0 <= len(inlist):
            for j in range(size0):
                expanded_list.append(row_expand_helper(inlist[row+j], size1, row+j, inlist))

    print("EXPAND LIST: ", expanded_list)
    G_all = disjoint_helper(size0, size1, expanded_list, len(expanded_list[0]), len(expanded_list))

    return G_all


# main
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
y_, proto = estimate(samples, p_prob, prior)
accuracy = get_accuracy(y, y_, len(samples))
conf_m = confusion_matrix(y, y_, len(samples))
# get prototypical data
for i in range(len(samples)):
    correct_value = y[i]
    predict_value, _ = proto[i]
    if correct_value != predict_value:
        proto.pop(i)
protolist = [0] * 10
for key, value in proto.items():
    curr_protolist = []
    for number in range(TOTAL_DIG):
        if value[0] == number:
            curr_protolist.append(value[1])
    if protolist[value[0]] == 0:
        protolist[value[0]] = curr_protolist
    else:
        protolist[value[0]].extend(curr_protolist)

prototypical_dict = dict()
for index, likelihoods in enumerate(protolist):
    max = np.amax(likelihoods)
    min = np.amin(likelihoods)
    for key, value in proto.items():
        if value[1] == max:
            maxi = key
        if value[1] == min:
            mini = key
    prototypical_dict[index] = (maxi, mini)
# prepare for visualizing prototypical data
with open(TEST_DATA, 'r') as test_data:
    testdata_list = [y.strip('\n') for y in test_data.readlines()]

proto_path = open('prototypical.txt', 'w')
for key, values in prototypical_dict.items():
    proto_path.write("Current Number: %s" % key)
    proto_path.write("\nMaximum Likelihood: \n")
    for row in range(values[0] * HEIGHT, (values[0]+1)*HEIGHT):
        proto_path.write("%s\n" % testdata_list[row])
    proto_path.write("\nMinimum Likelihood: \n")
    for row in range(values[1] * HEIGHT, (values[1]+1)*HEIGHT):
        proto_path.write("%s\n" % testdata_list[row])
    proto_path.write("\n")

# get four most confused pairs
confusion_values = dict()
for i in range(conf_m.shape[0]):
    for j in range(conf_m.shape[1]):
        if i != j:
            confusion_values[(i, j)] = conf_m[i][j]

most_confused_paris = sorted(confusion_values, key=confusion_values.get, reverse=True)[:4]

log_prob = dict()
# We need to change the p_prob value
for index in range(TOTAL_DIG):
    for each in range(TOTAL_PIXEL):
        if index not in log_prob:
            log_prob[index] = [0]*TOTAL_PIXEL
        log_prob[index][each] = log(p_prob[index][each])

# Now we should have all log_likelihood value in this list of probability
"""
for i in range(len(most_confused_paris)):
    curr_odd_ratio = odds_ratio(most_confused_paris[i], p_prob)
    displayHeatMap(log_prob[most_confused_paris[i][0]],
                   log_prob[most_confused_paris[i][1]], curr_odd_ratio)
"""

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print('Test Labels: ', y)
print('Predicted Labels: ', y_)
print('Accuracy: ', accuracy * 100, '%')
print('Confusion Matrix: \n', conf_m)
print("Most Confused Pairs: ", most_confused_paris)
print('Prototypical: (maximum posterior, minimum posterior)\n', prototypical_dict)

inlist = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
print("DEBUG:", reformat_overlap(2,3,inlist))