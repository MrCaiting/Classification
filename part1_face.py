"""The function that contains all the necessary method for Bayes."""
from math import log
import random
import numpy as np
import time

TRAIN_LABEL = 'FaceData/facedatatrainlabels'
TRAIN_DATA = 'FaceData/facedatatrain'
TEST_LABEL =  'FaceData/facedatatestlabels'
TEST_DATA = 'FaceData/facedatatest'
WIDTH = 60
HEIGHT = 70
TOTAL_PIXEL = WIDTH*HEIGHT
TOTAL_DIG = 2
TOTAL_IMG = 451

# Frequently changed global variable
SIZE_0 = 4
SIZE_1 = 2
MODE = ["disjoint", "overlapping"]
K = 0.1
V = pow(2, SIZE_0*SIZE_1)
mode = MODE[1]


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


def train(training_data, training_label, size0, size1, patch_mode):
    """train.

    DESCRIPTION: The function is used to read all the training data and calculate
        probability value corresponds to each pixel of each number

    INPUTS:
        1. training_data: the file path of the training data
        2. training_label: the file path of the training label
        3. patch_mode: tell the method which grouping method we'd lie to use
    OUTPUT:
        p_prob: a dictionary that holds all the probability value of each
            pixel of each number
        total_group: the total grouping amount that we have under this patching method
    """
    # Calculate the total value based on patch_mode
    if (patch_mode == "disjoint"):
        total = int(WIDTH*HEIGHT / (size0*size1))
    elif (patch_mode == "overlapping"):
        total = (WIDTH - size1 + 1)*(HEIGHT - size0 + 1)
    else:
        print("Invalid Patching Method Value!")
        return False

    # dictionary that holds the count of any pixel
    p_counts = dict()
    # dictionary that holds all the count of label
    label_counts = dict()

    #######################################
    # dictionary that holds all the probability
    p_prob = dict()
    #######################################

    # Get all the training labels as a list
    with open(training_label, 'r') as train_l:
        t_labels = [int(x.strip('\n')) for x in train_l.readlines()]

    # Get all the training data as a list
    with open(training_data, 'r') as train_d:
        image = [y.strip('\n') for y in train_d.readlines()]

    reformat_start = time.clock()
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
            p_counts[t_labels[index]] = [0] * total

        # Starting converting this_image[] into list of 0 or 1
        for row, line in enumerate(this_image):
            templine = []

            for col, char in enumerate(line):
                # if it is a hashtag, we add 1 to it
                if char == "#":
                    templine.append(1)

                # if it is a plus sign, which denote the border, we still give 1
                elif char == "+":
                    templine.append(1)

                # otherwise, we assaign 0
                else:
                    templine.append(0)
            this_image[row] = templine

        """
        So far, we have a list of the current image filled with either 0 or 1
          Need call reformatting function to group pixels in tuple as desired
        """
        # Chose patching function as requested
        if (patch_mode == "disjoint"):
            G_temp = reformat_disjoint(size0, size1, this_image)
        elif (patch_mode == "overlapping"):
            G_temp = reformat_overlap(size0, size1, this_image)
        else:
            print("How the heck did you get here??")
            return False
        ################################################
        # Testing Line:
        if (len(G_temp) != total):
            print("We have unmatched G_temp length!!")
            return False
        ################################################
        """ The old training counting method
        for sub_index, g_ij in enumerate(G_temp):
            count_one = 0
            temp_total = len(g_ij)
            for element in g_ij:
                if (element == 1):
                    count_one += 1/temp_total
                else:
                    count_one += 0
            # Update the count in the p_counts dictionary
            p_counts[t_labels[index]][sub_index] += count_one
        """
        for sub_index, g_ij in enumerate(G_temp):
            # Check if we already have a sub dictionary here
            if p_counts[t_labels[index]][sub_index] == 0:
                #group_dict = dict()
                #group_dict[g_ij] = 1
                p_counts[t_labels[index]][sub_index] = {g_ij: 1}

            elif g_ij not in p_counts[t_labels[index]][sub_index]:
                p_counts[t_labels[index]][sub_index][g_ij] = 1

            # If not, simply updating the dictionary
            else:
                p_counts[t_labels[index]][sub_index][g_ij] += 1

    reformat_time = time.clock() - reformat_start
    print("Reformatting Time: ", reformat_time, " seconds")

    train_start = time.clock()
    # have all the information, we start calculating the probability
    for num in range(TOTAL_DIG):

        this_l_count = label_counts[num]

        # initialize the probability term in the dict
        p_prob[num] = [dict() for x in range(total)]
        for pix in range(total):
            n_pix_dict = p_counts[num][pix]

            for key, value in n_pix_dict.items():
                p_pix = (value + K) / (this_l_count + K*V)
                p_prob[num][pix][key] = p_pix

    train_time = time.clock() - train_start
    print("Training Time: ", train_time, " seconds")
    # print(len(p_prob[0]))
    return p_prob, label_counts, total


def estimate(samples, p_prob, prior, label_counts, total):
    """estimate.

    DESCRIPTION: Predict label of the input samples using MAP decision rule.
    :param
        1. sample: samples in test data
        2. p_prob: pixel probabilities matrix
        3. prior: prior values
    :return: y_: predicted labels of the sample data
    """
    y_ = ['*'] * len(samples)
    big_dic = dict()
    # print('probability Dict: ', p_prob)
    test_start = time.clock()
    for index, sample in enumerate(samples):
        curr_max_likelihood = None
        # print("DEBUG:", len(sample))
        for number in range(0, TOTAL_DIG):
            log_likelihood = log(prior[number])

            for pixel in range(total):
                curr_pixel = sample[pixel]  # a tuple
                if (curr_pixel in p_prob[number][pixel]):
                    curr_prob = p_prob[number][pixel][curr_pixel]
                else:
                    curr_prob = K / (label_counts[number] + K*V)
                # log_likelihood += temp * log(curr_prob) + (1-temp) * log(1-curr_prob)
                log_likelihood += log(curr_prob)
                # update the max log likelihood and the predicted label for current sample
            if curr_max_likelihood is None or log_likelihood > curr_max_likelihood:
                curr_max_likelihood = log_likelihood
                y_[index] = number
        big_dic[index] = (y_[index], curr_max_likelihood)
    print("Testing Time: ", time.clock() - test_start, " seconds")
    return y_, big_dic


def get_accuracy(y, y_, length):
    correct = 0
    for i in range(length):
        if y[i] == y_[i]:
            correct += 1
    return correct/length


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
    # print("dfdfdf", len(inlist[0]))
    for row in range(len(inlist)):
        if row+size0 <= len(inlist):
            for j in range(size0):
                expanded_list.append(row_expand_helper(inlist[row+j], size1, row+j, inlist))

    # print(len(expanded_list), len(expanded_list[0]))
    G_all = disjoint_helper(size0, size1, expanded_list, len(expanded_list[0]), len(expanded_list))
    # print(len(G_all))
    return G_all


def read_test_data(size0, size1, pmode):
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
            curr_list = np.resize(curr_list, (HEIGHT, WIDTH))
            # print("CURR:", curr_list.shape)
            if (pmode == "disjoint"):
                samples.append(reformat_disjoint(size0, size1, curr_list))
                curr_list = []
            elif (pmode == "overlapping"):
                samples.append(reformat_overlap(size0, size1, curr_list))
                curr_list = []
            else:
                print("Wrong Method!")
                return False

    return samples

####################################################
# main
# read test labels


print('The Current Method: ', mode)
print('Patching Size: ', SIZE_0, ' x ', SIZE_1)

testlabels = open(TEST_LABEL, 'r')
y = [int(x.strip('\n')) for x in testlabels.readlines()]

# read test data
samples = read_test_data(SIZE_0, SIZE_1, mode)
# get prior
prior = get_prior(TRAIN_LABEL)

# train data
p_prob, label_counts, total = train(TRAIN_DATA, TRAIN_LABEL, SIZE_0, SIZE_1, mode)
# print("TOTAL:", total)
# start estimating
y_, _ = estimate(samples, p_prob, prior, label_counts, total)
accuracy = get_accuracy(y, y_, len(samples))


np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print('Accuracy: ', accuracy * 100, '%')
print('Test Labels: ', y)
print('Predicted Labels: ', y_)
