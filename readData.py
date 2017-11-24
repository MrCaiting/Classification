"""The file that handles all the data files reading."""
from util import TOTAL_IMG, TOTAL_DIG

WIDTH = 28
HEIGHT = 28
TOTAL_PIXEL = WIDTH*HEIGHT


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

    for index in TOTAL_IMG:
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
                    p_counts[t_labels[index]][temp_count] += 0.5
                # otherwise, we do move on without doing anything
                else:
                    p_counts[t_labels[index]][temp_count] += 0

    # have all the information, we start calculating the probability
    for num in range(TOTAL_DIG):
        this_l_count = label_counts[num]
        # initialize the probability term in the dict
        p_prob[num] = [0] * TOTAL_PIXEL
        for pix in range(TOTAL_PIXEL):
            n_pix = p_counts[num][pix]
            p_pix = (n_pix + 1.) / (this_l_count + 1.)
            p_prob[num][pix] = p_pix

    return p_prob
