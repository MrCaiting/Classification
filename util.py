"""Various Utility Functions that we need."""
TOTAL_IMG = 5000
TOTAL_DIG = 10


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
