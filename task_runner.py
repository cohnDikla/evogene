import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from datetime import datetime
import time
from Classifier import Classifier
import data_loader


input_dir = os.getcwd()
input_path = os.path.join(input_dir, "input.csv")
results_path = os.path.join(input_dir, "results.txt")
fig_path = os.path.join(input_dir, "quality_as_function_of_train_size.pdf")
fig_hist_path = os.path.join(input_dir, "abs_scores_hist.pdf")
FMT = '%H:%M:%S'
NUMBER_OF_HIST_BINS = 1000
TRAIN_SIZE_STEP = 15
THRESHOLD_STEP = 10
TRAIN_PROPORTION = 0.8


def draw_histogram(scores):
    """
    Creates the histogram of pair scores and saves it as pdf file.
    :param scores: the scores of all pairs
    """
    maximum = max(scores)
    minimum = min(scores)
    bin_numbers = np.linspace(minimum, maximum, NUMBER_OF_HIST_BINS)
    fig = plt.figure()
    fig.suptitle("Histogram of pair scores", fontsize=14)
    ax = fig.add_subplot(111)
    plt.hist(scores, bin_numbers, color="#0000FF")
    ax.set_xlabel('Score')
    ax.set_ylabel('Counts')
    plt.savefig(fig_hist_path)
    print("saving figure: ", fig_hist_path)


def get_pair_samples():
    """
    Returns pairs of Sample objects (healthy and sick)
    :return: the healthy-sick pairs
    """
    samples = data_loader.read_input_file()
    pairs = []
    healthy_samples = []
    sick_samples = []
    for sample in samples:
        if sample.is_healthy():
            healthy_samples.append(sample)
        else:
            sick_samples.append(sample)
    for healthy in healthy_samples:
        for sick in sick_samples:
            pairs.append((healthy, sick))
    return pairs


def calculate_scores(pairs):
    """
    Calculates the scores for all pairs.
    :param pairs: the given pairs of Sample objects (healthy and sick)
    :return: the scores of all pairs
    """
    scores_all_pairs = []
    for healthy, sick in pairs:
        healthy_vals = healthy.get_species_avgs()
        sick_vals = sick.get_species_avgs()
        sum_diff = 0
        for i in range(len(healthy_vals)):
            sum_diff += abs(healthy_vals[i] - sick_vals[i])
        scores_all_pairs.append(sum_diff)
    return scores_all_pairs


def set_true_labels_per_threshold(sorted_scores, train_data, test_data, idx):
    """
    For each threshold, set the true labels for the train and test set.
    :param sorted_scores: all of the pair scores, sorted in ascending order
    :param train_data: the train scores
    :param test_data: the test scores
    :param idx: index of threshold
    :return: the train and test labels
    """
    # set true labels for the train data according to each threshold
    threshold = sorted_scores[idx]
    train_labels = []
    test_labels = []
    for score in train_data:
        if score > threshold:
            train_labels.append(1)
        else:
            train_labels.append(0)
    # set true labels for the test data
    for score in test_data:
        if score > threshold:
            test_labels.append(1)
        else:
            test_labels.append(0)
    return train_labels, test_labels


def draw_quality_vs_train_size_graph(train_sizes, mean_f1_score):
    """
    Creates the graph of quality as a function of the training set size,
    and saves it as pdf file.
    :param train_sizes: the x-axis values
    :param mean_f1_score: the y-axis values
    """
    plt.scatter(train_sizes, mean_f1_score, s=1)
    plt.xlabel("Training set size")
    plt.ylabel("Quality of classification")
    plt.title("Quality of classification as function of the training set size")
    plt.savefig(fig_path)


def print_total_time(start_time):
    """
    Prints the total run time.
    :param start_time: the starting time stamp
    """
    end_time = datetime.fromtimestamp(time.time()).strftime(FMT)
    tdelta = datetime.strptime(end_time, FMT) - datetime.strptime(start_time, FMT)
    print("total time: ", tdelta)


def get_train_and_test_data(train_indices, sorted_scores):
    """
    Set the training and test data.
    :param train_indices: indices of the train examples
    :param sorted_scores: all of the pair scores, sorted in ascending order
    :return: the train and test data
    """
    train_data = []
    test_data = []
    for i in range(len(sorted_scores)):
        if i in train_indices:
            train_data.append(sorted_scores[i])
        else:
            test_data.append(sorted_scores[i])
    return train_data, test_data


def add_avg_f1(avg_f1, results_file, mean_f1_scores, train_size):
    """
    Prints the current average f1-score and updates the list of mean f1 scores.
    :param avg_f1: current average f1-score
    :param results_file: the output file
    :param mean_f1_scores: the list of mean f1 scores
    :param train_size: current training set size
    :return: the updated list of mean f1 scores
    """
    print("average f1 score: ", avg_f1)
    mean_f1_scores.append(avg_f1)
    results_file.write(str(train_size) + "\t" + str(avg_f1) + "\n")
    return mean_f1_scores


def run_task():
    """
    The main function that runs the task.
    """
    with open(results_path, "w+") as results_file:
        pairs = get_pair_samples()
        scores_all_pairs = calculate_scores(pairs)
        # visualize the histogram of scores across all pairs
        draw_histogram(scores_all_pairs)
        # sort all scores in ascending order
        sorted_scores = sorted(scores_all_pairs)
        # set the maximal train size to be 80% of the total number of pairs
        max_train_size = math.ceil(TRAIN_PROPORTION * len(pairs))
        all_data_indices = [i for i in range(len(pairs))]
        train_sizes = []
        mean_f1_scores = []

        # run over the different train set sizes
        for train_size in range(TRAIN_SIZE_STEP, max_train_size+1, TRAIN_SIZE_STEP):
            print("train size: ", train_size)
            train_sizes.append(train_size)
            f1_vals = []
            # randomly sample the training set examples
            train_indices = random.sample(all_data_indices, train_size)
            train_data, test_data = get_train_and_test_data(train_indices, sorted_scores)

            # run over different thresholds and set the true labels
            for idx in range(len(sorted_scores)-1, 0, -THRESHOLD_STEP):
                train_labels, test_labels = set_true_labels_per_threshold(sorted_scores,
                                                            train_data, test_data, idx)
                # our classifier needs samples of at least 2 classes in the data
                if (0 not in train_labels) or (1 not in train_labels):
                    continue

                # train the linear SVM classifier
                lin_classifier = Classifier(train_data, train_labels,
                                            test_data, test_labels)
                lin_classifier.train_classifier()

                # evaluate the classifier
                f1_vals.append(lin_classifier.evaluate_classifier())

            mean_f1_scores = add_avg_f1(np.mean(f1_vals), results_file, mean_f1_scores, train_size)

        # display the final graph
        draw_quality_vs_train_size_graph(train_sizes, mean_f1_scores)


def main():
    start_time = datetime.fromtimestamp(time.time()).strftime(FMT)
    run_task()
    print_total_time(start_time)


if __name__ == '__main__':
    main()
