import csv
from default_clf import DefaultNSL
from itertools import chain
from time import process_time

import numpy as np
import pandas as pd

NUM_PASSES = 100

def get_current_charge():
    with open('/sys/class/power_supply/BAT0/charge_now') as f:
        return int(f.readline())


def check_load_training(clf, path):
    start = process_time()
    clf.load_training_data(path)
    end = process_time()
    return end - start


def check_load_testing(clf, path):
    start = process_time()
    clf.load_test_data(path)
    end = process_time()
    return end - start


def check_training(clf):
    start = process_time()
    clf.train_clf()
    end = process_time()
    return end - start


def check_testing_entire_dataset(clf, train=False):
    start = process_time()
    clf.test_clf(train)
    end = process_time()
    return end - start


def check_predict_row(clf, row):
    start = process_time()
    clf.predict(row)
    end = process_time()
    return end - start


def get_stats(arr, function, *args, **kwargs):
    charge_start = get_current_charge()
    for i in range(NUM_PASSES):
        arr[i] = function(*args, **kwargs)
    charge_end = get_current_charge()
    mean = arr.mean()
    std = arr.std()
    return [mean, std, (charge_start - charge_end)]


def evaluate_classifier(clf):
    train_path = 'data/KDDTrain+.csv'
    test_path = 'data/KDDTest+.csv'
    res = np.empty(shape=(NUM_PASSES, 1))
    load_train = get_stats(res, check_load_training, clf, train_path)
    print('Loading Training: ', load_train)
    load_test = get_stats(res, check_load_testing, clf, test_path)
    print('Loading Testing: ', load_test)
    train = get_stats(res, check_training, clf)
    print('Training: ', train)
    test_dataset = get_stats(res, check_testing_entire_dataset, clf)
    print('Testing dataset: ', test_dataset)
    row = clf.testing[0].iloc[0].values.reshape(1, -1)
    test_row = get_stats(res, check_predict_row, clf, row)
    print('Testing one row: ', test_row)
    with open('results.csv', 'a', newline='') as csvf:
        csv_writer = csv.writer(csvf)
        csv_writer.writerow([clf.__class__.__name__, 'Number of Passes:', NUM_PASSES])
        csv_writer.writerow(['Function', 'Time (s) Mean', 'Time Std',
                             'Total Power (microwatt-hour)'])
        csv_writer.writerow(['Loading Training Data'] + load_train)
        csv_writer.writerow(['Loading Testing Data'] + load_test)
        csv_writer.writerow(['Training Classifier'] + train)
        csv_writer.writerow(['Testing Dataset'] + test_dataset)
        csv_writer.writerow(['Testing One Row'] + test_row)
