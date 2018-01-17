from default_clf import DefaultNSL
from time import clock

def get_current_charge():
    with open('/sys/class/power_supply/BAT0/charge_now') as f:
        return f.readline()


def check_load_training(clf, path):
    charge_start = get_current_charge()
    start = clock()
    for i in range(10000):
        clf.load_training_data(path)
    end = clock()
    charge_end = get_current_charge()

    return [(end - start), charge_end - charge_start]


def check_load_testing(clf, path):
    charge_start = get_current_charge()
    start = clock()
    for i in range(10000):
        clf.load_testing_data(path)
    end = clock()
    charge_end = get_current_charge()

    return [(end - start), charge_end - charge_start]


def check_training(clf):
    charge_start = get_current_charge()
    start = clock()
    for i in range(10000):
        clf.train_clf()
    end = clock()
    charge_end = get_current_charge()

    return [(end - start), charge_end - charge_start]

def check_testing_entire_dataset(clf, train=False):
    charge_start = get_current_charge()
    start = clock()
    for i in range(10000):
        clf.test_clf(train)
    end = clock()
    charge_end = get_current_charge()

    return [(end - start), charge_end - charge_start]


def check_predict_row(clf, row):
    charge_start = get_current_charge()
    start = clock()
    for i in range(10000):
        clf.predict(row)
    end = clock()
    charge_end = get_current_charge()

    return [(end - start), charge_end - charge_start]


def evaluate_classifier(clf):
    train_path = 'data/KDDTrain+.csv'
    test_path = 'data/KDDTest+.csv'
    results = []
    results.append(check_load_training(clf, train_path))
    results.append(check_load_testing(clf, test_path))
    results.append(check_testing_entire_dataset(clf))
    row = clf.testing.iloc[0]
    results.append(check_predict_row(clf, row))
    return results
