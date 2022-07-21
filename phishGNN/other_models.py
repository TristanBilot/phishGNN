import numpy as np
import torch
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import dataprep


# from models.ffn import FeedforwardNeuralNetModel
from .models import FeedforwardNeuralNetModel


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def train_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    return clf, acc


def train_logistic_regression(X_train, X_test, y_train, y_test):
    reg = LogisticRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    return reg, acc


def train_svm(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='rbf', random_state=0)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy:', acc)
    return svm, acc


def train_ffn(X_train, X_test, y_train, y_test, epochs=50):
    model = FeedforwardNeuralNetModel(
        input_dim=len(X_train[0]),
        hidden_dim=128,
        output_dim=2,
    )
    lr = 0.01
    weight_decay = 4e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for epoch in range(epochs):
        loss = model.fit(X_train, y_train, optimizer, loss_fn)
        train_acc = model.test(X_train, y_train)
        test_acc = model.test(X_test, y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        # print(f'Epoch: {(epoch+1):03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    return model, test_accs


def do_experiments(n: int = 10):
    df, X, y = dataprep.load_train_set('data/train/raw/both.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    names = [
        'Nearest Neighbors',
        'Linear SVM',
        'RBF SVM',
        'Decision Tree',
        'Random Forest',
        'Neural Net',
        'AdaBoost',
        'Naive Bayes',
        'QDA',
        'LogisticRegression',
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel='linear', C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(),
    ]

    for i, clf in enumerate(classifiers):
        for _ in range(n):
            samples = []
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = metrics.accuracy_score(y_test, y_pred)
            samples.append(acc)
        print(f'{names[i]:20} \t{np.mean(samples)} +- {np.std(samples)}')

    _, ffns = train_ffn(X_train, X_test, y_train, y_test, epochs=n)
    print(f'Feed Forward: \t{np.mean(ffns)} +- {np.std(ffns)}')


if __name__ == '__main__':
    do_experiments()
