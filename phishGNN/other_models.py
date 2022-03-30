import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

import dataprep
from models import FeedforwardNeuralNetModel


def train_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    return clf


def train_logistic_regression(X_train, X_test, y_train, y_test):
    reg = LogisticRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    return reg


def train_svm(X_train, X_test, y_train, y_test):
    svm = SVC(kernel = 'rbf', random_state = 0)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    return svm


def train_ffn(X_train, X_test, y_train, y_test):
    model = FeedforwardNeuralNetModel(
        input_dim=len(X_train[0]),
        hidden_dim=128,
        output_dim=2,
    )
    lr = 0.01
    weight_decay = 4e-5
    epochs = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    for epoch in range(epochs):
        loss = model.fit(X_train, y_train, optimizer, loss_fn)
        train_acc = model.test(X_train, y_train)
        test_acc = model.test(X_test, y_test)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f'Epoch: {(epoch+1):03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    df, X, y = dataprep.load_train_set("data/train/raw/both.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
