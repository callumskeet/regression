import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.datasets import make_regression, make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split

from models import LinearRegression, LogisticRegression


def linear_test():
    X, y = make_regression(n_features=1, noise=20, random_state=1234)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1234
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    plt.figure(1, figsize=(5, 4))
    plt.scatter(X_test, y_test, c="black")
    plt.plot(X_test, lr.theta * X_test + lr.bias, linewidth=1, c="red")
    plt.axhline(0.5, color=".5")

    plt.ylabel("y")
    plt.xlabel("X")
    plt.legend(
        ("Linear Regression Model",),
        loc="lower right",
        fontsize="small",
    )
    plt.tight_layout()
    plt.show()


def logistic_test():
    n_samples = 100
    np.random.seed(0)
    X_train = np.random.normal(size=n_samples)
    y_train = (X_train > 0).astype(float)
    X_train[X_train > 0] *= 4
    X_train += 0.3 * np.random.normal(size=n_samples)

    X_train = X_train[:, np.newaxis]

    X, y = make_classification(
        n_features=1,
        n_classes=2,
        n_redundant=0,
        n_informative=1,
        n_clusters_per_class=1,
        class_sep=0.75,
        shuffle=True,
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0
    )

    df_test = pd.DataFrame(data=[X_test.flatten(), y_test]).T
    df_test.columns = ["X", "y"]

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    score = [1 if yi == yi_pred else 0 for yi, yi_pred in zip(y_test, y_pred)]
    print(np.sum(score) / len(score))

    # and plot the result
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X_train.ravel(), y_train, color="black", zorder=20)

    df_test["loss"] = expit(X_test * lr.theta + lr.bias).ravel()
    df_test = df_test.sort_values("X")
    plt.plot(df_test["X"], df_test["loss"], color="red", linewidth=3)

    ols = LinearRegression()
    ols.fit(X_train, y_train)
    plt.plot(X_test, ols.theta * X_test + ols.bias, linewidth=1)
    plt.axhline(0.5, color=".5")

    plt.ylabel("y")
    plt.xlabel("X")
    plt.xticks(range(-5, 10))
    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.25, 1.25)
    plt.xlim(-2, 2)
    plt.legend(
        ("Logistic Regression Model", "Linear Regression Model"),
        loc="lower right",
        fontsize="small",
    )
    plt.tight_layout()
    plt.show()
