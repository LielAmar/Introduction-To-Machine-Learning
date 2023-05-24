from IMLearn.learners.classifiers.decision_stump import DecisionStump
import numpy as np

if __name__ == "__main__":
    stump = DecisionStump()
    X1 = np.random.normal(1, 5, 1000)
    X2 = np.random.normal(1, 5, 1000)
    X3 = np.random.normal(1, 5, 1000)
    X4 = np.random.normal(1, 5, 1000)
    X = np.c_[X1, X2, X3, X4]

    y = np.random.randint(-2, 2, 1000)
    y = np.sign(y+0.00001)

    stump.fit(X, y)


    X1 = np.random.normal(1, 5, 1000)
    X2 = np.random.normal(1, 5, 1000)
    X3 = np.random.normal(1, 5, 1000)
    X4 = np.random.normal(1, 5, 1000)
    X = np.c_[X1, X2, X3, X4]

    y = np.random.randint(-2, 2, 1000)
    y = np.sign(y+0.00001)

    y_hat = stump.predict(X)