import StableNaiveBayesianMultivar
import BetaNaiveBayesian
import TNaiveBayesian
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import load_wine

_data = load_wine()
X = _data.data
y = _data.target
X_train, X_test, y_train, y_test = train_test_split(_data.data, _data.target, test_size=0.33)

# Stable Naive Bayes
snbm = StableNaiveBayesianMultivar.StableNB()
snbm.fit(X_train, y_train)
predictions_snb = snbm.predict(X_test)

# Beta Naive Bayes
bnb = BetaNaiveBayesian.BetaNB()
bnb.fit(X_train, y_train)
predictions_bnb = bnb.predict(X_test)

# Student's - t Naive Bayes 
tnb = TNaiveBayesian.TNB()
tnb.fit(X_train, y_train)
predictions_tnb = tnb.predict(X_test)

## Performing K-fold cross validation
def cross_val(model, X, Y, k):
    # Initialize KFold
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=4)

    precisions = []
    recalls = []
    specificities = []
    f1s = []
    accuracies = []

    # Iterate through the folds
    for fold, (train_index, test_index) in enumerate(kf.split(X, Y)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)

        prediction = model.predict(X_test)

        cm = metrics.confusion_matrix(Y_test, prediction)

        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (tn + fp)

        f1 = (2 * prec * rec / (prec + rec))

        acc = (tp + tn)/(tp + tn + fp + fn)

        precisions.append(prec)
        recalls.append(rec)
        specificities.append(spec)
        f1s.append(f1)
        accuracies.append(acc)

    print("The accuracy is:", round(100*sum(accuracies)/len(accuracies), 3), "%")
    print("The precision is:", round(100*sum(precisions)/len(precisions), 3), "%")
    print("The recall is:", round(100*sum(recalls)/len(recalls), 3), "%")
    print("The specificity is:", round(100*sum(specificities)/len(specificities), 3), "%")
    print("The f1 is:", round(100*sum(f1s)/len(f1s), 3), "%")


cross_val(snbm, X, y, 5)
