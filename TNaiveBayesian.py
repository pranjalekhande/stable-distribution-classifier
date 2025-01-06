import numpy as np
import scipy.stats as stats
from sklearn.utils.validation import check_X_y

class TNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes = np.unique(y)
        # priors = []

        self.df_s = []
        self.scales = []
        self.locs = []
        self.priors = []

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors.append(len(X_c) / len(X))

            # print(X_c[:,0]) # X_c[:,f] all data points for fth feature
            # print(X_c[0,:]) # X_c[i,:] ith data point with all features

            df_s = []
            scales = []
            locs = []

            for f in range(X_c.shape[1]):
                # print(f+1, 'th feature for', i+1, 'th class:', X_c[:,f])

                v = X_c[:, f]
                # print('feature' , i+1, X_cls[:,i])

                df1, loc1, scale1 = stats.t.fit(v)

                df_s.append(df1)
                scales.append(scale1)
                locs.append(loc1)

            # print('as for ', i+1, 'th class', df_s)
            self.df_s.append(df_s)
            self.scales.append(scales)
            self.locs.append(locs)

        self.df_s = np.asarray(self.df_s)
        self.scales = np.asarray(self.scales)
        self.locs = np.asarray(self.locs)

    def _pdf(self, x, df1, scale1, loc1):

        pdf = stats.t.pdf(x, df1, loc1, scale1)

        return pdf

    def predict(self, X):
        predictions = []
        for x in X:

            class_likelihood = []

            for i, c in enumerate(self.classes):
                feature_likelihood = []
                for f in range(X.shape[1]):
                    feature_likelihood.append(self._pdf(x[f], self.df_s[i, f], self.scales[i, f], self.locs[i, f]))

                feature_likelihood = [1e-10 if x == 0 else x for x in feature_likelihood]
                feature_likelihood.append(self.priors[i])
                class_likelihood.append(np.sum(np.log(feature_likelihood)))

            predictions.append(self.classes[np.argmax(class_likelihood)])

        return np.asarray(predictions)