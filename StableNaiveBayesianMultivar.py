import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
pandas2ri.activate()
numpy2ri.activate()


class StableNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes = np.unique(y)

        self.alphas = []
        self.betas = []
        self.gammas = []
        self.deltas = []
        self.priors = []

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors.append(len(X_c) / len(X))

            # print(X_c[:,0]) # X_c[:,f] all data points for fth feature
            # print(X_c[0,:]) # X_c[i,:] ith data point with all features

            alphas = []
            betas = []
            gammas = []
            deltas = []

            for f in range(X_c.shape[1]):
                # print(f+1, 'th feature for', i+1, 'th class:', X_c[:,f])

                robjects.globalenv["vector"] = X_c[:, f]
                # print('feature' , i+1, X_cls[:,i])
                # R script to calculate alphas
                robjects.r('''
                            options(encoding = "UTF-8")
                            library(stable)

                            fit <- stable.fit(vector, method = 1)
                            alpha.est <- fit['alpha']
                            # print(alpha.est)
                            beta.est <- fit['beta']
                            # print(beta.est)
                            gamma.est <- fit['gamma']
                            # print(gamma.est)
                            delta.est <- fit['delta']
                            # print(delta.est)

                            ''')

                alphas.append(robjects.r['alpha.est'][0])
                betas.append(robjects.r['beta.est'][0])
                gammas.append(robjects.r['gamma.est'][0])
                deltas.append(robjects.r['delta.est'][0])

            # print('alphas for ', i+1, 'th class', alphas)
            self.alphas.append(alphas)
            self.betas.append(betas)
            self.gammas.append(gammas)
            self.deltas.append(deltas)

        self.alphas = np.asarray(self.alphas)
        self.betas = np.asarray(self.betas)
        self.gammas = np.asarray(self.gammas)
        self.deltas = np.asarray(self.deltas)

    def _pdf(self, x, alpha_est, beta_est, gamma_est, delta_est):
        # Pass data to the R environment
        robjects.globalenv["alpha.est"] = alpha_est
        robjects.globalenv["beta.est"] = beta_est
        robjects.globalenv["gamma.est"] = gamma_est
        robjects.globalenv["delta.est"] = delta_est
        robjects.globalenv["x"] = x

        # R script to calculate alphas
        robjects.r('''
                 # print(x)
                 # print(alpha.est)
                 # print(beta.est)
                 # print(gamma.est)
                 # print(delta.est)
                 pdf <- dstable(x, alpha = alpha.est, beta = beta.est, gamma = gamma.est, delta = delta.est)
                ''')

        pdf = robjects.r['pdf']

        return pdf

    def predict(self, X):
        predictions = []
        for x in X:
            class_likelihood = []

            for i, c in enumerate(self.classes):
                feature_likelihood = []
                for f in range(X.shape[1]):
                    feature_likelihood.append(self._pdf(x[f], self.alphas[i, f], self.betas[i, f], self.gammas[i, f], self.deltas[i, f])[0])

                feature_likelihood = [1e-10 if x == 0 else x for x in feature_likelihood]
                feature_likelihood.append(self.priors[i])
                class_likelihood.append(np.sum(np.log(feature_likelihood)))

            predictions.append(self.classes[np.argmax(class_likelihood)])

        return np.asarray(predictions)

class get_stat:
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes = np.unique(y)

        self.alphas = []
        self.betas = []
        self.gammas = []
        self.deltas = []
        self.priors = []

        for f in range(X.shape[1]):
            # print(f+1, 'th feature for', i+1, 'th class:', X_c[:,f])

            robjects.globalenv["vector"] = X[:, f]
            # print('feature' , i+1, X_cls[:,i])
            # R script to calculate alphas
            robjects.r('''
                        options(encoding = "UTF-8")
                        library(stable)

                        fit <- stable.fit(vector, method = 1)
                        alpha.est <- fit['alpha']
                        # print(alpha.est)
                        beta.est <- fit['beta']
                        # print(beta.est)
                        gamma.est <- fit['gamma']
                        # print(gamma.est)
                        delta.est <- fit['delta']
                        # print(delta.est)

                        ''')

            self.alphas.append(robjects.r['alpha.est'][0])
            self.betas.append(robjects.r['beta.est'][0])
            self.gammas.append(robjects.r['gamma.est'][0])
            self.deltas.append(robjects.r['delta.est'][0])

        self.alphas = np.asarray(self.alphas)
        self.betas = np.asarray(self.betas)
        self.gammas = np.asarray(self.gammas)
        self.deltas = np.asarray(self.deltas)

        mean_alpha =  np.mean(self.alphas)
        median_alphas = np.median(self.alphas)
        mean_beta =  np.mean(self.betas)
        median_betas = np.median(self.betas)

        return mean_alpha, median_alphas, mean_beta, median_betas