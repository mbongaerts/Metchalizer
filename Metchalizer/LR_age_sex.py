# Copyright (c) 2019 Michiel Bongaerts.
# All rights reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.


"""
configuration and utility functions 
@author: Michiel Bongaerts
"""
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class WeightedMeanSigma:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def fit(self, age_train, sigma_train):
        assert len(age_train) == len(sigma_train)
        self.age_train = age_train
        self.sigma_train = sigma_train

    def predict(self, ages):
        means = []
        for age in ages:
            weigths = np.exp(-np.abs(self.age_train - age) / (self.a + self.b * age))
            weigths = np.round(weigths, 10)
            weigths = weigths / np.sum(weigths)

            weighted_mean = np.sum(weigths * self.sigma_train)
            means.append(weighted_mean)

        return means


class LR_age_sex:
    def __init__(
        self,
        n_jobs=1,
        polynomials=3,
        a=35,
        b=0.1,
        normalize=False,
        Z_outlier_threshold=3,
    ):

        self.normalize = normalize
        self.n_jobs = n_jobs
        self.LR = LinearRegression(
            normalize=self.normalize, n_jobs=self.n_jobs, fit_intercept=False
        )
        self.polynomials = polynomials
        self.a = a
        self.b = b
        self.fitted = False
        self.Z_outlier_threshold = Z_outlier_threshold

    def fit(self, X_train, y_train, interaction=False):
        assert X_train.columns.tolist()[0] == "Age_in_days"
        assert X_train.columns.tolist()[1] == "Sex"

        self.fitted = True

        self.interaction = interaction

        self.y_train_median = y_train.median()
        self.y_train_mad = y_train.mad()

        # Determine outliers
        y_train_Z = (y_train - self.y_train_median) / (self.y_train_mad * 1.48)
        self.not_outlier_IDs = y_train_Z[
            y_train_Z.abs() < self.Z_outlier_threshold
        ].index.tolist()

        # Add polynomials
        poly = PolynomialFeatures(self.polynomials, include_bias=True)
        X_poly = pd.DataFrame(
            poly.fit_transform(X_train[["Age_in_days"]]), index=X_train.index
        )
        X_poly = X_poly.assign(Sex=X_train["Sex"])

        if self.interaction == True:
            X_poly = X_poly.assign(interaction=X_poly["Sex"] * X_poly[1])

        self.feature_names = X_poly.columns.tolist()

        # Remove outliers and use Z-tranformed y_train
        self.X_train_not_outliers = X_train.loc[self.not_outlier_IDs]
        self.X_train_poly = X_poly.loc[self.not_outlier_IDs]
        self.y_train = y_train.loc[self.not_outlier_IDs]
        self.ages = X_train["Age_in_days"].loc[self.not_outlier_IDs].values

        self.LR.fit(self.X_train_poly.values, self.y_train.values)
        self.y_fit = self.LR.predict(self.X_train_poly.values)
        self.sigma_train = (self.y_train - self.y_fit) ** 2

        self.WMS = WeightedMeanSigma(self.a, self.b)
        self.WMS.fit(self.ages, self.sigma_train)

        self.sigma_WMS = self.WMS.predict(self.ages)

        self.Sigma = np.diag(self.sigma_WMS)
        self.XTX_inv = np.linalg.inv(
            np.dot(np.transpose(self.X_train_poly.values), self.X_train_poly.values)
        )
        self.cov_beta = np.dot(
            np.dot(
                np.dot(np.dot(self.XTX_inv, self.X_train_poly.values.T), self.Sigma),
                self.X_train_poly.values,
            ),
            self.XTX_inv,
        )

    def predict(self, X_test):
        assert X_test.columns.tolist()[0] == "Age_in_days"
        assert X_test.columns.tolist()[1] == "Sex"

        poly = PolynomialFeatures(self.polynomials, include_bias=True)
        X_poly = pd.DataFrame(
            poly.fit_transform(X_test[["Age_in_days"]]), index=X_test.index
        )
        X_poly = X_poly.assign(Sex=X_test["Sex"])

        if self.interaction == True:
            X_poly = X_poly.assign(interaction=X_poly["Sex"] * X_poly[1])

        self.X_test = X_poly.values
        self.ages_test = X_test["Age_in_days"]

        self.y_pred = self.LR.predict(self.X_test)

        self.var_1 = np.array(
            [np.dot(np.dot(g, self.cov_beta), g) for g in self.X_test]
        )
        self.var_2 = self.WMS.predict(self.ages_test)
        self.std_pred = np.sqrt(self.var_1 + self.var_2)

        results = pd.DataFrame(self.y_pred, columns=["y_pred"])
        results.loc[:, "std_pred"] = self.std_pred

        return results

    def p_values(self, n_bootstraps=50, train_size=0.95):
        if self.fitted != True:
            raise ValueError("Fit your model first on all the data ")

        def Z_p_val(x):
            return (stats.norm.sf(np.abs(x))) * 2

        p_vals_fits = []
        coefs = []
        for i in range(n_bootstraps):

            X_train, X_test, y_train, y_test = train_test_split(
                self.X_train_not_outliers, self.y_train, test_size=1 - train_size
            )

            model = LR_age_sex(
                n_jobs=10, polynomials=self.polynomials, a=self.a, b=self.b
            )
            model.fit(X_train, y_train, interaction=self.interaction)

            beta_means = model.LR.coef_
            beta_vars = np.diag(model.cov_beta)

            p_vals_fit = pd.Series(
                Z_p_val((beta_means) / np.sqrt(beta_vars)), index=model.feature_names
            )
            p_vals_fits.append(p_vals_fit)
            coefs.append(list(model.LR.coef_))

        # Results from theoretical framework
        self.p_vals_fits = pd.DataFrame(
            p_vals_fits
        )  # .assign( N= len(self.not_outlier_IDs) )
        self.mean_coefs = pd.Series(
            np.mean(coefs, axis=0), index=model.X_train_poly.columns
        )
