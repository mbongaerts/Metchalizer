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

@author: Michiel Bongaerts
"""

import copy
import json
from collections import defaultdict

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures


# Functions
def box_cox(x, lambda_1, lambda_2):
    return ((x + lambda_2) ** lambda_1 - 1) / (lambda_1)


# Classes
class NormMetchalizer:
    def __init__(
        self,
        raw_data,
        batch_with_IDs,
        prefix_IS="IS_",
        prefix_QC_samples="QC_",
        remove_outliers_Z_threshold=2,
        interia_per_threshold=75,
        lambda_1=0.5,
        lambda_2=1,
    ):
        """Data should contain raw abundancies. No pre-scaled data is allowed. Format: Samples as rows and metabolites/features as columns"""
        self.batch_with_IDs = batch_with_IDs
        self.data = raw_data
        self.prefix_QC_samples = prefix_QC_samples
        self.prefix_IS = prefix_IS

        self.remove_outliers_Z_threshold = remove_outliers_Z_threshold
        self.interia_per_threshold = interia_per_threshold
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.features_with_failed_normalization = []

        IDs_having_batch_ID = []
        for batch, IDs_in_batch in self.batch_with_IDs.items():
            IDs_having_batch_ID.extend(IDs_in_batch)

        # Check if all sample IDs are present in IDs_with_batch
        if ( not len(set(self.data.index.tolist()).difference(set(IDs_having_batch_ID))) == 0 ):
            raise ValueError(
                "Sample IDs present in data with are not in IDs_with_batch dictionary or vice versa."
            )

        # Check if Internal Standards are present
        self.IS_cols = [col for col in self.data.columns if (self.prefix_IS in col)]

        if( len(self.IS_cols) <= 1 ):
            raise ValueError(
                "The columns should contain at least 2 internal standard(s) indicates by "
                + prefix_IS
                + ". "
            )
        print("Total amount of standards:", len(self.IS_cols))

        # Check if QC samples are present, although they are not used in this algorithm
        self.QC_samples = [ind for ind in self.data.index if (prefix_QC_samples in ind)]
        if( len(self.QC_samples) == 0 ):
            print(" No QC samples found ")

    def normalize(self):
        data = self.data.copy()

        # Make sure float are ok and data contain no NaN's
        data = data.apply(np.float64)
        assert not data.isnull().any().any()

        # Boxcox
        data = data.apply(lambda x: box_cox(x, self.lambda_1, self.lambda_2))

        # Select IS data
        X_predict = data.loc[:, self.IS_cols]

        # Scale IS data
        X_predict = (X_predict - X_predict.median(axis=0)) / X_predict.mad(axis=0)

        batch_labels = pd.Series()
        for k, (batch, IDs) in enumerate(self.batch_with_IDs.items()):
            batch_labels = batch_labels.append(pd.Series([k for i in range(len(IDs))], index=IDs) )
            

        # PLS
        le = LabelBinarizer()
        pls = PLSRegression(n_components=X_predict.shape[1])
        y_pls = le.fit_transform(X_predict.assign(batch_labels=batch_labels)["batch_labels"] )
        pls.fit(X_predict, y_pls)

        # Obtain (PLS) PC's and store them in X_trans
        X_trans = pls.transform(X_predict)
        X_trans = pd.DataFrame(X_trans, index=X_predict.index)

        # Determine how many LV's (latent variables) are needed for regression / mixed effect model fit
        inertias = []
        for i in range(0, X_predict.shape[1]):
            inertia = 0
            for k, (batch, IDs) in enumerate(self.batch_with_IDs.items()):
                X_trans_batch = X_trans.loc[IDs].iloc[:, 0:i]
                c = X_trans_batch.mean(axis=0)
                inertia += np.sum(np.linalg.norm((X_trans_batch - c).values, axis=1) ** 2 )
                
            print("K(" + str(i) + ") = " + str(round(inertia)))
            inertias.append(inertia)

        inertias_perc = inertias / np.max(inertias) * 100
        ind = np.where(inertias_perc > self.interia_per_threshold)[0][0]
        self.n_components_PLS = range(0, X_predict.shape[1])[ind] + 1
        print(str(self.n_components_PLS) + " LVs are taken.")

        # Refit PLS with the choosen amount of LV's
        LVs_cols = ["LV" + str(i + 1) for i in range(self.n_components_PLS)]
        pls = PLSRegression(n_components=self.n_components_PLS)
        pls.fit(X_predict, y_pls)

        X_PLS = pd.DataFrame(pls.transform(X_predict), index=X_predict.index, columns=LVs_cols )
        self.pls = pls

        # Iterate over all features/ metabolites
        df = pd.DataFrame([])
        for i, metab in enumerate(data.columns):
            print("Batch correction ", str(i) + "/" + str(len(data.columns)), metab)

            # Get abundancies for metabolite and rescale
            y_metab = data[metab]
            y_median = y_metab.median()
            y_mad = y_metab.mad()
            y_metab = (y_metab - y_median) / y_mad

            # Copy PLS decomposition matrix
            X_predict = X_PLS.copy()

            not_outliers = []
            for k, (batch, IDs) in enumerate(self.batch_with_IDs.items()):

                # Get (not-)outliers from within batch Z-score
                y_k = data.loc[IDs, metab]
                y_k = (y_k - y_k.mean()) / (y_k.std())
                not_outliers.extend(y_k[y_k.abs() < self.remove_outliers_Z_threshold].index.tolist() )
                
                # Update X_predict with group name
                X_predict.loc[IDs, "group"] = k

            X_predict = X_predict.assign(y=y_metab)
            X_predict["Intercept"] = 1

            predictor_cols = ["Intercept"]
            predictor_cols.extend(LVs_cols)

            # Fit Mixed Effect Model on only the non-outliers
            fit_succes = False
            try:
                endog = X_predict["y"].loc[not_outliers]
                exog = X_predict[predictor_cols].loc[not_outliers]
                groups = X_predict["group"].loc[not_outliers]

                md = sm.MixedLM(endog, exog, groups=groups, exog_re=exog["Intercept"])
                mdf = md.fit()
                fit_succes = True

            except Exception as e:
                print(e, "We take all samples for this feature (no outlier removal)")

                # Try another time with all samples:
                try:
                    endog = X_predict["y"].loc[:]
                    exog = X_predict[predictor_cols].loc[:]
                    groups = X_predict["group"].loc[:]

                    md = sm.MixedLM(endog, exog, groups=groups, exog_re=exog["Intercept"] )
                    
                    mdf = md.fit()
                    fit_succes = True

                # This features is really not nice
                except Exception as e:
                    print("Cannot fit the model on this feature!")
                    fit_succes = False

            # Normalize abundancies using fitted model
            if fit_succes == True:

                df_ = pd.DataFrame([])
                for grp, grp_data in X_predict.groupby(by="group"):
                    df__ = (
                        (grp_data["y"] - mdf.random_effects[grp].values[0])
                        .reset_index()
                        .set_index("index")
                    )
                    df__.columns = ["y_batch_corrected"]
                    y_IS_correction = mdf.fe_params["Intercept"]
                    for LV in LVs_cols:
                        y_IS_correction += (
                            mdf.fe_params[LV] * X_predict.loc[grp_data.index, LV]
                        )

                    df__ = df__.assign(y_IS_correction=y_IS_correction)
                    df_ = pd.concat([df_, df__])

                df_ = df_.join(X_predict[["LV1"]])
                df_ = df_.assign(
                    **{
                        str(metab): (
                            df_["y_batch_corrected"]
                            - df_["y_IS_correction"]
                            + df_["y_IS_correction"].median()
                        )* y_mad + y_median # <-- scaling back
                    }
                )
                df = df.T.append(df_[str(metab)].T).T

            # Return original abundancies
            else:
                df = df.T.append(data[metab].T).T
                self.features_with_failed_normalization.append(metab)

        # Replace
        data = df

        # Dont allow negative numbers
        data[data < 0] = 0
        data = data.fillna(0)

        self.data_normalized = data

        print("Done with normalization")
