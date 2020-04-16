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
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.formula.api as smf


# Functions
def box_cox(x, lambda_1, lambda_2):
    return ((x + lambda_2) ** lambda_1 - 1) / (lambda_1)


# Classes
class NormBestCorrIS:
    def __init__(self, raw_data, batch_with_IDs, prefix_IS="IS_"):
        """Data should contain raw abundancies. No pre-scaled data is allowed. Format: Samples as rows and metabolites as columns"""
        self.batch_with_IDs = batch_with_IDs
        self.data = raw_data
        self.prefix_IS = prefix_IS

        IDs_having_batch_ID = []
        for batch, IDs_in_batch in self.batch_with_IDs.items():
            IDs_having_batch_ID.extend(IDs_in_batch)

        # Check if all sample IDs are present in IDs_with_batch
        if (
            not len(set(self.data.index.tolist()).difference(set(IDs_having_batch_ID)))
            == 0
        ):
            raise ValueError(
                "Sample IDs present in data with are not in IDs_with_batch dictionary or vice versa."
            )

        # Check if Internal Standards are present
        self.IS_cols = [col for col in self.data.columns if (self.prefix_IS in col)]
        if len(self.IS_cols) == 0:
            raise ValueError(
                "The columns should contain internal standard(s) indicated by "
                + prefix_IS
                + ". "
            )
        print("Total amount of standards:", len(self.IS_cols))

    def normalize(self):
        data = self.data.copy()

        # Make sure float are ok and data contain no NaN's
        data = data.apply(np.float64)
        assert not data.isnull().any().any()

        data = data.apply(lambda x: box_cox(x, 0.5, 1))

        metab_cols = data.columns.tolist()

        corrs = pd.DataFrame([])
        for k, (batch, IDs) in enumerate(self.batch_with_IDs.items()):
            rs = []
            for IS in self.IS_cols:
                rs_ = []
                for i, metab in enumerate(metab_cols):
                    if i % 50 == 0:
                        print(batch, IS, i)
                    x, y = data.loc[IDs, IS], data.loc[IDs, metab]
                    r, _ = stats.spearmanr(x, y)
                    rs_.append(r)
                rs.append(rs_)

            rs = pd.DataFrame(rs, index=self.IS_cols).T
            rs = rs.assign(metab=metab_cols)
            rs = rs.assign(batch=batch)
            corrs = pd.concat([corrs, rs])

        mean_corr = corrs.groupby(by="metab").apply(lambda x: x[self.IS_cols].mean())
        best_IS = mean_corr.apply(lambda x: x == x.max(), axis=1)
        metab_with_corr_IS = best_IS.apply(
            lambda x: x.index[x == True].tolist()[0], axis=1
        ).to_dict()

        self.metab_with_corr_IS = metab_with_corr_IS
        self.corrs = corrs

        # Normalize data
        df = pd.DataFrame([])
        for metab in metab_cols:
            y = data.loc[:, metab]
            IS_ = data.loc[:, metab_with_corr_IS[metab]]

            y_norm = (y / IS_) * IS_.median()
            y_norm.name = metab
            y_norm = y_norm.reset_index().set_index("index").T
            df = pd.concat([df, y_norm])

        df = df.T

        # Replace
        data = df

        # Dont allow negative numbers
        data[data < 0] = 0
        data = data.fillna(0)

        self.data_normalized = data

        print("Done with normalization")
