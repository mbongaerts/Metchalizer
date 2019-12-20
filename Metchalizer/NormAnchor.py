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


'''

@author: Michiel Bongaerts
'''

import pandas as pd
import numpy as np
import scipy.stats as stats
from collections import defaultdict
import copy



# Classes
class NormAnchor:
    def __init__(self, raw_data, batch_with_IDs, tag_QC= 'QC_' ):
        '''Data should contain raw abundancies. No pre-scaled data is allowed. Format: Samples as rows and metabolites as columns'''
        self.batch_with_IDs = batch_with_IDs
        self.data = raw_data
        self.tag_QC = tag_QC


        IDs_having_batch_ID = []
        for batch, IDs_in_batch in self.batch_with_IDs.items():
            IDs_having_batch_ID.extend(IDs_in_batch)


        # Check if all sample IDs are present in IDs_with_batch
        if( not len( set( self.data.index.tolist() ).difference( set( IDs_having_batch_ID ) ) )  == 0 ):
            raise ValueError("Sample IDs present in data with are not in IDs_with_batch dictionary or vice versa.")

        # Check if Anchor samples are present
        self.anchor_samples = [ID for ID in self.data.index if ( self.tag_QC in ID )  ]
        if( len(self.anchor_samples) == 0 ):
            raise ValueError("The index should contain Anchor samples and should contain "+tag_QC+" in their sample name!")

        print("Total amount of Anchor samples:", len(self.anchor_samples) )


    def normalize(self):
        data = self.data.copy()


        df = pd.DataFrame([])
        for batch_nr, IDs in self.batch_with_IDs.items():
            data_batch = data.loc[ IDs,:]

            anchor_samples_in_batch = [ ID for ID in IDs if ID in self.anchor_samples]
            if( len(anchor_samples_in_batch)  == 0):
                raise ValueError("Batch {} doesn't contain any Anchor samples".format(str(batch_nr) ) )

            anchor_median = data_batch.loc[anchor_samples_in_batch].median().fillna(1).replace(0,1)
            df_ = data_batch / anchor_median
            df = pd.concat([df, df_])

        data_norm = df
         
        # Dont allow negative numbers
        data_norm[ data_norm < 0 ] = 0
        data_norm = data_norm.fillna(0)

        self.data_normalized = data_norm

        print("Done with normalization")






