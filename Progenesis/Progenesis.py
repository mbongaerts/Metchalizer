# Copyright (c) 2010-2018 Michiel Bongaerts.
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
Contains main functions and classes for loading and processing Progenesis export data  
@author: Michiel Bongaerts
'''

import pandas as pd
import numpy as np
# import warnings
import os
import sys
import json
import time
import dask 
import matplotlib.pyplot as plt

from collections import defaultdict as ddict
from multiprocessing import cpu_count
import statsmodels.api as sm
from scipy import interpolate

#### Functions ####

def convert_to_float(x):
    try:
        return float(x)
    except:
        return x


def get_amount_of_CPUs(nCores = 3):
    if( cpu_count() == 4):
        nCores = 4

    elif( int(cpu_count()*0.75) < nCores):
        nCores = int(cpu_count()*0.75)

    print('Amount of CPUs which will be used:',nCores)
    return nCores



class renamer():
    '''This renamer is used to add stars (*) to columns in dataframes when they exists as duplicates '''
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = ''
            return x
        else:
            self.d[x] += "*"
            return "%s%s" % (x, self.d[x])


def remove_mz_from_list(names):
    '''This function is called a lot of times in the pipeline! It removes names which are only values from the input list.'''
    indentified_names = []
    for name in names:
        name_stripped = name.strip('(-)').strip('*')
        try:
            float(name_stripped)
        except:
            indentified_names.append(name ) 
      
    return indentified_names

def remove_accepted_descriptions_from_list(names):
    indentified_names = []
    for name in names:
        name_stripped = name.strip('(-)').strip('*')
        try:
            float(name_stripped)
            indentified_names.append(name ) 
        except:
            pass
      
    return indentified_names

def file_name_to_batch_name(file_name, mode):
    batch_name = file_name.strip('.csv')
    if( mode in batch_name.lower() ):
        batch_name = batch_name.strip('_'+mode).strip('_'+mode.capitalize() ).strip('_'+mode.upper() )
    return batch_name




def load_ms_ms_spec_data( path ):
    '''This function can be used to load msp-files. '''

    with open( path, 'r') as f:
        ms_ms_data = [l.strip('\n') for l in f]

    y = []
    ms_ms_spec = {}
    start = False
    for line in ms_ms_data:
        if 'Comment:' in line:
            compound = line.split('Comment:')[-1].strip(' ') 
            ms_ms_spec[compound] = {}
        

        if( 'Name:' in line ):
            start = False

        if(start == True):
            spec = line.split(' ')
            if( len(spec) ==2 ):
                a,b = spec
                ms_ms_spec[compound].update({ float(a):float(b) })

        if( 'Num Peaks:' in line ):
            start = True
            
    return ms_ms_spec    





def compare_spec(spec_1,spec_2,round_precision = 3):
    '''Code from https://github.com/pymzml/pymzML/blob/master/pymzml/spec.py'''
    vector1 = ddict(int)
    vector2 = ddict(int)
    mzs = set()

    for mz, i in spec_1.items():
        vector1[round(mz, round_precision)] += i
        mzs.add(round(mz, round_precision))
    for mz, i in spec_2.items():
        vector2[round(mz, round_precision)] += i
        mzs.add(round(mz, round_precision))

    z = 0
    n_v1 = 0
    n_v2 = 0

    for mz in mzs:
        int1 = vector1[mz]
        int2 = vector2[mz]
        z += int1 * int2
        n_v1 += int1 * int1
        n_v2 += int2 * int2
        
    try:
        cosine = z / (np.sqrt(n_v1) * np.sqrt(n_v2))
    except:
        cosine = 0.0
        
    return cosine





############################## Classes #############################################
class ProgenesisData:
    adducts = {'pos': {  'M+2H':{'mass':2.0146,'Q': 2}, 
                         'M+H+Na':{'mass':23.9965, 'Q': 2},
                         'M+H-H2O':{'mass':-17.0033, 'Q': 1},
                         'M+H':{'mass':1.0073, 'Q': 1},
                         'M+2H':{'mass':1.0073*2, 'Q': 2},
                         'M-e':{'mass':-0.0005, 'Q': 1},
                         'M+Na':{'mass':22.9892, 'Q': 1},
                         'M+K':{'mass':38.9632, 'Q': 1},
                         'M+ACN+H':{'mass':42.0338, 'Q': 1}, 
                         # '2M+H':{'mass':1.0073/2, 'Q': 1/2}, # we divide mass and Q by 2 since then we get the standard form back
                         # '2M+Na':{'mass':22.9892/2, 'Q': 1/2},    
                        },


                'neg': { 'M-H':{'mass':-1.0073,'Q': 1}, 
                         'M-2H':{'mass':-1.0073*2,'Q': 2},
                         'M-H2O-H':{'mass':- 18.0106 -1.0073 ,'Q': 1},
                          
                        }
              }


    # Some metabolites are needed to be summed (see summation_of_specific_metabolites() )
    sum_metabolites = [ ('Methionine','Methioninesulfoxide'),
                        ('Glutamine','Glutamic acid / N-Methyl-D-Aspartic acid'),
                        ('Aspartic acid','Asparagine'),
                      
                      ]



    def __init__(   self, path_to_file , mode, progenesis_normalized = False, 
                    merge_duplicates_of_identified_metabs = True,
                    process_data = True, 
               
                    ):

        print(path_to_file)
        self.path_to_file = path_to_file
        self.mode = mode 
        self.progenesis_normalized = progenesis_normalized
     
        self.data = None
        self.all_IDs = None

        self.raw_data = pd.read_csv(path_to_file, delimiter=',')
        self.create_DataFrame()
        print(self.data.shape)

        if( process_data == True):
            self.make_metabolites_columns()
            self.rename_duplicate_column_names()
            self.summation_of_specific_metabolites()
            self.rename_IDs_with_E_in_name()

            if(merge_duplicates_of_identified_metabs == True):
                self.merging_duplicates_of_identified_metabolites()
                print(self.data.shape)
            print(self.data.shape)
      


    def create_DataFrame(self ):
        '''This function turns the Progenesis export into a nice format to work with. And selects whether we want the Progenesis normalized
		abundancies or raw abundancies.

		'''
        col_index_norm_abun = self.raw_data.columns.tolist().index('Normalised abundance')
        col_index_raw_abun = self.raw_data.columns.tolist().index('Raw abundance')
        col_index_tags = self.raw_data.loc[0].tolist().index('Tags')
        cols = [col for col in self.raw_data.loc[0].tolist()]


        col_index_accepted_description = self.raw_data.loc[1].tolist().index('Accepted Compound ID')
        col_index_compound_link = self.raw_data.loc[1].tolist().index('Compound Link')
        col_index_compound = self.raw_data.loc[1].tolist().index('Compound')

        # Take Progenesis normalized abundancies or raw abundancies?
        if( self.progenesis_normalized==True):
            abun_columns = self.raw_data.columns.tolist()[col_index_norm_abun :col_index_raw_abun ]
        else:
            abun_columns = self.raw_data.columns.tolist()[col_index_raw_abun :col_index_tags ]

        tags_columns = self.raw_data.columns.tolist()[col_index_accepted_description:col_index_compound_link+1]#[col_index_tags : ] 
        properties_columns = self.raw_data.columns.tolist()[col_index_compound: col_index_norm_abun ]  

        # Merge all relevant columns
        columns = [col for col in abun_columns]
        for col in tags_columns:
            columns.append(col)    
        for col in properties_columns:
            columns.append(col)

        # Get ID's from the .csv
        self.all_IDs = self.raw_data[ self.raw_data.columns.tolist()[col_index_norm_abun:col_index_raw_abun]].loc[1].tolist()

        # Make data from relevant columns
        column_names = self.raw_data[columns].loc[1].tolist()
        data = self.raw_data[columns].loc[2:]
        data.columns = column_names

        # Convert to elements to float if possible
        self.data = data.applymap(convert_to_float)

        return self.data


    def make_metabolites_columns(self):
        '''Makes the columns the features/peaks and gives names to the peaks. When a peak is annotated this will become the name.'''
        data = self.data

        def return_description_or_m_z_value(row):
            description = row['Accepted Description']
            m_z_value = row['m/z']
            if( pd.isnull(description) == True):
                return str(round(m_z_value,5))
            else:
                return description

        columns_names = data.apply(lambda row: return_description_or_m_z_value(row),axis=1  ).tolist()
        data.loc[:, 'Accepted Description'] = columns_names
        data = data.set_index('Accepted Description')
        data_with_matabs_columns = data.T

        # Remove columns having only nan numbers
        x = data_with_matabs_columns.isnull().sum(axis=0)
        nan_cols = x[ x == data_with_matabs_columns.shape[0] ].index.tolist()
        if( len(nan_cols) > 0 ):
            data_with_matabs_columns = data_with_matabs_columns.T.drop(nan_cols).T
        
        self.data = data_with_matabs_columns
        return self.data
      

    def rename_duplicate_column_names(self):
        '''Add stars to column names when they are duplicates, to make unique names in this way'''
        self.data = self.data.rename(columns=renamer())
        duplicate_columns = self.data.columns.duplicated()

        if( np.sum( duplicate_columns ) > 0):
         raise ValueError("Still duplicate columns in data ")

        return self.data 


    def merging_duplicates_of_identified_metabolites(self):
        '''Sometimes Progenesis makes separate entries for adducts from the same metabolite. Here we sum the abundancies for these separate entries and make a merged row.'''

        #Get all data with identified metabolites
        identified_data = self.data[ remove_mz_from_list( self.data.columns ) ].copy()
        metabs_stripped = np.unique( [ col.strip('*') for col in identified_data.columns]  ).tolist()

        # We sum all abundancies together and merge it to the entry having the most abundancy. 
        # We delete the other entries. 
        for metab in metabs_stripped:
            matching_cols = [col_ for col_ in identified_data.columns if col_.strip('*') == metab ]
            if( len(matching_cols) > 1 ):
                 
                y = identified_data[ matching_cols ].copy()
                sum_abundancies = y.loc[self.all_IDs].sum(axis=1)
                most_abundant_col = y.loc[self.all_IDs].median(axis=0).sort_values().iloc[-1:].index.tolist()[0]

                self.data.loc[self.all_IDs, most_abundant_col] = sum_abundancies

                # Update some information in column which is remained.
                remove_cols = [col for col in matching_cols if col != most_abundant_col]
                for col in remove_cols:
                    for info in ['Compound','Adducts']:
                        info_most_abundant_col = self.data.loc[info, most_abundant_col] # remember that this is updated in loop
                        info_col = self.data.loc[info, col]
                        self.data.loc[info, most_abundant_col] = info_most_abundant_col+','+info_col+'*'
                        print('Merging compounds and remove column:', col, info_most_abundant_col+','+info_col+'*')
                  
                    del self.data[col]



    def rename_IDs_with_E_in_name( self ):
        '''If the ID contains a "E" in their name this is sometimes problematic, since it might be seen as a scientic number format. This
        function replaces E for EE to avoid this from happening.
        '''
        f = lambda x: x.replace("E","EE") if 'E' in x else x
        rename_dict = { ID:f(ID) for ID in self.all_IDs}

        self.data = self.data.rename(index = rename_dict)
        self.all_IDs = [rename_dict[ID] for ID in self.all_IDs ]



    def summation_of_specific_metabolites(self):
        '''This function introduces new columns since we sometimes want to sum some metabolites since they react spontaneously during storage for example.'''
        data = self.data

        for pair in ProgenesisData.sum_metabolites:
            try:
                print("Sum:",pair)
                col_1 = data.loc[:,data.columns.str.strip('*') == pair[0]] 
                col_1 = col_1[col_1.columns.tolist()[0]]      
                         
                col_2 = data.loc[:,data.columns.str.strip('*') == pair[1]] 
                col_2 = col_2[col_2.columns.tolist()[0]]    
                
                data = data.assign( **{pair[0]+' + '+pair[1]:col_1 + col_2})
            except:
                print('Error in merging (likely, pair not found):',pair)

        self.data = data


    @staticmethod
    def calculate_neutral_mass(mz,mode):
        assert( mode in ProgenesisData.adducts )

        result = { }
        for adduct_name, adduct_info in ProgenesisData.adducts[mode].items():
            adduct_nm = mz*adduct_info['Q'] - adduct_info['mass'] # rounding is necessary to find overlapping neutral masses between adducts
            result[adduct_nm] = adduct_name
        return result




class MergedProgenesisData:
    '''This Class is used to merge mutiple Progenesis files. '''
    def __init__(self, files,  mz_ppm_error=1, RT_percentage_error = 5, isotope_dist_percentage_error = 20, median_error_threshold = 200, correct_retention_times = True, **kwargs):

        self.kwargs = kwargs

        self.mode = kwargs['mode']
        self.data = None
        self.files = files

        self.data_objects = [] # Possiblity to store the ProgenesisData objects
        self.all_IDs = []


        # Store some info along processing        
        self.IDs_in_batches = {}
        self.column_matches = {}
        self.not_column_matches = {}
        self.matches_from_msms = {}
        self.matches_from_identifications = {}
        self.matches_from_mz = {}
        self.matches_from_nm = {}


        ## merging settings
        self.median_error_threshold = median_error_threshold
        self.RT_percentage_error = RT_percentage_error
        self.isotope_dist_percentage_error = isotope_dist_percentage_error

        self.mz_ppm_error = mz_ppm_error
        self.correct_retention_times = correct_retention_times # True / False
 
        print("Load data sets")

        # This function calls different other functions to find matching features between the reference batch (first batch in 'files') and other batches
        # The results are stored in:
        #   1) self.matches_from_identifications
        #   2) self.matches_from_mz
        #   3) self.matches_from_nm
        #   4) self.matches_from_msms
        self.determine_overlapping_features()

        # This function combines all previously founded matches and stores it in self.column_matches.
        # We can also adjust some of the merging settings (only less strict then used in determine_overlapping_features() )
        self.determine_matching_columns( median_error_threshold = self.median_error_threshold, 
                                        isotope_dist_percentage_error = self.isotope_dist_percentage_error,
                                        RT_percentage_error = self.RT_percentage_error, 
                                        RT_percentage_error_MSMS = 2 * self.RT_percentage_error )

        # Create the merged dataset                                        
        self.merge_data_sets()
        self.check_merged_columns_on_duplicates()
        self.check_multiple_occuring_IDs()

       

    def load_data(self,file,ID_addition=None):
        '''We pass all the file paths through this function to make sure all data keeps updated'''
        obj = ProgenesisData(path_to_file=file, **self.kwargs)
        
        # Add addition to sample names to make sure we don't confuse samples with the same name in one batch with the ones in another batch
        if( ID_addition != None):
            assert( '*' in ID_addition) 
            print('Adding addition to IDs')
            obj.data = obj.data.rename(index={ID:ID+ID_addition for ID in obj.all_IDs})
            obj.all_IDs = [ ID+ID_addition for ID  in obj.all_IDs]
            
        batch_name, _ = os.path.splitext(os.path.basename( file ) )
        self.IDs_in_batches[batch_name] = obj.all_IDs
        self.data_objects.append(obj)

        return obj


    # This functions calls different other funtions which searches for overlapping metabolites/features
    def determine_overlapping_features(self):
        '''This function matches columns / features / metabolites between different datasets and stores it in self.column_matches '''

        # Load first data. This is the dataset which is to be analysed and reference batch
        obj = self.load_data( self.files[0] )
        main_data = obj.data.copy()
        main_data_IDs = [ ID for ID in obj.all_IDs ] # store IDs present in dataset
     
        # Now iterate over all the other files/batches which are needed to be merged to the reference/first dataset
        for i,file in enumerate(self.files[1:]):
            
            # Create addition to make sure IDs in different batches are unique (and we keep track of them to which batch they belong)
            addition = ''
            for j in range( i+1 ):
                # be aware that you are only allowed to add stars. Else update function Get_ID() since this function retrieves the 'pure' ID
                # Other functionalities in the pipeline might also depend on these stars!!!!! 
                addition+='*'
            
            # load dataset
            obj = self.load_data( file, ID_addition=addition )


            # We can correct the retention times of obj.data with LOWESS regression.
            if( self.correct_retention_times == True):
                print("Correct retention times for data")
                RT_corrected = MergedProgenesisData.correction_function_retention_times_with_LOWESS_regression( main_data, main_data_IDs, obj.data.copy() , obj.all_IDs, os.path.basename( obj.path_to_file )   )
                # fig.savefig( paths['path_to_processed_batches']+str(i)+'.png')
                obj.data.loc['Retention time (min)_old',:] = obj.data.loc['Retention time (min)',:]
                obj.data.loc['Retention time (min)',:] = RT_corrected


            ##### We split merging in categories ###
            # 1) on identified features ( note that all adducts for an identification are already merged for every batch in class object ProgenesisData )
            # 2  on MSMS spectra matches,neutral mass or mz-value  and retention time 
            # 3) on neutral mass and retention time 
            # 4) on mz values and retention time 
                 
            # First we match the identified metabs from both datasets with each other
            main_data_identified = main_data[ remove_mz_from_list(main_data.columns ) ].copy()
            obj_data_identified =    obj.data[ remove_mz_from_list(obj.data.columns )  ].copy()
            print("Determine identified matches")
            self.matches_from_identifications[i] = MergedProgenesisData.find_same_identifications_between_datasets( main_data_identified, 
                                                                                                                    main_data_IDs, 
                                                                                                                    obj_data_identified, 
                                                                                                                    obj.all_IDs, 
                                                                                                                    )

      

            #Secondly , we find all features having only a m/z-value and we find matching mz-values between the two datasets
            print("Determine mz matches")
            main_data_mz = main_data.loc[ :,main_data.loc['Neutral mass (Da)'].isnull()].copy()
            obj_data_mz =  obj.data.loc[ :,obj.data.loc['Neutral mass (Da)'].isnull()].copy()
            self.matches_from_mz[i] = MergedProgenesisData.find_matching_features_between_datasets( main_data_mz, 
                                                                                                    main_data_IDs, 
                                                                                                    obj_data_mz, 
                                                                                                    obj.all_IDs,
                                                                                                    ppm_error= self.mz_ppm_error, 
                                                                                                    RT_percentage_error=  self.RT_percentage_error,
                                                                                                    merge_on = 'm/z',                      
                                                                                                   )


    
            #Thirdly , we find all matching features having a neutral mass, meaning that multiple adducts has been merged already. 
            main_data_neutral = main_data.loc[ :,main_data.loc['Neutral mass (Da)'].notnull()].copy()
            obj_data_neutral =  obj.data.loc[ :,obj.data.loc['Neutral mass (Da)'].notnull()].copy()
            print("Determine neutral mass matches")
            self.matches_from_nm[i] = MergedProgenesisData.find_matching_features_between_datasets( main_data_neutral, 
                                                                                                    main_data_IDs, 
                                                                                                    obj_data_neutral, 
                                                                                                    obj.all_IDs, 
                                                                                                    ppm_error= self.mz_ppm_error, 
                                                                                                    RT_percentage_error=  self.RT_percentage_error,
                                                                                                    merge_on = 'Neutral mass (Da)',
                                                                                                    )


            #Fourthly , we find all matching features based on MS/MS spectra similarity
            print("Determine MSMS matches")
            self.matches_from_msms[i] = MergedProgenesisData.find_matching_features_between_datasets_from_MSMS_spectra( self.files[0],  main_data, main_data_IDs, 
                                                                                                                        file, obj.data, obj.all_IDs, 
                                                                                                                        mode = self.mode ,
                                                                                                                        ppm_error= self.mz_ppm_error, 
                                                                                                                        RT_percentage_error= self.RT_percentage_error * 2, # we add more window since we have MSMS as extra info, so we are more rolerant here
                                                                                                                        msms_spec_similarity = 0.8 
                                                                                                                        )



    # In determine_overlapping_features() we searched all matches features/metabolites per batch with reference batch
    # The following function combines these findings. 
    def determine_matching_columns(self, median_error_threshold = 100, isotope_dist_percentage_error = 5, RT_percentage_error = 5, RT_percentage_error_MSMS = 10):

        for i,file in enumerate(self.files[1:]):
            print(i,file)

            obj_1 = self.data_objects[0]
            obj_2 = self.data_objects[i+1]

            data_1 = obj_1.data.copy()
            data_2 = obj_2.data.copy()
            IDs_1 = obj_1.all_IDs
            IDs_2 = obj_2.all_IDs

            assert( obj_2.path_to_file == file)

            # Add identification matches
            column_matches = self.matches_from_identifications[i].copy()


            # Add MSMS matches
            df = self.matches_from_msms[i]
            if( not df.empty ):
                df = df.loc[ ~df['columns_1'].isin(column_matches['columns_1']) ]

                # Filter median errors
                df = df.assign(RT_error = (df['RT_1'] - df['RT_2']).abs() / df['RT_1'] * 100 )
                df = df.loc[ df['RT_error'] <= RT_percentage_error_MSMS ]
                df = df.loc[ df['median_error'] <= median_error_threshold ]

                column_matches = pd.concat([column_matches,df])



            # Add neutral mass matches
            df = self.matches_from_nm[i]
            if( not df.empty ):
                df = df.loc[ ~df['columns_1'].isin(column_matches['columns_1']) ]

                # Filter median errors and isotopes
                df = df.assign(RT_error = (df['RT_1'] - df['RT_2']).abs() / df['RT_1'] * 100 )
                df = df.loc[ df['RT_error'] <= RT_percentage_error ]
                df = df.loc[ df['median_error'] <= median_error_threshold ]
                df = df.assign( IsDi_errors_check = df['IsDi_errors'].apply(lambda x: np.sum([ 0 if el < isotope_dist_percentage_error else 1 for el in x ] ) ))
                df = df.loc[ df['IsDi_errors_check'] == 0 ]

                column_matches = pd.concat([column_matches,df])


            # Add mz matches
            df = self.matches_from_mz[i]
            if( not df.empty ):
                df = df.loc[ ~df['columns_1'].isin(column_matches['columns_1']) ]

                # Filter median errors and isotopes
                df = df.assign(RT_error = (df['RT_1'] - df['RT_2']).abs() / df['RT_1'] * 100 )
                df = df.loc[ df['RT_error'] <= RT_percentage_error ]
                df = df.loc[ df['median_error'] <= median_error_threshold ]
                df = df.assign(IsDi_errors_check = df['IsDi_errors'].apply(lambda x: np.sum([ 0 if el < isotope_dist_percentage_error else 1 for el in x ] ) ))
                df = df.loc[ df['IsDi_errors_check'] == 0 ]

                column_matches = pd.concat([column_matches,df])

            # Throw away double identified features
            cols_count_1  = column_matches['columns_1'].value_counts()
            cols_1 = cols_count_1[ cols_count_1 == 1].index.tolist()
            cols_count_2  = column_matches['columns_2'].value_counts()
            cols_2 = cols_count_2[ cols_count_2 == 1].index.tolist()

            column_matches = column_matches.loc[ column_matches['columns_1'].isin(cols_1) &  column_matches['columns_2'].isin(cols_2) ]

            self.column_matches[i] = column_matches



    def merge_data_sets(self):
        '''After determine_overlapping_features() is called we can merge the data.'''
        obj = self.data_objects[0]
        merged_data = obj.data.copy()
        self.all_IDs      = [ID for ID in obj.all_IDs]

        for i,file in enumerate(self.files[1:]):
            obj = self.data_objects[i+1]
            # We pass the founded matches to the following function and merge all the 
            merged_data, not_merged = MergedProgenesisData.merge_datasets_on_matching_features( merged_data, 
                                                                                                self.all_IDs, 
                                                                                                obj.data.copy(), 
                                                                                                obj.all_IDs,
                                                                                                self.column_matches[i], #column_matches contains all the information needed to merge the features
                                                                                               )
         
            self.not_column_matches[i] = not_merged
            self.all_IDs = list( set(self.all_IDs).union( set( obj.all_IDs ) ))

        self.data = merged_data
    


    def merge_datasets_on_matching_features(data_1, IDs_1, data_2, IDs_2, info_column_matches ):
        '''This function returns the merged data with the given input. Note, that info_column_matches is determined from within the class itself but this
           set-up allows for other uses also.
        '''

        # Make sure we will only match one feature with one other feature
        assert( (info_column_matches['columns_1'].unique().shape[0] == info_column_matches.shape[0]) and (info_column_matches['columns_2'].unique().shape[0] == info_column_matches.shape[0]))

        # Check extra if all columns are still present in data_1. This is not necessary true since maybe we already merged some data and thus some columns are removed.
        info_column_matches = info_column_matches.loc[ info_column_matches['columns_1'].isin(data_1.columns) ]

        # Select data from set 1 and 2 for merge
        x_1 = data_1.loc[IDs_1,info_column_matches['columns_1']]
        x_2 = data_2.loc[IDs_2,info_column_matches['columns_2']]

        columns_1_not_merged =  list( set(remove_mz_from_list(data_1.columns)).difference( 
                                                set(remove_mz_from_list(info_column_matches['columns_1']))
                                                )
                                        )

        columns_2_not_merged =  list( set(remove_mz_from_list(data_2.columns)).difference( 
                                                set(remove_mz_from_list(info_column_matches['columns_2']))
                                                )
                                        )

        not_merged = {'columns_1':columns_1_not_merged,'columns_2':columns_2_not_merged}

        # We use the information of data_1 as base, this will be used in the newly obtained merged dataset
        RT_mz_12 = data_1.loc[['m/z','Retention time (min)','Isotope Distribution','Accepted Compound ID','Neutral mass (Da)','Compound','Adducts'],info_column_matches['columns_1']]

        # Get IDs 
        ind = [ ID for ID in x_1.index.tolist()]
        ind.extend([ ID for ID in x_2.index.tolist()])
        # Concat data. We don't need to check column order since we have already the right order of columns. So we just use numpy concat r_ function. This is a faster than use pandas concat
        data_merged = pd.DataFrame( np.r_[x_1.values,x_2.values],columns=x_1.columns,index=ind )
        assert( (x_1.shape[1] == data_merged.shape[1]) and (x_2.shape[1] == data_merged.shape[1]) )

        # Add RT, isotope distribution and mz values to the merged data
        data_merged = pd.concat([RT_mz_12,data_merged])

        return (data_merged,not_merged)




    def check_multiple_occuring_IDs(self):
        print("Check if all IDs are unique ")
        if( self.data.index.duplicated().any() ):
            raise ValueError("IDs are not unique!")
        else:
            print("Ok")
            

    def check_merged_columns_on_duplicates(self):
        print("Check duplicate columns")
        data = self.data
        duplicate_columns = data.columns.get_duplicates()
        if(len(duplicate_columns) > 0):
            print("Still duplicate columns in data ")
            self.data = data.rename(columns=renamer())
            print("Duplicates removed")
        else:
            print("Ok")



    def correction_function_retention_times_with_LOWESS_regression( data_1, IDs_1, data_2, IDs_2, name_data_2 ):
        '''This function returns the fitted LOWESS regression curve on RT_1 as function of RT_2.'''

        fig = plt.figure(figsize=(18,6))
        fig.subplots_adjust(wspace=0.5)
   
        Xs = []
       
        for data, IDs in zip([data_1, data_2],[IDs_1, IDs_2] ) :
      
            med = data.loc[IDs].median(axis=0).apply(lambda x: x)
            med = ( med - med.mean() ) / med.std() 
            med.name= 'median'
            rt = data.loc["Retention time (min)"]
            rt.name = 'RT'
            mz = data.loc["m/z"]
            mz.name = 'mz'
            X = pd.DataFrame([med,rt,mz]).T.sort_values(by='RT')
            Xs.append(X)


        X_ref =  Xs[0].sort_values(by='median',ascending = False).iloc[0:]
        X_ref.columns = [col+'_ref' for col in X_ref.columns]
        X_ref = X_ref.assign(mz_rounded = X_ref['mz_ref'].round(3))
        X_ref = X_ref.assign(ID = range(X_ref.shape[0] ) ) 
        X_ref.index.name = 'Metab_ref'
        X_ref = X_ref.reset_index()

        X_merge_tot = pd.DataFrame([])
            
        # We search iterative to matching peaks between the batches. We go from more certain peaks to less certain peaks. But when certain peaks are 
        # already taken previously they can not be taken the next iterations.
        amount = []
		# We increase per iteration the amount of peaks, and the tolerances for matching.
        for rt_error,med_error,peaks in zip(np.linspace(0,10,100),
                                            np.linspace(80,20,100),
                                            [ int(x) for x in np.round(np.linspace(20000,18000,100)) ] ):           

			# Select the most abundant peaks 
            X_cur =  Xs[1].sort_values(by='median',ascending = False).iloc[0:peaks]
            X_cur.index.name = 'Metab'
            X_cur = X_cur.reset_index()
            X_cur = X_cur.assign(mz_rounded = X_cur['mz'].round(3))

			# Merge on rounded mz-values
            X_merge = X_ref.merge(X_cur, on='mz_rounded')
			# Calculate ppm error and remove matches when error is too high
            X_merge = X_merge.assign(ppm = (X_merge['mz_ref'] - X_merge['mz'] ).abs() / X_merge['mz_ref'] *10**6 )
            X_merge = X_merge.loc[X_merge['ppm'] < 1]
			# Calculate median abundancy error and remove when error is too high
            X_merge = X_merge.assign(error_med = ((X_merge['median_ref'] - X_merge['median']  )/ X_merge['median_ref'] *100 ).abs() )
            X_merge = X_merge.loc[ X_merge['error_med'] < med_error ]
			# Calculate RT difference and remove when difference is too big
            X_merge = X_merge.assign(error_rt = (X_merge['RT_ref'] - X_merge['RT'] ).abs() / X_merge['RT_ref'] *100)
            X_merge = X_merge.loc[ X_merge['error_rt'] < rt_error ]

            X_merge = X_merge.loc[ ~X_merge['ID'].duplicated() ]

            if( X_merge_tot.empty):
                ID_already_taken= []
            else:
                ID_already_taken = X_merge_tot['ID'].tolist()

            # Only take peaks which are not yet matched previously
            X_merge = X_merge.loc[ ~X_merge['ID'].isin(ID_already_taken) ]
            X_merge_tot = pd.concat([X_merge_tot, X_merge])
            
            amount.append(X_merge_tot.shape[0])
          

        RT_1, RT_2 = X_merge_tot['RT_ref'], X_merge_tot['RT']
        overlap_metabs  = X_cur.loc[ X_cur['Metab'].str.lower().isin(remove_mz_from_list(X_ref['Metab_ref'].str.lower())) ]['Metab'].tolist()

        ax = fig.add_subplot(1,3,1  )
        ax.scatter( RT_1, RT_2 , color='navy')
        # plot annotated peaks which are present in both datasets
        ax.scatter( X_ref.set_index('Metab_ref').loc[overlap_metabs]['RT_ref'], X_cur.set_index('Metab').loc[overlap_metabs]['RT'], marker='x', s=100, c='darkred' )
        ax.set_xlabel( 'RT_1')
        ax.set_ylabel( 'RT_2')
        ax.set_title('Retention times for dataset 1 (ref batch) and dataset 2')
            
		# Fit lowess regression on the matching peaks
        lowess = sm.nonparametric.lowess( RT_1, RT_2, frac=1 / 3.5 )
        lowess_x = np.array(list(zip(*lowess))[1])
        lowess_y = np.array(list(zip(*lowess))[0])
        f_correction = interpolate.interp1d(lowess_x, (lowess_x - lowess_y), bounds_error=False)
        
        ax = fig.add_subplot(1,3,2  )
        ax.scatter(lowess_x,(lowess_x - lowess_y),s=10,c='darkred',alpha=0.5)
        ax.plot( np.linspace(0,max(lowess_x),100),f_correction(np.linspace(0,max(lowess_x),100).tolist() ) ,color='navy')
        ax.set_xlabel( 'RT_1')
        ax.set_ylabel( 'RT_1 - RT_2')
        ax.set_title('Difference in retention time')

        RT =  data_2.loc['Retention time (min)'].copy()
        RT_corr = (RT + f_correction(RT.tolist())).copy()
        nan_ind = RT_corr[RT_corr.isnull()].index
        RT_corr.loc[nan_ind] = RT.loc[nan_ind]

        ax = fig.add_subplot(1,3,3  )
        ax.plot( range( len(amount) ), amount, color = 'navy' )
        ax.set_xlabel( 'Iteration')
        ax.set_ylabel( 'Amount of matching peaks')

        ax.set_title(name_data_2)

        fig.savefig( str(round(time.time()))+'.png' )
        
        return RT_corr
        


    def find_same_identifications_between_datasets(data_1, IDs_1, data_2, IDs_2):
 
        # We clean and check the availability of some data in the input datasets 
        def check_clean_data(data, IDs):
            assert( ('m/z' in data.index) and ('Retention time (min)' in data.index) and ('Isotope Distribution' in data.index) )
            ind = [ID for ID in IDs]
            ind.extend(['Retention time (min)','m/z','Isotope Distribution','Accepted Compound ID','Neutral mass (Da)','Compound','Adducts'])
            return data.loc[ind]

        data_1 = check_clean_data(data_1,IDs_1)
        data_2 = check_clean_data(data_2,IDs_2)

        info_1 = data_1.loc[['Retention time (min)','m/z','Neutral mass (Da)','Isotope Distribution','Adducts','Compound']].T.reset_index()
        info_1.columns  = ['columns_1','RT_1','mz_1','nm_1','IsDi_1','Adducts_1','Compound_1']
        info_1 = info_1.loc[ info_1['columns_1'].isin(remove_mz_from_list(data_1.columns)) ]

        info_2 = data_2.loc[['Retention time (min)','m/z','Neutral mass (Da)','Isotope Distribution','Adducts','Compound']].T.reset_index()
        info_2.columns  = ['columns_2','RT_2','mz_2','nm_2','IsDi_2','Adducts_2','Compound_2']
        info_2 = info_2.loc[ info_2['columns_2'].isin(remove_mz_from_list(data_2.columns)) ]

        # We strip the names from the metabolites and make them lowercase such that matches between the two datasets can be easilier established.
        identified_1 = { col:col.strip("*").strip(' ').lower()    for col in  remove_mz_from_list(data_1.columns) }
        identified_2 = { col:col.strip("*").strip(' ').lower()    for col in  remove_mz_from_list(data_2.columns) }

        info_1 = info_1.assign(column_stripped = info_1['columns_1'].apply(lambda x: identified_1[x]) )
        info_2 = info_2.assign(column_stripped = info_2['columns_2'].apply(lambda x: identified_2[x]) )

        # Find overlapping metabolites 
        overlap = set(identified_1.values()).intersection( set(identified_2.values()))

        # Merge when the column 'column_stripped' is the same
        info_column_matches = info_1.merge(info_2, on = 'column_stripped')
        assert( len(overlap) == info_column_matches.shape[0])

        # Get median abundancies data_1
        x_1 = data_1.loc[IDs_1].median(axis=0)
        x_1.index.name = 'columns_1'
        x_1.name = 'median_abun_1'
        x_1 = x_1.reset_index()

        # Get median abundancies data_2
        x_2 = data_2.loc[IDs_2].median(axis=0)
        x_2.index.name = 'columns_2'
        x_2.name = 'median_abun_2'
        x_2 = x_2.reset_index()

        info_column_matches = info_column_matches.merge(x_1, on ='columns_1').merge(x_2, on ='columns_2')
        info_column_matches = info_column_matches.assign(median_error = ((info_column_matches['median_abun_1'] - info_column_matches['median_abun_2']).abs() / info_column_matches['median_abun_1'] *100 ).round() )

        return info_column_matches


    def find_matching_features_between_datasets( data_1, IDs_1, data_2, IDs_2, ppm_error=1, RT_percentage_error= 5, merge_on = 'm/z' ):
        '''Finding matching features is based on four criteria: 1) ppm_error 2) retention time error 3) isotope distribution  '''

        rename_merge_on_dict = { 'Neutral mass (Da)':'nm', 'm/z':'mz'}

        assert( data_1.shape[0] > 2 and data_2.shape[0] > 2 )
        print(data_1.shape,data_2.shape)


        assert( merge_on in ['m/z', 'Neutral mass (Da)'] )

        # Get amount of CPUs for parrallel processing statistics
        nCores = get_amount_of_CPUs()

        # We clean and check the availability of some data in the input datasets 
        def check_clean_data(data, IDs):
            assert( ('m/z' in data.index) and ('Retention time (min)' in data.index) and ('Isotope Distribution' in data.index) )
            ind = [ID for ID in IDs]
            ind.extend(['Retention time (min)','m/z','Isotope Distribution','Accepted Compound ID','Neutral mass (Da)','Compound','Adducts'])
            return data.loc[ind]

        data_1 = check_clean_data(data_1,IDs_1)
        data_2 = check_clean_data(data_2,IDs_2)

        # Realize that we merge on m/z. Since we also have merged adducts if a neutral mass is known we need to be careful here.
        # However, you can observe that the m/z belonging to the neutral mass is 'always' M+H. So we can still safely match neutrall masses on m/z values
        info_1 = data_1.loc[['Retention time (min)',merge_on,'Isotope Distribution','Adducts','Compound']].T.reset_index()
        info_1.columns  = ['columns_1','RT_1',rename_merge_on_dict[merge_on]+'_1','IsDi_1','Adducts_1','Compound_1']

        info_2 = data_2.loc[['Retention time (min)',merge_on,'Isotope Distribution','Adducts','Compound']].T.reset_index()
        info_2.columns  = ['columns_2','RT_2',rename_merge_on_dict[merge_on]+'_2','IsDi_2','Adducts_2','Compound_2']


        def search_metabs_in_RT(info_1_sub):
            info_column_matches = pd.DataFrame([])

            for i,(ind,row) in enumerate(info_1_sub.iterrows()):
                
                if(i%500==0): print(i)
                df_ = pd.DataFrame(row).T
                df_ = df_.assign(interval_id=0)

                # Make interval for searching
                df__ = info_2.loc[ (info_2['RT_2'] - row['RT_1'] ).abs() <= row['RT_1'] * (RT_percentage_error / 100) ].copy()
                df__ = df__.assign(interval_id=0)
                df = df_.merge(df__,on='interval_id')

                # Determine ppm error and remove metabs not satisfying the allowed ppm error 
                df = df.assign(ppm_error=(df[rename_merge_on_dict[merge_on]+'_2'] - df[rename_merge_on_dict[merge_on]+'_1']).abs().divide(df[rename_merge_on_dict[merge_on]+'_1'])*10**6)
                df = df.loc[ df['ppm_error'] <= ppm_error ]
              

                info_column_matches = pd.concat([info_column_matches,df])

            info_column_matches = info_column_matches.rename(columns = {'ppm_error': 'ppm_error_'+rename_merge_on_dict[merge_on] })
            return info_column_matches[['columns_1','columns_2','Compound_1','Compound_2','RT_1','RT_2',rename_merge_on_dict[merge_on]+'_1',rename_merge_on_dict[merge_on]+'_2','IsDi_1','IsDi_2','ppm_error_'+rename_merge_on_dict[merge_on], 'Adducts_1','Adducts_2']]

        # Split searching of metabs in descrite block used for parrallel computing
        info_1 = info_1.reset_index(drop=True)


        # Divide data into blocks
        block_size = int(info_1.shape[0]/nCores)
        for i in range(nCores -1):
             info_1.loc[ block_size*i: block_size*(i+1),'block'] = i
        info_1.loc[ block_size*(i+1): ,'block'] = i+1


        print("Start searching metabs in RT-window")
        # We put all the sliced data into Dask delayed function so we can process the block parrallel over different processors
        output = []
        for block, info_1_sub in info_1.groupby('block'):
             output.append( dask.delayed(search_metabs_in_RT)(info_1_sub) )

        # Concat the results of the individual processed blocks
        def concat_dfs(dfs):
            df = pd.DataFrame([])
            for df_ in dfs:
                df = pd.concat([df,df_]) 
            return df

        # Tell Dask first to parrallel process the blocks (output var) and then apply the concat_dfs function
        total = dask.delayed(concat_dfs)(output)

        info_column_matches = total.compute( scheduler='processes',num_workers = nCores)
        info_column_matches = info_column_matches.reset_index(drop=True)

        # The following function will determine the difference between the isotope distribution peaks and takes the maximum difference for further evaluation
        def check_isotope_similarity(a,b, penalty = 5):
            try:

                a = np.array([ float(el) for el in str(a).split('- ') ])
                b = np.array([ float(el) for el in str(b).split('- ') ])

                max_length = max(len(a),len(b))

                if( len(a) == len(b) ):
                    pass

                elif( len(a) >= len(b) ):
                    b = np.append(b, np.zeros( max_length - len(b) ) )

                else:
                    a = np.append(a, np.zeros( max_length - len(a) ) )

                return list( np.round( np.abs(a - b) / ( a+penalty ) *100,2) ) # We add a penalty such that small numbers don't result in mismatching isotope patterns

            except Exception as e:
                print(e)
                return [100]


        # For some reason the apply function raised an error here so I used a loop to determine the isotope similarities.
        val = []
        index = []
        for ind,row in info_column_matches.iterrows():
            a,b = row.loc['IsDi_1'],row.loc['IsDi_2']
            index.append(ind)
            val.append( check_isotope_similarity(a,b) )

        IsDi_errors = pd.Series(val, index = index)
        info_column_matches = info_column_matches.assign(IsDi_errors = IsDi_errors )


        # Get median abundancies data_1
        x_1 = data_1.loc[IDs_1].median(axis=0)
        x_1.index.name = 'columns_1'
        x_1.name = 'median_abun_1'
        x_1 = x_1.reset_index()

        # Get median abundancies data_2
        x_2 = data_2.loc[IDs_2].median(axis=0)
        x_2.index.name = 'columns_2'
        x_2.name = 'median_abun_2'
        x_2 = x_2.reset_index()


        info_column_matches = info_column_matches.merge(x_1, on ='columns_1').merge(x_2, on ='columns_2')
        info_column_matches = info_column_matches.assign(median_error = ((  info_column_matches['median_abun_1'] - 
                                                                            info_column_matches['median_abun_2']).abs() / info_column_matches['median_abun_1'] *100 ).round() )


        cols_count_1  = info_column_matches['columns_1'].value_counts()
        cols_1 = cols_count_1[ cols_count_1 == 1].index.tolist()
        cols_count_2  = info_column_matches['columns_2'].value_counts()
        cols_2 = cols_count_2[ cols_count_2 == 1].index.tolist()

        info_column_matches = info_column_matches.loc[ info_column_matches['columns_1'].isin(cols_1) &  info_column_matches['columns_2'].isin(cols_2) ]


        return info_column_matches




    def find_matching_features_between_datasets_from_MSMS_spectra( path_file_1, data_1, IDs_1, path_file_2, data_2, IDs_2, mode, ppm_error=1, RT_percentage_error = 5, 
                                                                   msms_spec_similarity = 0.8 ):


        try:
            # Search the MSMS files which are supossed the have the same path and almost same name as the Progenesis file
            path_file_1 = path_file_1.replace('_UMETA_','_MS2_').replace('csv','msp')
            path_file_2 = path_file_2.replace('_UMETA_','_MS2_').replace('csv','msp')

            # Load files
            ms_ms_spec_1 = load_ms_ms_spec_data( path_file_1 )
            ms_ms_spec_2 = load_ms_ms_spec_data( path_file_2 )


            # When compounds are merged earlier we might first want to unfold them 'back' since MSMS data is stored on the compound ID of progenesis
            # A star in the compound  (ID) indicates that two compounds are merged in the pipeline earlier.
            df_1 = data_1.loc[['Compound','Retention time (min)','Neutral mass (Da)','m/z'], ~data_1.loc['Compound'].str.contains('*',na=False,regex=False) ].T
            df_1 = df_1.assign( Compound_1 = df_1['Compound'])
            df_1.index.name = 'columns_1'
            df_1 = df_1.reset_index()

            # unfold 
            for col,row in data_1.loc[['Compound','Retention time (min)','Neutral mass (Da)','m/z'], data_1.loc['Compound'].str.contains('*',na=False,regex=False) ].T.iterrows():
                merged_compounds = [ el.strip('*') for el in row.loc['Compound'].split(',')]
                for compound in merged_compounds:
                    df_1 = df_1.append(row.append(pd.Series({'columns_1':col,'Compound_1':compound}) ),ignore_index=True)

                        
            df_1 =  df_1.loc[ df_1['Compound_1'].isin(ms_ms_spec_1.keys()) ]
            df_1 = df_1.rename(columns = {'Retention time (min)':'RT_1','Neutral mass (Da)':'nm_1','m/z':'mz_1'})


            df_2 = data_2.loc[['Compound','Retention time (min)','Neutral mass (Da)','m/z'], ~data_2.loc['Compound'].str.contains('*',na=False,regex=False) ].T
            df_2 = df_2.assign( Compound_2 = df_2['Compound'])
            df_2.index.name = 'columns_2'
            df_2 = df_2.reset_index()

            # unfold 
            for col,row in data_2.loc[['Compound','Retention time (min)','Neutral mass (Da)','m/z'], data_2.loc['Compound'].str.contains('*',na=False,regex=False) ].T.iterrows():
                merged_compounds = [ el.strip('*') for el in row.loc['Compound'].split(',')]
                for compound in merged_compounds:
                    df_2 = df_2.append(row.append(pd.Series({'columns_2':col,'Compound_2':compound}) ),ignore_index=True)

            df_2 =  df_2.loc[ df_2['Compound_2'].isin(ms_ms_spec_2.keys()) ]
            df_2 = df_2.rename( columns = {'Retention time (min)':'RT_2','Neutral mass (Da)':'nm_2','m/z':'mz_2'})    


            # Search features in retention time window
            df = pd.DataFrame([])
            for i,row in df_1.iterrows():
                row_ = pd.DataFrame(row).T
                row_ = row_.assign(merge=1)
                RT = row_['RT_1'].values[0]

                row_= row_.merge( df_2.loc[ (df_2['RT_2'] - RT).abs() < RT * ( RT_percentage_error / 100 ) ].assign(merge=1), on= 'merge', how='left')
                df = pd.concat([df,row_])

            del df['merge']

            # Get median abundancies data_1
            x_1 = data_1.loc[IDs_1].median(axis=0)
            x_1.index.name = 'columns_1'
            x_1.name = 'median_abun_1'
            x_1 = x_1.reset_index()

            # Get median abundancies data_2
            x_2 = data_2.loc[IDs_2].median(axis=0)
            x_2.index.name = 'columns_2'
            x_2.name = 'median_abun_2'
            x_2 = x_2.reset_index()

            df = df.merge(x_1, on ='columns_1').merge(x_2, on ='columns_2')
            df = df.assign( median_error = ((df['median_abun_1'] - df['median_abun_2']).abs()/ df['median_abun_1'] *100 ).round() )


            # We need to make sure that a feature in one batch having a neutral mass while the other has not, can still be merged. 
            # When a neutral mass is known, and the other is only a m/z value we need to get the same neutral mass from the m/z value when we use a certain adduct
            df = df.assign( ppm_error_mz =  (( df['mz_1'] - df['mz_2']).abs() / df['mz_1'] *10**6 ) )
            df = df.assign( ppm_error_nm =  (( df['nm_1'] - df['nm_2']).abs() / df['nm_1'] *10**6 ) )

         
            # We split the set in 4 parts. sometimes we only have a mz-values and sometimes we have a neutral mass as well.
            # We allow a mz-value to be merged with an neutral mass only when the MSMS spectra are very similar and the median abundance error is not to high

            # We calculate all possible neutral masses from the adducts we know to check if the match with the neutral mass is a real match.
            df_1 = df.loc[ df['nm_1'].notnull() & df['nm_2'].isnull() ]
            df_1 = df_1.assign(possible_nms_2 = df_1['mz_2'].apply(lambda x: list( ProgenesisData.calculate_neutral_mass(x, mode = mode ).keys() ) ) )
            df_1 = df_1.assign(min_ppm_error_possible_nms_2 = df_1.apply(lambda x: min([ np.abs((el - x['nm_1']))/x['nm_1']*10**6 for el in x['possible_nms_2']]), axis=1) )                 
            df_1 = df_1.loc[ df_1['min_ppm_error_possible_nms_2'] < 1]

            df_2 = df.loc[ df['nm_1'].isnull() & df['nm_2'].notnull() ]
            df_2 = df_2.assign(possible_nms_1 = df_2['mz_1'].apply(lambda x: list( ProgenesisData.calculate_neutral_mass(x, mode = mode ).keys() ) ) )
            df_2 = df_2.assign(min_ppm_error_possible_nms_1 = df_2.apply(lambda x: min([ np.abs((el - x['nm_2']))/x['nm_2']*10**6 for el in x['possible_nms_1']]), axis=1) )
            df_2 = df_2.loc[ df_2['min_ppm_error_possible_nms_1'] < 1]

            df_3 = df.loc[ df['nm_1'].isnull() & df['nm_2'].isnull() ]
            df_3 = df_3.loc[ df_3['ppm_error_mz'] < 1]

            df_4 = df.loc[ df['nm_1'].notnull() & df['nm_2'].notnull() ]
            df_4 = df_4.loc[ df_4['ppm_error_nm'] < 1]

            df = pd.concat([df_1, df_2, df_3, df_4])

			
            metabs_1 = df['Compound_1'].unique().tolist()
            gb_compound_1 = df.groupby('Compound_1')
            similarity_df = pd.DataFrame([])
            for i,metab_1 in enumerate(metabs_1):
                result = []
                for metab_2 in gb_compound_1.get_group(metab_1)['Compound_2'].tolist():
                    result.append( [metab_1,metab_2,compare_spec( ms_ms_spec_1[metab_1],ms_ms_spec_2[metab_2] ,3) ] )

                result = pd.DataFrame(result).sort_values(by=2,ascending=False)
                result.columns = ['Compound_1','Compound_2','similarity_score']
                similarity_df = pd.concat([similarity_df, result])


            df = df.merge(similarity_df, on =['Compound_1','Compound_2'])
            df = df.loc[ df['similarity_score'] > msms_spec_similarity] # score of MSMS which is needed for a match


            cols_count_1  = df['columns_1'].value_counts()
            cols_1 = cols_count_1[ cols_count_1 == 1].index.tolist()
            cols_count_2  = df['columns_2'].value_counts()
            cols_2 = cols_count_2[ cols_count_2 == 1].index.tolist()

            df = df.loc[ df['columns_1'].isin(cols_1) &  df['columns_2'].isin(cols_2) ]

            return df 

        except Exception as e:
            print("Cannot find the MSMS-file for {} or {}. Make sure the names are correct and in the same directory!".format(path_file_1, path_file_2) )
            df = pd.DataFrame([],columns=['columns_1','similarity_score'])
            return df 




