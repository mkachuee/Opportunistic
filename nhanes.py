import pdb
import glob
import copy
import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.feature_selection

class FeatureColumn:
    def __init__(self, category, field, preprocessor, args=None, cost=None):
        self.category = category
        self.field = field
        self.preprocessor = preprocessor
        self.args = args
        self.data = None
        self.cost = cost

class NHANES:
    def __init__(self, db_path=None, columns=None):
        self.db_path = db_path
        self.columns = columns # Depricated
        self.dataset = None # Depricated
        self.column_data = None
        self.column_info = None
        self.df_features = None
        self.df_targets = None
        self.costs = None

    def process(self):
        df = None
        cache = {}
        # collect relevant data
        df = []
        for fe_col in self.columns:
            sheet = fe_col.category
            field = fe_col.field
            data_files = glob.glob(self.db_path+sheet+'/*.XPT')
            df_col = []
            for dfile in data_files:
                print(80*' ', end='\r')
                print('\rProcessing: ' + dfile.split('/')[-1], end='')
                # read the file
                if dfile in cache:
                    df_tmp = cache[dfile]
                else:
                    df_tmp = pd.read_sas(dfile)
                    cache[dfile] = df_tmp
                # skip of there is no SEQN
                if 'SEQN' not in df_tmp.columns:
                    continue
                #df_tmp.set_index('SEQN')
                # skip if there is nothing interseting there
                sel_cols = set(df_tmp.columns).intersection([field])
                if not sel_cols:
                    continue
                else:
                    df_tmp = df_tmp[['SEQN'] + list(sel_cols)]
                    df_tmp.set_index('SEQN', inplace=True)
                    df_col.append(df_tmp)

            try:
                df_col = pd.concat(df_col)
            except:
                #raise Error('Failed to process' + field)
                raise Exception('Failed to process' + field)
            df.append(df_col)
        df = pd.concat(df, axis=1)
        #df = pd.merge(df, df_sel, how='outer')
        # do preprocessing steps
        df_proc = []#[df['SEQN']]
        for fe_col in self.columns:
            field = fe_col.field
            fe_col.data = df[field].copy()
            # do preprocessing
            if fe_col.preprocessor is not None:
                prepr_col = fe_col.preprocessor(df[field], fe_col.args)
            else:
                prepr_col = df[field]
            # handle the 1 to many
            if (len(prepr_col.shape) > 1):
                fe_col.cost = [fe_col.cost] * prepr_col.shape[1]
            else:
                fe_col.cost = [fe_col.cost]
            df_proc.append(prepr_col)
        self.dataset = pd.concat(df_proc, axis=1)
        return self.dataset
    
    def index(self, renew_cache=False):
        # check if we don't have to renew cache
        cache_path = self.db_path + 'cache/index_cache.pkl'
        if (not renew_cache) and (os.path.exists(cache_path)):
            print('Loading from cache:', cache_path)
            try:
                with open(cache_path, 'rb') as f:
                    self.column_data, self.column_info = pickle.load(f)
            except:
                self.column_data, self.column_info = joblib.load(cache_path)
            return
        # indexed cache file
        self.column_data = {}
        self.column_info = {}
        # get the list of all sheets
        sheets = [p.split('/')[-1] for p in glob.glob(self.db_path+'/*')]
        # for each sheet read and index each data-file
        for sheet in sheets:
            dfiles = glob.glob(self.db_path+sheet+'/*.XPT')
            for dfile in dfiles:
                print('\rProcessing:', dfile, end='')
                df = pd.read_sas(dfile)
                if 'SEQN' not in df.columns:
                    continue
                # read file columns and index them
                df.set_index('SEQN', drop=True, inplace=True)
                if not df.index.is_unique:
                    continue
                for col in df.columns:
                    # if the column is not cached ever
                    if col not in self.column_data:
                        self.column_data[col] = df.loc[:,[col]]
                        # ignore duplicates
                        self.column_data[col] = self.column_data[col].groupby(level=0).last()
                        # get column info
                        self.update_column_info(col, dfile)
                    # else, we have cached info for this column
                    else:
                        # merge them
                        self.column_data[col] = pd.concat(
                            [self.column_data[col], df.loc[:,[col]]],
                            axis=0, verify_integrity=False)
                        # ignore duplicates
                        #if not self.column_data[col].index.is_unique:
                        #    pdb.set_trace()
                        #self.column_data[col] = self.column_data[col].groupby(level=0).last()            
                        self.column_data[col] = self.column_data[col][~self.column_data[col].index.duplicated(keep=False)]
        # store/update cache file
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print('\rStoring to cache:', cache_path)
        try:
            with open(cache_path, 'wb+') as f:
                pickle.dump((self.column_data, self.column_info), f)
        except:
            joblib.dump((self.column_data, self.column_info), cache_path, compress=9)
        return
    
    def process_supervised(self, target_col, exclude_cols, include_cols=None,
                           preproc_target=None, preproc_target_args=None,
                           missing_threshold=1.0, muinfo_threshold=0.0):
        
        # remove missing target values
        if type(target_col) is list:
            self.df_targets = self.column_data[target_col[0]]
            for t_col in target_col[1:]:
                self.df_targets = pd.concat([self.df_targets, self.column_data[t_col]], 
                                  axis=1, join='inner')
        else:
            self.df_targets = self.column_data[target_col].copy()
        self.df_targets.dropna(axis=0, how='any', inplace=True)
        # get target dataframe
        #if type(preproc_target) is tuple:   
        if preproc_target != None:
            self.df_targets = preproc_target(self.df_targets, preproc_target_args)
            self.df_targets.dropna(axis=0, how='any', inplace=True)
            
        # process features
        self.df_features = pd.DataFrame()
        for col in self.column_data.keys():
            print(80*' ', end='\r')
            print('Processing:', col, end='\r')
            # check if we should skip this column
            if col in exclude_cols or col == target_col:
                continue
            if self.column_data[col].dtypes[0] == np.dtype('O'):
                continue
            if include_cols != None:
                if col not in include_cols:
                    continue
            # find the intersection of the two
            df_valid = pd.concat([self.df_targets, self.column_data[col]], 
                                  axis=1, join='inner')
            df_col_valid = df_valid.loc[:,[col]]
            # if too many nans, skip the column
            if df_col_valid.isna().mean()[0]>missing_threshold or df_valid.shape[0]==0:
                continue
            # low r-value, skip the column
            xx = df_valid.values
            np.nan_to_num(xx, copy=False)
            try:
                mu_info = sklearn.feature_selection.mutual_info_classif(
                    xx[:,1].reshape(-1, 1), xx[:,0].ravel().astype(np.int))
                #mu_info = np.abs(scipy.stats.pearsonr(xx[:,0], xx[:,1])[0])
                #print(mu_info)
            except:
                continue
            if mu_info < muinfo_threshold:
                continue            
            # do outer join
            self.df_features = pd.concat([self.df_features, df_col_valid], axis=1, join='outer')
            count_thresh = self.df_features.shape[0]*(1-missing_threshold)
            self.df_features.dropna(axis=1, thresh=count_thresh, inplace=True)
            
        # preprocess all features
        self.costs = []
        prep_features = []
        for col in self.df_features.columns:
            prepp_col = preprocess(self.df_features[col], self.column_info[col], None)
            if type(prepp_col) == type(None):
                continue
            prep_features.append(prepp_col)
            prep_len = 1
            if len(prepp_col.shape) == 2: 
                prep_len = prepp_col.shape[1]
            self.costs.extend([self.column_info[col]['cost']] * prep_len)
        self.df_features = pd.concat(prep_features, axis=1)
        self.costs = np.array(self.costs)
        # shuffle them
        inds_perm = copy.deepcopy(self.df_features.index.values)
        np.random.shuffle(inds_perm)
        self.df_features = self.df_features.loc[inds_perm]
        self.df_targets = self.df_targets.loc[inds_perm]
        return (self.df_features, self.df_targets)
            
    def update_column_info(self, col, dfile):
        self.column_info[col] = {}
        # get column values
        vals = self.column_data[col][col]
        vals_unique = len(vals.unique())
        if vals_unique < 20:
            self.column_info[col]['type'] = 'categorical'
        else:
            self.column_info[col]['type'] = 'real'
        # set feature costs
        sheet = dfile.split('/')[-2]
        if sheet == 'Demographics':
            self.column_info[col]['cost'] = 2.0
        elif sheet == 'Dietary':
            self.column_info[col]['cost'] = 4.0
        elif sheet == 'Examination':
            self.column_info[col]['cost'] = 5.0
        elif sheet == 'Questionnaire':
            self.column_info[col]['cost'] = 4.0
        elif sheet == 'Laboratory':
            self.column_info[col]['cost'] = 9.0
        else:
            raise NotImplementedError
        return
    
    def save_supervised(self, filename):
        save_dict = {'df_features':self.df_features, 
                     'df_targets':self.df_targets, 
                     'costs':self.costs}
        with open(filename, 'wb+') as f:
            pickle.dump(save_dict, f)
            
    def load_supervised(self, filename):
        with open(filename, 'rb') as f:
            load_dict = pickle.load(f)
        self.df_features = load_dict['df_features']
        self.df_targets = load_dict['df_targets']
        self.costs = load_dict['costs']
    
    def get_distribution(self, phase, balanced=True):
        features = self.df_features.values
        targets = self.df_targets.values
        # check the phase
        inds_tst = np.arange(1,features.shape[0]*0.15, dtype=np.int)
        inds_val = np.arange(features.shape[0]*0.15, 
                              features.shape[0]*0.30, dtype=np.int)
        inds_trn = np.arange(features.shape[0]*0.30, 
                              features.shape[0]*1.00, dtype=np.int)
        # if the phase is validation
        if phase == 'validation':
            phase_features = features[inds_val,:]
            phase_targets = targets[inds_val]
        # if the phase is  test
        elif phase == 'test':
            phase_features = features[inds_tst,:]
            phase_targets = targets[inds_tst]
        # if the phase is  train
        elif phase == 'train':
            phase_features = features[inds_trn,:]
            phase_targets = targets[inds_trn]
        elif phase == 'all':
            phase_features = features[:,:]
            phase_targets = targets[:]
        else:
            raise NotImplementedError('phase not found.')
        # sampling
        i = 0
        y = 0
        n_cls = np.max(phase_targets) + 1
        while True:
            i += 1
            ind = i % phase_features.shape[0]
            # balance dataset
            if balanced:
                while phase_targets[ind] != y:
                    i += 1
                    ind = i % phase_features.shape[0]
                y += 1
                if y >= n_cls:
                    y = 0
            # yield the sample
            yield (ind, phase_features[ind], phase_targets[ind], self.costs, 1.0)
            
    def get_batch(self, n_size, phase, balanced=True):
        dataset_features = self.df_features.values
        dataset_targets = self.df_targets.values
        # select indices
        n_samples = dataset_features.shape[0]
        n_classes = int(dataset_targets.max() + 1)
        if phase == 'test':
            inds_sel = np.arange(0, int(n_samples*0.15), 1)
        elif phase == 'validation':
            n_samples = dataset_features.shape[0]
            inds_sel = np.arange(int(n_samples*0.15), int(n_samples*0.30), 1)
        elif phase == 'train':
            n_samples = dataset_features.shape[0]
            inds_sel = np.arange(int(n_samples*0.30), n_samples, 1)
        else:
            raise NotImplementedError
        inds_sel = np.random.permutation(inds_sel)
        batch_inds = []
        # if we should balance the data
        if balanced:
            for cl in range(n_classes):
                inds_cl = inds_sel[dataset_targets[inds_sel] == cl]
                batch_inds.extend(inds_cl[:n_size//n_classes])
        else:
            batch_inds = inds_sel[:n_size]
        batch_inds = np.random.permutation(batch_inds)
        return dataset_features[batch_inds], dataset_targets[batch_inds]
    

def preprocess(df_col, info_col, preprocessor=None):
    if df_col.dtypes == np.dtype('O'):
        return None#pd.DataFrame()
    #df_col[pd.isna(df_col)] = df_col.mean()
    if info_col['type'] == 'categorical':
        df_col =  preproc_onehot(df_col)
        #df_col[pd.isna(df_col)] = df_col.mean()
    elif info_col['type'] == 'real':
        df_col =  preproc_real(df_col)
    else:
        raise NotImplementedError
    return df_col
        


def preproc_onehot(df_col, args=None):
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')

def preproc_real(df_col, args=None):
    if args is None:
        args={'cutoff':np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    # statistical normalization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col

def preproc_impute(df_col, args=None):
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    return df_col

def preproc_cut(df_col, bins):
    # limit values to the bins range
    df_col = df_col[df_col >= bins[0]]
    df_col = df_col[df_col <= bins[-1]]
    return pd.cut(df_col.iloc[:,0], bins, labels=False)

def preproc_dropna(df_col, args=None):
    df_col.dropna(axis=0, how='any', inplace=True)
    return df_col

class Dataset():
    """ 
    Dataset manager class
    """
    def  __init__(self, data_path=None):
        """
        Class intitializer.
        """
        # set database path
        if data_path == None:
            self.data_path = './run_data/'
        else:
            self.data_path = data_path
        # feature and target vecotrs
        self.features = None
        self.targets = None
        self.costs = None

    def load_diabetes(self, opts=None):
        columns = [
            # TARGET: Fasting Glucose
            FeatureColumn('Laboratory', 'LBXGLU', 
                              #preproc_dropna, None),   
                              preproc_impute, None),
            # Gender
            FeatureColumn('Demographics', 'RIAGENDR', 
                                 preproc_real, None, cost=2),
            # Age at time of screening
            FeatureColumn('Demographics', 'RIDAGEYR', 
                                 preproc_real, None, cost=2),
            FeatureColumn('Demographics', 'RIDRETH3', 
                                 preproc_onehot, None, cost=2),
            # Race/ethnicity
            FeatureColumn('Demographics', 'RIDRETH1', 
                                 preproc_onehot, None, cost=2),
            # Annual household income
            FeatureColumn('Demographics', 'INDHHINC', 
                                 preproc_real, {'cutoff':11}, cost=4),
            # Education level
            FeatureColumn('Demographics', 'DMDEDUC2', 
                                 preproc_real, {'cutoff':5}, cost=2),
            # Blood pressure
            FeatureColumn('Examination', 'BPXSY1', 
                                 preproc_real, None, cost=5),
            FeatureColumn('Examination', 'BPXDI1', 
                                 preproc_real, None, cost=5),
            FeatureColumn('Examination', 'BPXSY2', 
                                 preproc_real, None, cost=5),
            FeatureColumn('Examination', 'BPXDI2', 
                                 preproc_real, None, cost=5),
            FeatureColumn('Examination', 'BPXSY3', 
                                 preproc_real, None, cost=5),
            FeatureColumn('Examination', 'BPXDI3', 
                                 preproc_real, None, cost=5),
            FeatureColumn('Examination', 'BPXSY4', 
                                 preproc_real, None, cost=5),
            FeatureColumn('Examination', 'BPXDI4', 
                                 preproc_real, None, cost=5),
            # BMI
            FeatureColumn('Examination', 'BMXBMI', 
                                 preproc_real, None, cost=5),
            # Waist
            FeatureColumn('Examination', 'BMXWAIST', 
                                 preproc_real, None, cost=5),
            # Height
            FeatureColumn('Examination', 'BMXHT', 
                                 preproc_real, None, cost=5),
            # Upper Leg Length
            FeatureColumn('Examination', 'BMXLEG', 
                                 preproc_real, None, cost=5),
            # Weight
            FeatureColumn('Examination', 'BMXWT', 
                                 preproc_real, None, cost=5),
            # Total Cholesterol
            FeatureColumn('Laboratory', 'LBXTC', 
                                 preproc_real, None, cost=9),
            # Triglyceride
            FeatureColumn('Laboratory', 'LBXTR', 
                                 preproc_real, None, cost=9),
            # fibrinogen
            FeatureColumn('Laboratory', 'LBXFB', 
                                 preproc_real, None, cost=9),
            # LDL-cholesterol
            FeatureColumn('Laboratory', 'LBDLDL', 
                                 preproc_real, None, cost=9),
            # Alcohol consumption
            FeatureColumn('Questionnaire', 'ALQ101', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'ALQ120Q', 
                                 preproc_real, {'cutoff':365}, cost=4),
            # Vigorous work activity
            FeatureColumn('Questionnaire', 'PAQ605', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'PAQ620', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'PAQ180', 
                                 preproc_real, {'cutoff':4}, cost=4),
            # Sleep
            FeatureColumn('Questionnaire', 'SLD010H', 
                                 preproc_real, {'cutoff':12}, cost=4),
            # Smoking
            FeatureColumn('Questionnaire', 'SMQ020', 
                                 preproc_onehot, None, cost=4),
            FeatureColumn('Questionnaire', 'SMD030', 
                                 preproc_real, {'cutoff':72}, cost=4),
            # Blood relatives have diabetes
            FeatureColumn('Questionnaire', 'MCQ250A', 
                                 preproc_real, {'cutoff':2}, cost=4),
            # Blood pressure history
            FeatureColumn('Questionnaire', 'BPQ020', 
                                 preproc_real, {'cutoff':2}, cost=4),
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        # extract feature and target
        features = df.loc[:, df.columns != 'LBXGLU'].values
        targets_LBXGLU = df['LBXGLU'].values
        targets = np.zeros(targets_LBXGLU.shape[0])
        targets[targets_LBXGLU <= 100] = 0
        targets[np.logical_and(targets_LBXGLU<125,targets_LBXGLU>100)] = 1
        targets[targets_LBXGLU >= 125] = 2
        # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
            [item for sublist in self.costs for item in sublist])
        
    def load_hypertension(self, opts=None):
        columns = [
            # TARGET: systolic BP average
            FeatureColumn('Examination', 'BPXSAR',
                             preproc_dropna, None, cost=5),
            # Gender
            FeatureColumn('Demographics', 'RIAGENDR', 
                                 preproc_real, None, cost=2),
            # Age at time of screening
            FeatureColumn('Demographics', 'RIDAGEYR', 
                                 preproc_real, None, cost=2),
            # Race/ethnicity
            FeatureColumn('Demographics', 'RIDRETH1', 
                                 preproc_onehot, None, cost=2),
            # Annual household income
            FeatureColumn('Demographics', 'INDHHINC', 
                                 preproc_real, {'cutoff':11}, cost=4),
            # Education level
            FeatureColumn('Demographics', 'DMDEDUC2', 
                                 preproc_real, {'cutoff':5}, cost=2),
            # Sodium eaten day before
            FeatureColumn('Dietary', 'DR2TSODI', 
                             preproc_real, {'cutoff':20683}, cost=4),
            # BMI
            FeatureColumn('Examination', 'BMXBMI', 
                                 preproc_real, None, cost=5),
            # Waist
            FeatureColumn('Examination', 'BMXWAIST', 
                                 preproc_real, None, cost=5),
            # Height
            FeatureColumn('Examination', 'BMXHT', 
                                 preproc_real, None, cost=5),
            # Upper Leg Length
            FeatureColumn('Examination', 'BMXLEG', 
                                 preproc_real, None, cost=5),
            # Weight
            FeatureColumn('Examination', 'BMXWT', 
                                 preproc_real, None, cost=5),
            # Total Cholesterol
            FeatureColumn('Laboratory', 'LBXTC', 
                                 preproc_real, None, cost=9),
            # Triglyceride
            FeatureColumn('Laboratory', 'LBXTR', 
                                 preproc_real, None, cost=9),
            # fibrinogen
            FeatureColumn('Laboratory', 'LBXFB', 
                                 preproc_real, None, cost=9),
            # LDL-cholesterol
            FeatureColumn('Laboratory', 'LBDLDL', 
                                 preproc_real, None, cost=9),
            # Alcohol consumption
            FeatureColumn('Questionnaire', 'ALQ101', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'ALQ120Q', 
                                 preproc_real, {'cutoff':365}, cost=4),
            # Vigorous work activity
            FeatureColumn('Questionnaire', 'PAQ605', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'PAQ620', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'PAQ180', 
                                 preproc_real, {'cutoff':4}, cost=4),
            # Sleep
            FeatureColumn('Questionnaire', 'SLD010H', 
                                 preproc_real, {'cutoff':12}, cost=4),
            # Smoking
            FeatureColumn('Questionnaire', 'SMQ020', 
                                 preproc_onehot, None, cost=4),
            FeatureColumn('Questionnaire', 'SMD030', 
                                 preproc_real, {'cutoff':72}, cost=4),
            # Blood relatives have hypertension/stroke
            FeatureColumn('Questionnaire', 'MCQ250F', 
                                 preproc_real, {'cutoff':2}, cost=4),
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        # extract feature and target
        # below 90/60 is hypotension, in between is normal, above 120/80 is prehypertension,
        # above 140/90 is hypertension 
        fe_cols = df.drop(['BPXSAR'], axis=1)
        features = fe_cols.values
        target = df['BPXSAR'].values
        # remove nan labeled samples
        inds_valid = ~ np.isnan(target)
        features = features[inds_valid]
        target = target[inds_valid]

        # Put each person in the corresponding bin
        targets = np.zeros(target.shape[0])
        targets[target < 140] = 0 # Rest (hypotsn, normal, prehyptsn)
        targets[target >= 140] = 1 # hypertension

       # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
            [item for sublist in self.costs for item in sublist])
    
    def load_arthritis(self, opts=None):
        columns = [
            # TARGET: systolic BP average
            FeatureColumn('Questionnaire', 'MCQ160A', 
                                    None, None, cost=4),
            # Gender
            FeatureColumn('Demographics', 'RIAGENDR', 
                                 preproc_real, None, cost=2),
            # Age at time of screening
            FeatureColumn('Demographics', 'RIDAGEYR', 
                                 preproc_real, None, cost=2),
            FeatureColumn('Demographics', 'RIDRETH3', 
                                 preproc_onehot, None, cost=2),
            # Race/ethnicity
            FeatureColumn('Demographics', 'RIDRETH1', 
                                 preproc_onehot, None, cost=2),
            # Annual household income
            FeatureColumn('Demographics', 'INDHHINC', 
                                 preproc_real, {'cutoff':11}, cost=4),
            # Education level
            FeatureColumn('Demographics', 'DMDEDUC2', 
                                 preproc_real, {'cutoff':5}, cost=2),
            # BMI
            FeatureColumn('Examination', 'BMXBMI', 
                                 preproc_real, None, cost=5),
            # Waist
            FeatureColumn('Examination', 'BMXWAIST', 
                                 preproc_real, None, cost=5),
            # Height
            FeatureColumn('Examination', 'BMXHT', 
                                 preproc_real, None, cost=5),
            # Upper Leg Length
            FeatureColumn('Examination', 'BMXLEG', 
                                 preproc_real, None, cost=5),
            # Weight
            FeatureColumn('Examination', 'BMXWT', 
                                 preproc_real, None, cost=5),
            # Total Cholesterol
            FeatureColumn('Laboratory', 'LBXTC', 
                                 preproc_real, None, cost=9),
            # Alcohol consumption
            FeatureColumn('Questionnaire', 'ALQ101', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'ALQ120Q', 
                                 preproc_real, {'cutoff':365}, cost=4),
            # Vigorous work activity
            FeatureColumn('Questionnaire', 'PAQ605', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'PAQ620', 
                                 preproc_real, {'cutoff':2}, cost=4),
            FeatureColumn('Questionnaire', 'PAQ180', 
                                 preproc_real, {'cutoff':4}, cost=4),
            FeatureColumn('Questionnaire', 'PAD615', 
                                 preproc_real, {'cutoff':780}, cost=4),
            # Doctor told overweight (risk factor)
            FeatureColumn('Questionnaire', 'MCQ160J', 
                                 preproc_onehot, {'cutoff':2}, cost=4),
            # Sleep
            FeatureColumn('Questionnaire', 'SLD010H', 
                                 preproc_real, {'cutoff':12}, cost=4),
            # Smoking
            FeatureColumn('Questionnaire', 'SMQ020', 
                                 preproc_onehot, None, cost=4),
            FeatureColumn('Questionnaire', 'SMD030', 
                                 preproc_real, {'cutoff':72}, cost=4),
            # Blood relatives with arthritis
            FeatureColumn('Questionnaire', 'MCQ250D',
                                 preproc_onehot, {'cutoff':2}, cost=4),
            # joint pain/aching/stiffness in past year
            FeatureColumn('Questionnaire', 'MPQ010',
                                 preproc_onehot, {'cutoff':2}, cost=4),
            # symptoms began only because of injury
            FeatureColumn('Questionnaire', 'MPQ030',
                                 preproc_onehot, {'cutoff':2}, cost=4),
            # how long experiencing pain
            FeatureColumn('Questionnaire', 'MPQ110',
                                 preproc_real, {'cutoff':4}, cost=4),
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        fe_cols = df.drop(['MCQ160A'], axis=1)
        features = fe_cols.values
        target = df['MCQ160A'].values
        # remove nan labeled samples
        inds_valid = ~ np.isnan(target)
        features = features[inds_valid]
        target = target[inds_valid]

        # Put each person in the corresponding bin
        targets = np.zeros(target.shape[0])
        targets[target == 1] = 0 # yes arthritis
        targets[target == 2] = 1 # no arthritis

       # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
            [item for sublist in self.costs for item in sublist])



    def get_dataset(self, phase):
        # check the phase
        inds_tst = np.arange(1,self.features.shape[0]*0.15, dtype=np.int)
        inds_val = np.arange(self.features.shape[0]*0.15, 
                              self.features.shape[0]*0.30, dtype=np.int)
        inds_trn = np.arange(self.features.shape[0]*0.30, 
                              self.features.shape[0]*1.00, dtype=np.int)
        # if the phase is validation
        if phase == 'validation':
            phase_features = self.features[inds_val,:]
            phase_targets = self.targets[inds_val]
        # if the phase is  test
        elif phase == 'test':
            phase_features = self.features[inds_tst,:]
            phase_targets = self.targets[inds_tst]
        # if the phase is  train
        elif phase == 'train':
            phase_features = self.features[inds_trn,:]
            phase_targets = self.targets[inds_trn]
        elif phase == 'all':
            phase_features = self.features[:,:]
            phase_targets = self.targets[:]
        else:
            raise NotImplementedError('phase not found.')
        # sampling
        i = 0
        y = 0
        n_cls = np.max(phase_targets) + 1
        while True:
            i += 1
            ind = i % phase_features.shape[0]
            # balance dataset
            while phase_targets[ind] != y:
                i += 1
                ind = i % phase_features.shape[0]
            y += 1
            if y >= n_cls:
                y = 0
            # yield the sample
            yield (ind, phase_features[ind], phase_targets[ind], self.costs, 1.0)
