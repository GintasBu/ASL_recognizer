import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # initiate initial values from model and logL at n=min_n_components
        model=None #GaussianHMM(n_components=self.min_n_components, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
        #logL0=model.score(self.X, self.lengths)
        #BICs=np.empty( self.max_n_components-self.min_n_components+1, dtype=float)
        n_features=self.X.shape[1] 
        #N=len(self.sequences) 
        N=self.X.shape[0]
        #print(self.X.shape[1], len(self.X[0]))
        #n=self.min_n_components
        #p=n*n+2*n*n_features-1
        BIC0=float('inf') #-2*logL0+p*math.log(N)
        #BICs[0]=BIC0
        #print(BIC0, n_features, N, n, p)
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))
                logL= hmm_model.score(self.X, self.lengths) 
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, num_states))
                logL=np.nan
            p=n*n+2*n*n_features-1
            BIC=-2*logL+p*math.log(N)
            #BICs[n-self.min_n_components]=BIC
            if BIC<BIC0:
                model=hmm_model
                BIC0=BIC
        #print(BICs, BIC0)
        return model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        DICn=float('-inf')
        modeln=None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model=GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                if self.verbose:
                    print("model created for {} with {} states".format(self.this_word, num_states))
                evid=model.score(self.X, self.lengths)
            except:
                if self.verbose:
                    print("failure on {} with {} states, anti_evid stage".format(self.this_word, num_states))
                continue # if model fails, restart with different n. No need to calculate scores for other words
            anti_evid=0
            M_1=0
            for word in self.words.keys():
                if self.this_word !=word:
                    X_anti, lengths_anti=self.hwords[word]
                    anti_evid+=model.score(X_anti, lengths_anti)
                    M_1+=1
            DIC=evid-anti_evid/M_1
            if DICn<DIC:
                modeln=model
                DICn=DIC
        return modeln


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if len(self.sequences)>2: # if only one data point available to train (like word WRITE) do not do CV and return n=3
            split_method=KFold(n_splits=min(len(self.sequences), 3))  # to account for situations when less than 3 sequences available. If only 2 available, 2-fold split
            logLs=np.empty( [min(len(self.sequences), 3), self.max_n_components-self.min_n_components+1], dtype=float)
            i=0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                combined_train, c_train_lengths=combine_sequences(cv_train_idx, self.sequences)
                combined_test, c_test_lengths=combine_sequences(cv_test_idx, self.sequences)
                for n in range(self.min_n_components, self.max_n_components+1):
                    try:
                        hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(combined_train, c_train_lengths)
                        if self.verbose:
                            print("model created for {} with {} states".format(self.this_word, num_states))
                        logLs[i, n-self.min_n_components]= hmm_model.score(combined_test, c_test_lengths) 
                    except:
                        if self.verbose:
                            print("failure on {} with {} states".format(self.this_word, num_states))
                        logLs[i, n-self.min_n_components]=np.nan
                i+=1
        #print(logLs)#logLs_mean=np.mean(logLs,axis=0)
            nn=np.nanargmax(np.nanmean(logLs, axis=0))+self.min_n_components  # takes mean values of different splits. 
            return self.base_model(nn)
        else: return self.base_model(3) # when 
		# TODO implement model selection using CV
        #raise NotImplementedError
