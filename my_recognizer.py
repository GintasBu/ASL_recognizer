import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    #D={}
    for ind, data in test_set._hmm_data.items():
        X, lengths=data[0], data[1]
        D={}
        #print(lengths)
        for word, model in models.items():
            #(X, lengths)=v
            try:
                logL=model.score(X, lengths)
                D[word]=logL
                #if self.verbose:
                #    print("score created for {} model with {} test_set index".format(word, ind))
            except:
                pass #if self.verbose:
                #print("failure to get score for {} with {} test_set index".format(word, ind))
        highest_prob=sorted(D.values(), reverse=True)[0:2]
        for k,v in D.items():
            if v==highest_prob[0]:
                guesses.append(k)
                break
        if highest_prob[0]==highest_prob[1]: # for an unlikely case when two words yeld the same highest logL
            raise ValueError('two words ahve highest probability')
        probabilities.append(D)
    #print(probabilities)
    return probabilities, guesses
    #raise NotImplementedError
