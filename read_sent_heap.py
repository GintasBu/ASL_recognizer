# will try UCS using heaps
def read_sentence_heap(logL_lm, probabilities3, starters_dict, indices):
    # indices are index's of words from the test_set.words that form a sentence
    import heapq
    import copy
    starters=[]
    answer=[]
    prob_heap=hmmprob_to_heaps(probabilities3) # word recognition pseudo-perplexity (1/p) from HMM
    t1=time.time()
    Cost=float('inf')
    while True: #for i in list(range(len(indices))):  # i is a line # in probabilities as well as an index of the word in test_set.wordlist
     
        if not len(starters): # for first word in a sentence
            #last_trigrams=[]
            for w1 in list(probabilities3[i].keys()):
                #print(w1)
                bigram=' '.join(['<s>',w1])
                logprob=probabilities3[indices[i]][w1] 
                logprob_lm=starters_dict[bigram] 
                #print(logprob, logprob_lm, w1)
                if not np.isnan(logprob) and not np.isnan(logprob_lm): 
                    pp=1/((logprob*logprob_lm)**(0.5)) # pseudo-perplexity of word in the begining of the sentence
               
                    heapq.heappush(starters, (pp, bigram)) #trigrams.append((trigram, prob, prob_lm))
            nodes=copy.copy(starters)
            print(starters)
        else:
            probab=copy.copy(prob_heap)
            #print(nodes)
            if len(probab[len(words)-1])<1:
                break
            new_node_a=heapq.heappop(nodes)
            if goal_test(new_node_a, indices): 
                break
                return new_node_a[1][1:-1]
            text=new_node_a[-1]
            words=text.split()
            bigram=' '.join([words[-2], words[-1]])
            if len(words)==len(indices)+2: 
                heapq.heappush(answer, new_node_a)
            
            
            if len(words)<len(indices)+1: 
                word=heapq.heappop(probab[len(words)-1]) # -1 to adjust for an added <s> word to the begining of the sentence
                logprob=1/word[0]
                    #print(word[0])
                trigram=' '.join([bigram, word[1]])
            else:
                trigram=' '.join([bigram, '</s>')
                
            if trigram in logL_lm:
                logprob_lm=logL_lm[trigram]
            else: logprob_lm=np.nan
            pp=(1/(1/new_node_a[0]*logprob*logprob_lm))**(1/3)
                
            if not np.isnan(pp): 
                heapq.heappush(nodes, (pp, ' '.join([text, word[1]])))
                print(pp, ' '.join([text, word[1]]), logprob, logprob_lm, new_node_a)


def goal_test(node, indices):
    return len(indices)==len(node[1])-2