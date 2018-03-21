def prob_arpa(probabilities, arpa_filename, indices):
    models=arpa.loadf(arpa_filename,'r')
    lm=models[0]
    frontier={}
    probabilities=normalize_probab_sentence(probabilities, indices)
         # initialize frontier
    for w, p in probabilities[indices[0]].items():
        try:
            s=' '.join(['<s>', w])
            lm1=lm.p(s) 
            frontier[(1, w)]=((p*lm1)**0.5 , w)        #(1/(lm1*p)**0.5, w)
        except:
            pass
    explored=set()
    while True:
            # next node (part of sentence to add to) to explore
        node=frontier[sorted(frontier.keys())[0]] # format: (probab, text)
        node_w=node[1].split()
        if len(node_w)<4: nw=node[1]
        else: nw=' '.join([node_w[-3], node_w[-2] ,node_w[-1]])
        #print(node, nw)
        del frontier[(len(node_w), nw)]

           # goal check: the goal is not to have any nodes in frontier that have less words than the length of sentence
        if len(indices)==sorted(frontier.keys())[0]:
            break
            return frontier, explored
        
        explored.add(node)

        for word , p in probabilities[indices[len(node_w)]].items():
            child_w=" ".join([node[1], word])
            text=child_w.split()
            try:
                child_lm=lm.p(child_w)
                
            except: child_lm=0
            child_p=(node[0]*p*child_lm)**(1/3)
            child_w_l=len(text)
           
            if child_w_l<4: node_name=child_w
            else: node_name=(child_w_l, " ".join([text[0], text[-3], text[-2], text[-1]]))
            
            if child_w_l==len(indices): 
                try: 
                    child_s=lm.p(child_w)
                except: child_ass=1e-10
             
            child_p=(child_s*child_p)**0.5    
            if child_w not in explored and node_name not in frontier:
                frontier[node_name]= (child_p, child_w)

            else:
                if node_name in frontier:
                    pr_f=frontier[node_name][0]
                    if child_p<pr_f:
                        frontier[node_name]=(child_p, child_w)

    return frontier, explored