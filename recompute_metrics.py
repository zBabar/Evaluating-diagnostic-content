#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from json import load
from math import ceil
import pandas as pd
import nltk
import pickle as pkl
import os, csv

from core.pycocoevalcap.meteor.meteor import Meteor
from core.pycocoevalcap.rouge.rouge import Rouge
from core.pycocoevalcap.cider.cider import Cider
from core.pycocoevalcap.bleu.bleu import Bleu


# In[ ]:


def compute_findings_metrics(test_set, predictions,testIDs):
    gts = {} # ground truth
    res = {}
    
    # Compile ground_truth and candidate predictions dictionaries
    for i, id in enumerate(testIDs):
        words = nltk.word_tokenize(test_set[i])
        test_set[i]=test_set[i].replace('[','').replace(']','').replace('.','').replace(',','').replace("'",'').split()#replace('<','').replace('>','').replace('S','').replace('/','').split()
        
        predictions[i]=predictions[i].replace('[','').replace('.','').replace(',','').replace(']','').replace("'",'').split()#replace('<','').replace('>','').replace('S','').replace('/','').split()
        
        #print(words)
        test=[]
        test.append(words)
        
        #print(predictions[i])
        #print('ground:',[' '.join(test_set[i])],'cand:',[' '.join(predictions[i])])
        gts[id] = [' '.join(test_set[i])]
        res[id] = [' '.join(predictions[i])]  
        
        
        
    #print(gts.keys())
    #print(res.keys())
    scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(),"METEOR"),
    (Rouge(), "ROUGE_L"),
    ]    

    all_scores = {}
    peer_by_peer_scores = []
    for scorer, method in scorers:
        #print(scorer)
        score, scores = scorer.compute_score(gts=gts, res=res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                sc = "{0:.3f}".format(sc)
                print("{}: {}".format(m, sc))
                all_scores[m] = sc
                peer_by_peer_scores.append(scs)  
        else:
            score = "{0:.3f}".format(score)
            print("{}: {}".format(method, score))
            all_scores[method] = score
            peer_by_peer_scores.append(scores) 
    # Save the peer-by-peer scores, candidate predictions and ground truth sentences to CSV
    #save_to_csv(peer_by_peer_scores, [res[id] for id in testIDs], [gts[id] for id in testIDs], filename)    
    return all_scores


# In[ ]:


Path='/media/zaheer/Data/Image_Text_Datasets/IU_Xray/latest/One_Image/New/sent/Sample1/'
Path1='/home/zaheer/pythonCode/DCM/Two_Images/true_list_f1.csv'
split='test'

#data={}
#generated_data=
with open(os.path.join(Path, '%s.file.names.pkl' %split), 'rb') as f:
        filenames = pkl.load(f)

#data=pd.read_csv(Path+'all.csv')

data=pd.read_csv(Path1)



#score=compute_findings_metrics(data['Ref'],data['Cand'],list(filenames))
score=compute_findings_metrics(data['ref_sent'],data['cand_sent'],list(data['report_id_x']))

print()