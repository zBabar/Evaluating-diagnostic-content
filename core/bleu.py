import cPickle as pickle
import os
import sys
sys.path.append('../coco-caption')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import pandas as pd

def save_all(final_scores_ind,ref,cand,path):
    final_scores_ind['cand']=cand
    final_scores_ind['ref']=ref

    final_scores_ind=pd.DataFrame(final_scores_ind)

    final_scores_ind.to_csv(path+'/all.csv')

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(),"CIDEr")
    ]
    final_scores = {}
    final_scores_ind={}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

        if type(scores)==list:
            if method=='METEOR':
                final_scores_ind[method] = scores
            else:
                for m,s in zip(method,scores):
                    final_scores_ind[m] = s
        else:
            final_scores_ind[method] = scores
    return final_scores,final_scores_ind
    

def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" %(split, split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.pkl" %(split, split))
    ref_lst=[]
    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)

    for i in ref:
        ref_lst.append(ref[i])
    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]
    
    # compute bleu score
    final_scores, final_scores_ind = score(ref, hypo)
    save_all(final_scores_ind,ref_lst,cand,data_path)


    # print out scores
    print 'Bleu_1:\t',final_scores['Bleu_1']  
    print 'Bleu_2:\t',final_scores['Bleu_2']  
    print 'Bleu_3:\t',final_scores['Bleu_3']  
    print 'Bleu_4:\t',final_scores['Bleu_4']
    # print 'Bleu_5:\t', final_scores['Bleu_5']
    # print 'Bleu_6:\t', final_scores['Bleu_6']
    # print 'Bleu_7:\t', final_scores['Bleu_7']
    print 'METEOR:\t',final_scores['METEOR']  
    print 'ROUGE_L:',final_scores['ROUGE_L']  
    print 'CIDEr:\t',final_scores['CIDEr']
    
    if get_scores:
        return final_scores
    
   
    
    
    
    
    
    
    
    
    
    


