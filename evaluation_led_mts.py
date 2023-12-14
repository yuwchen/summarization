import os
import re
import pandas as pd
import evaluate
import json
from utils import *

result_root = './Results/task-b-led-large-16384-pubmed-run-3-4096-1024_mts-testset1-chat/'
outputdir = './Evaluation_scores/mts_Led_4096-1024/'
gt_df = pd.read_csv('./data/mts-dialog/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')

gt_note = {}

soap_header = ['Full', 'subjective', 'objective', 'assessment_and_plan']


for idx, row in gt_df.iterrows():

    references = []
    predictions = []

    id = row['ID']
    dialog = row['dialogue']
    gt_note = row['section_text']
    gt_header = row['section_header']

    pred_note = open(os.path.join(result_root, str(id)+'.txt'),'r').read()
    pred_note = process_text(pred_note)


    pred_soap = {'Full':'', 'subjective':'', 'objective':'', 'assessment_and_plan':''}
    for key, value in pred_note.items():
        soap = TASK_B_SECTION_HEADER_MAP[key]
        pred_soap[soap] = pred_soap[soap] + value
        pred_soap['Full'] = pred_soap['Full'] + value
    
    for category in soap_header:
        references.append(gt_note)
        predictions.append(pred_soap[category])

    
    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    all_scores['pred_note'] = predictions
    all_scores['gt_note'] = references
    all_scores['section_header'] = gt_header
    
    outputname = os.path.join(outputdir, str(id)+'.json') 
    with open(outputname, 'w') as f:
        json.dump(all_scores, f, indent=2)

    
