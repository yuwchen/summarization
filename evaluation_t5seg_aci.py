import os
import re
import pandas as pd
import evaluate
import json
from utils import *


result_root = './Results/task-a-flan-t5-large-run-2-medical_summarization_aci-taskB-test1'
outputdir = './Evaluation_scores/aci_t5_sep_med/'
gt_df = pd.read_csv('./data/aci-bench/clinicalnlp_taskB_test1.csv')
soap_header = ['Full', 'subjective', 'objective', 'assessment_and_plan']

gt_note = {}
for idx, row in gt_df.iterrows():

    references = []
    predictions = []

    encounter_id = row['encounter_id']
    corpus_dataset = row['dataset']
    dialog = row['dialogue']
    gt_note = row['note']
    gt_note = process_gt_note(gt_note)

    sample_idx = str(corpus_dataset)+'-'+str(encounter_id)+'.txt'
    pred_note = open(os.path.join(result_root, sample_idx),'r').read()

    gt_note = process_text(gt_note)
    gt_soap = {'subjective':'', 'objective':'', 'assessment_and_plan':'', 'Full':''}
    for key, value in gt_note.items():
        soap = TASK_B_SECTION_HEADER_MAP[key]
        gt_soap[soap] = gt_soap[soap] + value
        gt_soap['Full'] = gt_soap['Full']+value
   
    for category in soap_header:
        references.append(gt_soap[category])
        predictions.append(pred_note)

    all_scores = {}
    for name, (scorer, kwargs, keys, save_keys) in scorers.items():
        scores = scorer.compute(references=references, predictions=predictions, **kwargs)
        for score_key, save_key in zip(keys, save_keys):
            all_scores[save_key] = scores[score_key]

    all_scores['pred_note'] = predictions
    all_scores['gt_note'] = references
    
    outputname = os.path.join(outputdir, sample_idx.replace('.txt','.json') )
    with open(outputname, 'w') as f:
        json.dump(all_scores, f, indent=2)
