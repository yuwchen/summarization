import pandas as pd
import numpy as np
from datasets import Dataset


df  = pd.read_csv("./data/mts-dialog/MTS-Dialog-TrainingSet.csv")

category_mapping = {
        'GENHX':'SUBJECTIVE',
        'FAM/SOCHX':'SUBJECTIVE',
        'PASTMEDICALHX':'SUBJECTIVE',
        'CC':'SUBJECTIVE',
        'PASTSURGICAL':'SUBJECTIVE',
        'ALLERGY':'SUBJECTIVE',
        'ROS':'SUBJECTIVE',
        'MEDICATIONS':'SUBJECTIVE',
        'ASSESSMENT':'ASSESSMENT',
        'EXAM':'OBJECTIVE',
        'DIAGNOSIS':'ASSESSMENT',
        'DISPOSITION':'PLAN',
        'PLAN':'PLAN',
        'EDCOURSE':'PLAN',
        'IMMUNIZATIONS':'SUBJECTIVE',
        'IMAGING':'OBJECTIVE',
        'GYNHX':'SUBJECTIVE',
        'PROCEDURES':'SUBJECTIVE',
        'OTHER_HISTORY':'SUBJECTIVE',
        'LABS':'OBJECTIVE'
}


id_list = []
dialogue_list = []
summary_list = []

for index, row in df.iterrows():
    id = row['ID']
    section_header = row['section_header']
    dialogue = row['dialogue']
    section_text = row['section_text']
    id_list.append(id)
    dialogue_list.append(dialogue)
    summary_list.append(section_text)

dataset = Dataset.from_dict({'id':id_list, 'dialogue':dialogue_list, 'summary':summary_list})

dataset.save_to_disk("./data/MTS-Dialog-TrainingSet_dataset")

print(dataset)
print(dataset.features)
print(len(dataset))
example = dataset[0]
print(example)
