from openai import OpenAI
import os
import json
import pandas as pd
import pickle


def creat_dir(dir_path):
  if not os.path.exists(dir_path):
      os.makedirs(dir_path)

df = pd.read_csv('./data/aci-bench/clinicalnlp_taskB_test1.csv')
creat_dir('./Results/openai_aci_general')


for index, row in df.iterrows():
  encounter_id = row['encounter_id']
  corpus_dataset = row['dataset']
  dialog = row['dialogue']
  gt_note = row['note']
  print(encounter_id)
  print(dialog)

  client = OpenAI()

  source_prefix = "Summarize the following patient-doctor dialogue. Include all medically relevant information. \n Dialogue:\n------------\n"
  text  = source_prefix+dialog
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "You are a helpful assistant"},
      {"role": "user", "content": text}
    ]
  )

  print(response)
  outputname =  os.path.join('./Results/openai_aci_general',str(corpus_dataset)+'-'+str(encounter_id)+'.json')
  results = {
      'encounter_id':encounter_id,
      'corpus_dataset':corpus_dataset,
      'dialogue':dialog,
      "results": response.choices[0].message.content,
      "model":response.model,
      'completion_tokens':response.usage.completion_tokens,
      'prompt_tokens':response.usage.prompt_tokens,
      'total_tokens':response.usage.total_tokens

  }

  with open(outputname, 'w') as f:
      json.dump(results, f, indent=2)
