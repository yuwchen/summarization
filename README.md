# Summarization

Code for Doctor-patient conversation summarization. 

## Dataset 
- [ACI-BENCH](https://github.com/wyim/aci-bench)  
- [MTS-Dialog](https://github.com/abachaa/MTS-Dialog)  

## Code

### Run OpanAI-model
(1) Set up API-key. 
```
sudo vim ~/.bash_profile
# Add 'export OPENAI_API_KEY='MY_OPENAPI_KEY' in the end of the ./bash_profile file
source ~/.bash_profile
# check whether the key is added successfully by 'echo $OPENAI_API_KEY'
```
(2) Run openai models and save the results
```
python openai_mts_general.py
python openai_mts_soap.py
python openai_aci_general.py
```

### Run Fine-tuned-LM-based model. 
```
python run_led.py           #run wanglab LED model on MTS-dialog data
python run_flant5.py        #run wanglab FLAN-T5 model on ACI-BENCH data
python run_flant5_seg.py    #run wanglab FLAN-T5 model on segmented ACI-BENCH data then merge the result using Falconsai/medical_summarization
```

### Run evaluation scores
evaluation_{model_name}_{dataset}  

sources: https://github.com/abachaa/MEDIQA-Chat-2023/blob/main/scripts/evaluate_summarization.py
```
python evalution_gpt_aci.py          
python evalution_gpt_soap_aci.py     
python evalution_led_aci.py
python evalution_led_mts.py
python evalution_t5_aci.py
python evalution_t5seg_aci.py
python evalution_openai_mts.py
python evalution_openai_soap_mts.py 

```
```
python count_num_token.py #cound number of token in the dir
```
### Summarize evaluation scores
```
python cross_domain_aci.py
python cross_domain_mts.py
```

### Data & Results
- [data](https://drive.google.com/drive/folders/1myB-eChZwRmXPg3hMP0_gaHQLC5OBimh?usp=drive_link)      #data copied from the dataset
- [Results](https://drive.google.com/drive/folders/1oEnUc2vNg6UNnHxV1gfIsPI5kvl8m3e2?usp=drive_link)   #generated notes
- [Evaluation_scores](https://drive.google.com/drive/folders/1VTri9HjcTR0w2O3MdpLUT9c0_7xO6Ljx?usp=sharing) #evaluation scores


------
### Fine-tuned FLAN-T5
```
python reformat_data.py          #reformat the data into hugging face dataset format
python finetune_t5large_mts.py   #fine-tuned flan-t5 model
```



