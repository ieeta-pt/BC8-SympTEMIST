import click

from utils import load_model, load_model_and_tokenizer, load_model_local
from data import load_train_test_split, BIOTagger, SelectModelInputs, EvaluationDataCollator
from transformers import AutoConfig, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForMaskedLM,AutoModel
import transformers
from decoder import decoder
from torch.utils.data import DataLoader
import torch
from ensemble import ensemble_entity_level, ensemble_span_level
import pandas as pd
from collections import defaultdict 
from tqdm import tqdm

import os
import json


class InferenceDataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                 testset_folder,
                 tokenizer,
                 context_size=64):
        super().__init__()
        
        self.context_size = context_size -1 # cls + sep
        self.center_tokens = tokenizer.model_max_length - 2*context_size
        self.dataset = []

        for test_file in os.listdir(testset_folder):
            if test_file[0] != ".":
                
                with open(os.path.join(testset_folder, test_file)) as f:
                    
                    document = ''.join([line for line in f]).strip()
                    
                    encoding = tokenizer(document, add_special_tokens=False)[0]
                    tokens = encoding.ids
                    offsets = encoding.offsets
                    
                    # add pad tokens to the beggining
                    attention_mask = [0] * self.context_size + [1] * len(tokens)
                    tokens = [tokenizer.pad_token_id] * self.context_size + tokens
                    offsets = [None] * self.context_size + offsets
                    
                    
                    #assert len(tokens)==len(offsets)
                    
                    for j,i in enumerate(range(self.context_size,len(tokens),self.center_tokens)):
                        
                        left_context_tokens = tokens[i-self.context_size:i]
                        central_tokens = tokens[i:i+self.center_tokens]
                        right_context_tokens = tokens[i+self.center_tokens:i+self.center_tokens+self.context_size]
                        
                        left_context_offsets = offsets[i-self.context_size:i]
                        central_offsets = offsets[i:i+self.center_tokens]
                        right_context_offsets = offsets[i+self.center_tokens:i+self.center_tokens+self.context_size]
                        
                        left_context_attention_mask = attention_mask[i-self.context_size:i]
                        central_attention_mask = attention_mask[i:i+self.center_tokens]
                        right_context_attention_mask = attention_mask[i+self.center_tokens:i+self.center_tokens+self.context_size]
                        
                        sample_tokens = [tokenizer.cls_token_id] + left_context_tokens + central_tokens + right_context_tokens + [tokenizer.sep_token_id]
                        sample_offsets = [None] + left_context_offsets + central_offsets + right_context_offsets + [None]
                        sample_attention_mask = [1] + left_context_attention_mask + central_attention_mask + right_context_attention_mask + [1]
                        
                        assert len(sample_tokens)<=tokenizer.model_max_length and len(sample_offsets)<=tokenizer.model_max_length
                        
                        if j==0:
                            low_offset, high_offset = sample_offsets[self.context_size+1][0], sample_offsets[-2][1]
                        else:
                            low_offset, high_offset = sample_offsets[1][0], sample_offsets[-2][1]
    
                        
                        
                        sample = {
                            "text": document,
                            "doc_id": test_file,
                            "sequence_id": j, 
                            "input_ids": sample_tokens,
                            "attention_mask": sample_attention_mask,
                            "offsets": sample_offsets,
                            "view_offset": (low_offset, high_offset),
                        }
                        
                        assert len(sample["input_ids"])<=tokenizer.model_max_length
                        assert len(sample["offsets"])<=tokenizer.model_max_length
                        
                        self.dataset.append(sample)

        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def decoder_from_samples(prediction_batch, context_size):
    
    documents = {}
    padding = context_size

    # reconsturct the document in the correct order
    for i in range(len(prediction_batch)):
        doc_id = prediction_batch[i]['doc_id'].split('/')[-1]
        if doc_id not in documents.keys():
            documents[doc_id] = {}
            
            # run 1 time is enough for this stuff

        documents[doc_id][prediction_batch[i]['sequence_id']] = {
            'output': prediction_batch[i]['output'],
            'offsets': prediction_batch[i]['offsets'],
            'text':prediction_batch[i]["text"]}

    print("DOCUMENTS:", len(documents))

    predicted_entities = {}
    # decode each set of labels and store the offsets
    for doc in documents.keys():
        text = documents[doc][0]["text"]
        current_doc = [documents[doc][seq]['output'] for seq in sorted(documents[doc].keys())]
        current_offsets = [documents[doc][seq]['offsets'] for seq in sorted(documents[doc].keys())]
        predicted_entities[doc] = {"decoder": decoder(current_doc, current_offsets, padding=padding, text=text), "document_text": text}
    return predicted_entities

def remove_txt(data):
    new_data = {}
    for k,v in data.items():
        new_k, _ = os.path.splitext(k)
        new_data[new_k]=v
        
    return new_data

@click.command()
@click.option("--checkpoint")
# @click.option("--revision", default="main")
# @click.option("--testset_folder")
@click.option("--out_folder", default="silver_standard_runs")
def main(checkpoint, out_folder):
    testset_folder = "../symptemist-train_all_subtasks+gazetteer+multilingual+test_all_subtasks+bg_231006/symptemist_background-set/all_txt"
    if torch.cuda.is_available():
        #single GPU bc CRF
        assert torch.cuda.device_count()==1
        
    model, tokenizer, config = load_model_local(checkpoint)
    model = model.to(f"cuda")
    tokenizer.model_max_length = 512
    
    test_ds = InferenceDataset(testset_folder, tokenizer=tokenizer, context_size=config.context_size)
    
    eval_datacollator = EvaluationDataCollator(tokenizer=tokenizer, 
                                              padding=True,
                                              label_pad_token_id=tokenizer.pad_token_id)
    
    dl = DataLoader(test_ds, batch_size=8, collate_fn=eval_datacollator)
    
    outputs = []
    for train_batch in tqdm(dl):
        with torch.no_grad():
            train_batch["output"] = model(**train_batch["inputs"].to("cuda"))
            train_batch |= train_batch["inputs"]
            del train_batch["inputs"]
        keys = list(train_batch.keys()) + ["output"]
        for i in range(len(train_batch["output"])):
            outputs.append({k:train_batch[k][i] for k in keys})
            
    predicted_entities = decoder_from_samples(outputs, context_size=config.context_size)
    predicted_entities = remove_txt(predicted_entities)
    
    
    fOut_name = "-".join(checkpoint.split("/")[-2:])
    

    with open(os.path.join(out_folder,f"{fOut_name}.json"),"w") as fOut:
        fOut.write(json.dumps(predicted_entities))
    
    
if __name__ == '__main__':
    main()