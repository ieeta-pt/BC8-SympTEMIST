import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel  
import click


def get_norm_emb(emb):
    return emb/np.linalg.norm(emb,ord=2, axis=-1, keepdims=True)

def cossine_sim(corpus_embeddings, val_embeddings):
    lookup_codes_emb_norm = get_norm_emb(corpus_embeddings).T
        
    val_embed_norm = get_norm_emb(val_embeddings) 
    scores = val_embed_norm @ lookup_codes_emb_norm
    return scores

def get_embeddings(text, batch_size, tokenizer, model):

    embeddings = []
    for i in tqdm(np.arange(0, len(text), batch_size)):
        tokens = tokenizer.batch_encode_plus(text[i:i+batch_size], 
                                           padding="max_length", 
                                           max_length=25, 
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k,v in tokens.items():
            toks_cuda[k] = v.cuda()
        # cls_rep = model(**toks_cuda)[0][:,0,:] # use CLS representation as the embedding
        cls_rep = model(**toks_cuda)[0].mean(axis=1)
        embeddings.append(cls_rep.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings




@click.command()
@click.argument("output")
@click.argument("batch_size", type=int)
@click.argument("model_type", type=int)
@click.argument("threshhold", type=float)
def main(output, batch_size, model_type, threshhold):
    
    print(output, batch_size, model_type, threshhold)
    
    #snomed corpues
    snomed_corpus = "../symptemist-train_all_subtasks+gazetteer+multilingual+test_all_subtasks+bg_231006/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv"
    snomed = pd.read_csv(snomed_corpus,sep="\t")
    snomed_code_dictionary = dict()
    #cast all terms to lower, and build dictionary for them
    for row in snomed.iterrows():
        if row[1]['term'].lower() in snomed_code_dictionary.keys():
            snomed_code_dictionary[row[1]['term'].lower()] = snomed_code_dictionary[row[1]['term'].lower()]+ "+"+ str(row[1]['code'])
        else:
            snomed_code_dictionary[row[1]['term'].lower()] = str(row[1]['code'])

    
    #train files
    data_file = "../symptemist-train_all_subtasks+gazetteer+multilingual+test_all_subtasks+bg_231006/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2.tsv"
    data = pd.read_csv(data_file,sep="\t") 
    data_dictionary = dict()
    for row in data.iterrows():
        if row[1]['text'].lower() in data_dictionary.keys():
            if str(row[1]['code']) not in data_dictionary[row[1]['text'].lower()] :
                data_dictionary[row[1]['text'].lower()] = data_dictionary[row[1]['text'].lower()]+ "+"+ str(row[1]['code'])
        else:
            data_dictionary[row[1]['text'].lower()] = str(row[1]['code'])

    #test file
    test_file = "../symptemist-train_all_subtasks+gazetteer+multilingual+test_all_subtasks+bg_231006/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv"
    test_data = pd.read_csv(test_file,sep="\t")

    #handles model loading
    if model_type == 1:
        checkpoint ="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR-large" #USE ME, ITS BETTER!
    else:
        checkpoint ="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    print(f"Using model checkpoint: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
    model = AutoModel.from_pretrained(checkpoint).cuda()

    #get embeddings
    snomed_text = list(snomed_code_dictionary.keys())
    test_data_text = list(test_data['text'].str.lower())
    data_text = list(data_dictionary.keys())
    all_data = snomed_text+data_text

    embeddings = get_embeddings(all_data, batch_size, tokenizer, model)
    test_data_embeddings = get_embeddings(test_data_text, batch_size, tokenizer, model)

    #performs similarity
    scores = cossine_sim(embeddings,test_data_embeddings)

    codes = list(snomed_code_dictionary.values()) + list(data_dictionary.values())

    #predictions
    list_of_codes_per_sample = []
    direct_matches_s = 0
    direct_matches_t = 0
    threshhold_matches = 0
    for text, score, index in zip(test_data_text, scores, np.argmax(scores, axis=-1)):
        #lookup exact matches:
        if text in data_dictionary.keys():
            list_of_codes_per_sample.append(str(data_dictionary[text]))
            direct_matches_t +=1
        elif text in snomed_code_dictionary.keys():
            list_of_codes_per_sample.append(str(snomed_code_dictionary[text]))
            direct_matches_s +=1
        #if not exact matching try find threshhold
        else:
            if score[index]>threshhold:
                threshhold_matches+=1
                list_of_codes_per_sample.append(codes[index])
            else:
                list_of_codes_per_sample.append('NO_CODE')
    print(direct_matches_s, direct_matches_t, threshhold_matches, len(test_data_text), (direct_matches_t+direct_matches_s)/len(test_data_text))
    test_data['code'] = list_of_codes_per_sample
    test_data.to_csv(output, sep = "\t")
    



if __name__ == '__main__':
    main()