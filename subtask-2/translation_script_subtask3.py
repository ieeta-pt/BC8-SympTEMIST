import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
# from utils import load_model
random.seed(42)

from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


test_file_dir = "../symptemist-train_all_subtasks+gazetteer+multilingual+test_all_subtasks+bg_231006/symptemist_test/subtask3-experimental_multilingual"
for f in os.listdir(test_file_dir):
    if f[0] != '.':
        file = test_file_dir+"/"+f
        translation_model = "Helsinki-NLP/opus-mt-"+f[-6]+f[-5]+"-es"
        print(translation_model)
        print(f)
        print(f[-6], f[-5])
        tokenizer = AutoTokenizer.from_pretrained(translation_model)
        model = MarianMTModel.from_pretrained(translation_model, cache_dir="translator").to("cuda")
    
        data = pd.read_csv(file,sep="\t")
    
        data["text_translated"] = data["text"].progress_apply(lambda x: tokenizer.batch_decode(model.generate(**tokenizer(x, return_tensors="pt", padding=True).to("cuda")), skip_special_tokens=True)[0])
    
    
        data.to_csv(f"{file[:-4]}_translated.tsv",sep="\t",index=False)