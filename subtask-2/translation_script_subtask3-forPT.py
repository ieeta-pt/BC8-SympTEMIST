import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
# from utils import load_model
random.seed(42)

from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


file = "../symptemist-train_all_subtasks+gazetteer+multilingual+test_all_subtasks+bg_231006/symptemist_test/subtask3-experimental_multilingual/symptemist_test_subtask3-pt.tsv"

data = pd.read_csv(file,sep="\t")

translation_model = "Helsinki-NLP/opus-mt-pt-ca"
tokenizer = AutoTokenizer.from_pretrained(translation_model)
model = MarianMTModel.from_pretrained(translation_model, cache_dir="translator").to("cuda")


data["text_translated_int"] = data["text"].progress_apply(lambda x: tokenizer.batch_decode(model.generate(**tokenizer(x, return_tensors="pt", padding=True).to("cuda")), skip_special_tokens=True)[0])

translation_model = "Helsinki-NLP/opus-mt-ca-es"
tokenizer = AutoTokenizer.from_pretrained(translation_model)
model = MarianMTModel.from_pretrained(translation_model, cache_dir="translator").to("cuda")

data["text_translated"] = data["text_translated_int"].progress_apply(lambda x: tokenizer.batch_decode(model.generate(**tokenizer(x, return_tensors="pt", padding=True).to("cuda")), skip_special_tokens=True)[0])

data.to_csv(f"{file[:-4]}_translated.tsv",sep="\t",index=False)