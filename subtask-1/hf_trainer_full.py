import random


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoConfig
import json

from data import load_train_test_split, SelectModelInputs, BIOTagger, RandomlyUKNTokens, EvaluationDataCollator, RandomlyReplaceTokens
from trainer import NERTrainer
from BERT_LSTM_CRF import BERTLstmCRF
from BERT_DENSE_CRF import BERTDenseCRF

from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

from utils import setup_wandb, create_config

import os
import argparse

from metrics import NERMetrics

parser = argparse.ArgumentParser(description="")
parser.add_argument("checkpoint", type=str)
parser.add_argument("--model_type_arch", type=str, default="bilstm")
parser.add_argument("--percentage_tags", type=float, default=0.2)
parser.add_argument("--augmentation", type=str, default=None)
parser.add_argument("--context", type=int, default=64)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--random_seed", type=int, default=42)


args = parser.parse_args()

model_checkpoint = args.checkpoint#"pubmed_bert_classifier_V2_synthetic/checkpoint-29268"

name = model_checkpoint.split("/")[0]

if args.augmentation is not None:
    dir_name = f"{name}-{args.model_type_arch}-{args.context}-{args.augmentation}-{args.percentage_tags}-{args.random_seed}-roberta-es-ner-trainer-full-v6"
else:
    dir_name = f"{name}-{args.model_type_arch}-{args.context}-{args.random_seed}-roberta-es-ner-trainer-full-v6"

setup_wandb()
training_args = create_config("roberta_trainer_config.yaml", 
                              output_dir=dir_name,
                              num_train_epochs=args.epochs,
                              dataloader_num_workers=4,
                              per_device_train_batch_size=args.batch,
                              #gradient_accumulation_steps= 2, # batch 16 - 32 -64
                              per_device_eval_batch_size= args.batch,
                              prediction_loss_only=False,
                              evaluation_strategy="no",
                              save_strategy="epoch",
                              seed=args.random_seed,
                              data_seed=args.random_seed)

#model_checkpoint = "pubmed_bert_classifier_V2_synthetic/checkpoint-32490"
#model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

random.seed(args.random_seed)

CONTEXT_SIZE = args.context

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.model_max_length = 512

transforms = [BIOTagger(), SelectModelInputs()]

train_augmentation = None
if args.augmentation:
    if args.augmentation=="ukn": 
        print("Note: The trainer will use RandomlyUKNTokens augmentation")
        train_augmentation = [RandomlyUKNTokens(tokenizer=tokenizer, 
                            context_size=CONTEXT_SIZE,
                            prob_change=0.5, 
                            percentage_changed_tags=args.percentage_tags)]
    elif args.augmentation=="random":
        print("Note: The trainer will use RandomlyReplaceTokens augmentation")
        train_augmentation = [RandomlyReplaceTokens(tokenizer=tokenizer, 
                            context_size=CONTEXT_SIZE,
                            prob_change=0.5, 
                            percentage_changed_tags=args.percentage_tags)]
    
train_ds, _ = load_train_test_split("dataset/medprocner_train",
                                          tokenizer=tokenizer,
                                          context_size=CONTEXT_SIZE,
                                          test_split_percentage=0,
                                          train_transformations=transforms,
                                          train_augmentations=train_augmentation,
                                          test_transformations=None)

id2label = {0:"O", 1:"B", 2:"I"}
label2id = {v:k for k,v in id2label.items()}

config = AutoConfig.from_pretrained(model_checkpoint)
config.id2label = id2label
config.label2id = label2id
config.vocab_size = tokenizer.vocab_size

config.args_random_seed = args.random_seed

config.augmentation = args.augmentation
config.context_size = args.context
config.percentage_tags = args.percentage_tags

config.freeze = False
config.crf_reduction = "mean"

config.model_type_arch = args.model_type_arch

def model_init():
    if args.model_type_arch=="bilstm":
        return BERTLstmCRF(config=config)
    elif args.model_type_arch=="dense":
        return BERTDenseCRF(config=config)

trainer = NERTrainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, 
                                                     padding="longest",
                                                     label_pad_token_id=tokenizer.pad_token_id),
    compute_metrics=NERMetrics(context_size=CONTEXT_SIZE)
    
)
# input_ids, attention_mask, labels

# decode(labels) -> true spans
# decode(predicted) -> 


trainer.train()