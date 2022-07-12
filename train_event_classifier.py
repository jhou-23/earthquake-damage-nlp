import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification

data_dir = "/home/james/Code/Simons22/data/CrisisNLP_labeled_data_crowdflower/event_balanced"
dataset = load_dataset('csv', data_files={'train':data_dir+"/train.csv", 'test':data_dir+"/test.csv"})

tokenizer = AutoTokenizer.load_tokenizer('roberta-base-cased')