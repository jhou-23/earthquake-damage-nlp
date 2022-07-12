import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, AdamW, get_scheduler, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_metric
import torch

accelerator = Accelerator()

data_dir = "/home/james/Code/Simons22/data/CrisisNLP_labeled_data_crowdflower/type_balanced"
dataset = load_dataset('csv', data_files={'train':data_dir+"/train.csv", 'test':data_dir+"/test.csv"})

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def transform_labels(label):
    label = label["label"]
    num = -1
    # if label == "injured_or_dead_people":
    #     num = 0
    # elif label == "donation_needs_or_offers_or_volunteering_services":
    #     num = 1
    # elif label == "sympathy_and_emotional_support":
    #     num = 2
    # elif label == "other_useful_information":
    #     num = 3
    # elif label == "not_related_or_irrelevant":
    #     num = 4
    # elif label == "infrastructure_and_utilities_damage":
    #     num = 5
    # elif label == "caution_and_advice":
    #     num = 6
    # elif label == "missing_trapped_or_found_people":
    #     num = 7
    # elif label == "displaced_people_and_evacuations":
    #     num = 8
    if label == "injured_or_dead_people":
        num=0
    else:
        num=1

    return {"labels":num}

def tokenize_data(example):
    return tokenizer(example["tweet_text"], padding='max_length')


dataset = dataset.map(tokenize_data, batched=True)

dataset = dataset.map(transform_labels, remove_columns=["tweet_id", "tweet_text", "label"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(dataset["train"], shuffle=True, batch_size=8, collate_fn=data_collator)
test_loader = DataLoader(dataset["test"], batch_size=8, collate_fn=data_collator)

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

train_loader, test_loader, model, optimizer = accelerator.prepare(train_loader, test_loader, model, optimizer)

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric1 = load_metric("accuracy")
metric2 = load_metric("f1")
model.eval()
for batch in test_loader:
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric1.add_batch(predictions=predictions, references=batch["labels"])
    metric2.add_batch(predictions=predictions, references=batch["labels"])

print(metric1.compute())
# print(metric2.compute())
print(metric2.compute(average="weighted"))
accelerator.unwrap_model(model).save_pretrained("/home/james/Code/Simons22/models/type_balanced")