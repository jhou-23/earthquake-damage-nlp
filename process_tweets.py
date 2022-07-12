import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
type_classifier = AutoModelForSequenceClassification.from_pretrained("/home/james/Code/Simons22/models/type_balanced").eval()
event_classifier = AutoModelForSequenceClassification.from_pretrained("/home/james/Code/Simons22/models/event_balanced").eval()

qna_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qna_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

def probe_info(text):
    question1 = "How many died?"
    question2 = "How many injured?"
    question3 = "Where is this?"

    inputs1 = qna_tokenizer(question1, text, return_tensors="pt")
    inputs2 = qna_tokenizer(question2, text, return_tensors="pt")
    inputs3 = qna_tokenizer(question3, text, return_tensors="pt")

    with torch.no_grad():
        outputs1 = qna_model(**inputs1)
        outputs2 = qna_model(**inputs2)
        outputs3 = qna_model(**inputs3)
    
    start_index1 = outputs1.start_logits.argmax()
    end_index1 = outputs1.end_logits.argmax()
    start_index2 = outputs2.start_logits.argmax()
    end_index2 = outputs2.end_logits.argmax()
    start_index3 = outputs3.start_logits.argmax()
    end_index3 = outputs3.end_logits.argmax()
    
    answer1 = qna_tokenizer.decode(inputs1.input_ids[0, start_index1 : end_index1 + 1])
    answer2 = qna_tokenizer.decode(inputs2.input_ids[0, start_index2 : end_index2 + 1])
    answer3 = qna_tokenizer.decode(inputs3.input_ids[0, start_index3 : end_index3 + 1])
    print(answer1)
    print(answer2)
    print(answer3)    

def filter_type(text):
    inputs = tokenizer.encode(text, return_tensors='pt')
    output = type_classifier(inputs)
    print('filter')
    print(output)

def filter_event(text):
    inputs = tokenizer.encode(text, return_tensors='pt')
    output = event_classifier(inputs)
    print(output)


    # logits = event_classifier()

def process_tweet(text):
    filter_event(text)
    filter_type(text)
    probe_info(text)

def main():
    data_dir = "/home/james/Code/Simons22/data/"
    df = pd.read_csv(data_dir + "us_csv_haiti.csv")
    df = df[df["lang"]=="en"]

    process_tweet("LetΓÇÖs keep Haiti ≡ƒç¡≡ƒç╣ #Haiti in our prayers. The toll has risen to at least 227 deaths, with hundreds injured and missing, after a 7.2 magnitude earthquake struck Haiti. Prime Minister Ariel Henry said he was rushing aid to damaged towns and hospitals overwhelmed with casualties.")



if __name__ == '__main__':
    main()