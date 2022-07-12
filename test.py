from datasets import load_dataset
import pandas as pd

data_path = "/home/james/Code/Simons22/data/CrisisNLP_labeled_data_crowdflower/"
# dataset = load_dataset("csv", data_files= data_path+'2013_Pakistan_eq/2013_Pakistan_eq_CF_labeled_data.tsv')

df = pd.read_csv(data_path + "2014_California_Earthquake/2014_California_Earthquake_CF_labeled_data.tsv", sep="\t")
print(df["label"].value_counts())
