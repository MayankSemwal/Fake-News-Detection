import string
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

nlp = spacy.load('en_core_web_sm')
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "!", "#", "$", "%", "&", "(", ")", "*", "+", "-",
                                                     ".", "/", ":", ";",
                                                     "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}",
                                                     "~", "'"]

dataset1000 = pd.read_csv(r'SampleDataset1000.csv', sep=",", engine='python')
dataset2000 = pd.read_csv(r'SampleDataset1000.csv', sep=",", engine='python')
dataset3000 = pd.read_csv(r'SampleDataset1000.csv', sep=",", engine='python')
dataset4000 = pd.read_csv(r'SampleDataset1000.csv', sep=",", engine='python')
dataset5000 = pd.read_csv(r'SampleDataset1000.csv', sep=",", engine='python')

print("Old shape:", dataset1000.shape)
print("Old shape:", dataset2000.shape)
print("Old shape:", dataset3000.shape)
print("Old shape:", dataset4000.shape)
print("Old shape:", dataset5000.shape)


# Data cleaning function
def clean_data(dataset):
    # Remove punctuations
    dataset['text'] = dataset['text'].str.replace(',', " ")
    dataset['text'] = dataset['text'].str.replace('“', " ")
    dataset['text'] = dataset['text'].str.replace("\'", " ")
    dataset['text'] = dataset['text'].str.replace('’', " ")
    # Make sure any double-spaces are single
    dataset['text'] = dataset['text'].str.replace('  ', ' ')
    dataset['text'] = dataset['text'].str.replace("\n", " ").replace("\r", " ")
    dataset['text'] = dataset['text'].str.replace("  ", " ")
    # Drop duplicate rows
    dataset.drop_duplicates(subset=['text'], inplace=True)
    print("New shape:", dataset.shape)
    return dataset
