import time
from collections import Counter
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, r2_score, classification_report
from sklearn.model_selection import (train_test_split, learning_curve, cross_val_score, cross_val_predict,
                                     ShuffleSplit, KFold, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
import spacy
import re
import string
from nltk.stem.porter import *

nlp = spacy.load('en_core_web_sm')
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

# 1: unreliable
# 0: reliable
train = pd.read_csv(r'TrainDataset.csv', sep=",", engine='python')
print("Old shape of train:", train.shape)
train = train.dropna()


def clean_data(dataframe):
    # dataframe['text'] = dataframe['text'].str.lower()
    dataframe.drop_duplicates(subset=['text'], inplace=True)
    print("New shape:", dataframe.shape)
    return dataframe


traindata = clean_data(train)
dataframe = traindata


def word_extraction(sentence, vocab):
    global tokens
    ignore_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                    'all', 'any', 'both', 'each', 'other', 'some', 'such', 'no', 'nor',
                    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                    'should', 'now', 'uses', 'use', 'using', 'used', 'one', 'also']
    # cleaned_text = [w.lower() for w in words if w not in ignore_words]
    # split into tokens by white space
    tokens = re.sub('[^\w]', " ", sentence).split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    tokens = [w.lower() for w in tokens if w not in STOPLIST]
    tokens = [w.lower() for w in tokens if w not in ignore_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    # output = [word for word in words if len(word) > 2]
    count = vocab.update(tokens)
    return tokens, count


def generate_bow(allsentences, vocab):
    bow = []
    for sentence in allsentences:
        words = word_extraction(sentence, vocab)
        # bag_vector = numpy.zeros(len(words))

        # for w in words:
        #     for i,word in enumerate(words):
        #         if word == w:
        #             bag_vector[i] += 1

        bow.append(words)
    return bow


start = time.time()
vocab = Counter()
data_set = generate_bow(dataframe["text"], vocab)

print(data_set)
print(len(vocab))
print(vocab.most_common(50))

df = pd.DataFrame(data_set)
df['News'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df["Type"] = dataframe["label"]
# duplicate_word_list = [word for word, count in Counter(df["News"]).most_common() if count > 1]
dfModel = pd.DataFrame(df[["News", "Type"]])
dfModel = dfModel.dropna()

X = dfModel.News.values
y = dfModel.Type.values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.1)


pipe = Pipeline([('cvec', CountVectorizer(stop_words=ENGLISH_STOP_WORDS)),
                 ('nb', MultinomialNB())])

# Tune GridSearchCV, So, the smaller the value of alpha(hyper parameter), the higher would be the magnitude of the coefficients.
pipe_params = {'cvec__ngram_range': [(1, 2), (1, 3)], 'nb__alpha': [10, 20, 30]}

kf = KFold(n_splits=3, shuffle=True)
gs = GridSearchCV(pipe, param_grid=pipe_params, cv=kf)
gs.fit(X_train, y_train)

# best_score_ is the 'Mean cross-validated score of the best_estimator.
# best_score is a measure that incorporates how your model performs in models that it has not seen.
print("Train score", gs.score(X_train, y_train))
print("Validation Score", gs.best_score_)
print("Predicted Test score", gs.score(X_test, y_test) * 100)
print(gs.best_params_)
print()
# scores = cross_val_score(gs, X_train, y_train, cv=5)
# print("Training Validated scores: Mean: %0.2f (+/- Std: %0.2f)" % (scores.mean(), scores.std() * 2)

y_true, y_pred = y_test, gs.predict(X_test)
print(classification_report(y_true, y_pred))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
end = time.time()
print("execution time is", end - start)
