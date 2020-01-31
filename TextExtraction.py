import re
import string
import pandas as pd
import numpy as np
import spacy
from nltk.corpus import stopwords
from sklearn.datasets import make_classification
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve


nlp = spacy.load('en_core_web_sm')
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "!", "#", "$", "%", "&", "(", ")", "*", "+", "-",
                                                     ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_",
                                                     "`", "{", "|", "}", "~"]
ignore_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                'should', 'now', 'uses', 'use', 'using', 'used', 'one', 'also']

dataset = pd.read_csv(r'SampleDataset.csv', sep=",", engine='python')
dataset = dataset.dropna()
print("Old shape:", dataset.shape)


# Data cleaning function
def clean_data(dataset):
    # Remove punctuations
    dataset['text'] = dataset['text'].str.replace(',', " ")
    dataset['text'] = dataset['text'].str.replace('“', " ")
    dataset['text'] = dataset['text'].str.replace('’', " ")
    # Make sure any double-spaces are single
    dataset['text'] = dataset['text'].str.replace('  ', ' ')
    dataset['text'] = dataset['text'].str.replace("\n", " ").replace("\r", " ")
    dataset['text'] = dataset['text'].str.replace("  ", " ")
    # Drop duplicate rows
    dataset.drop_duplicates(subset=['text'], inplace=True)
    print("New shape:", dataset.shape)
    return dataset


data = clean_data(dataset)
dataframe = data


def Remove_StopWords(docs):
    lemmas = []
    for tok in docs:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas
        tokens = [tok for tok in tokens if tok not in STOPLIST]
        tokens = [w.lower() for w in tokens if w not in ignore_words]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


rows_list = []
for output in dataframe.text:
    doc = nlp(output)
    vocabulary = Remove_StopWords(doc)
    vectorizer = CountVectorizer(tokenizer=vocabulary, max_features=10, stop_words=ENGLISH_STOP_WORDS,
                                 ngram_range=(1, 3))

    # Part of Speech (POS) Tagging
    diction = [x.lemma_ for x in [y for y in nlp(str(vectorizer.tokenizer)) if not y.is_stop and y.pos_ != 'PUNCT']]
    rows_list.append(diction)

df = pd.DataFrame(rows_list)
df['News'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df["Type"] = dataframe["Type"]

# duplicate_word_list = [word for word, count in Counter(df["News"]).most_common() if count > 1]
dfModel = pd.DataFrame(df[["News", "Type"]])


dfModel = dfModel.dropna()

X = dfModel.News.values
y = dfModel.Type.values
le = LabelEncoder()
y = le.fit_transform(y)

tf_idf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2))
X_tf_idf = tf_idf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tf_idf, y, test_size=0.3)
svm = svm.SVC(kernel='rbf', random_state=0, gamma=3, C=1.0)


def plot_learing_curve(pipe, title):
    size = 10
    cv = KFold(size, shuffle=False)
    X = dfModel['News']
    y = dfModel['Type']

    X_tf_idf = tf_idf.fit_transform(X)
    pipe = pipe.fit(X_tf_idf, y)

    X, y = make_classification(n_classes=2, n_clusters_per_class=2, shuffle=True)
    train_sizes, train_scores, test_scores = learning_curve(pipe, X, y, n_jobs=5, cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()

    # box-like grid
    plt.grid()

    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    plt.ylim(0, 1.1)
    plt.legend(loc="best")
    plt.show()


plot_learing_curve(svm, "SVM Classifier")