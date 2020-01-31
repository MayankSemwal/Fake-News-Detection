import string
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

nlp = spacy.load('en_core_web_sm')
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "!", "#", "$", "%", "&", "(", ")", "*", "+", "-",
                                                     ".", "/", ":", ";",
                                                     "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}",
                                                     "~", "'"]
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

dataset = pd.read_csv(r'SampleDataset.csv', sep=",", engine='python')
dataset.drop(['date', 'title'], axis=1, inplace=True)
print("Old shape:", dataset.shape)


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


data = clean_data(dataset)
dataframe = data
rows_list = []
lemmas = []
global tokens


def Remove_StopWords(docs):
    global tokens
    lemmas = []
    for tok in docs:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas
        tokens = [tok for tok in tokens if tok not in STOPLIST]
        tokens = [w.lower() for w in tokens if w not in ignore_words]
        tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


def generate_ngrams(s, n):
    ngrams = zip(*[s[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


for output in dataframe.text:
    doc = nlp(output)
    vocabulary = Remove_StopWords(doc)
    vectorizer = CountVectorizer(tokenizer=vocabulary, stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2))
    diction = [x.lemma_ for x in [y for y in nlp(str(vectorizer.tokenizer)) if not y.is_stop and y.pos_ != 'PUNCT']]
    bigram = generate_ngrams(diction, 2)
    rows_list.append(bigram)

df = pd.DataFrame(rows_list)
df['News'] = df[df.columns[0:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df["Type"] = dataframe["Type"]
dfModel = pd.DataFrame(df[{"News", "Type"}])
dfModel = dfModel.dropna()

X = dfModel.News.values
y = dfModel.Type.values

tf_idf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2))

X_tf_idf = tf_idf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tf_idf, y, test_size=0.3)

logit = LogisticRegression(solver='liblinear')

# train model
logit.fit(X_train, y_train)

# get predictions for article section
y_pred = logit.predict(X_test)

print("Logistic Regression F1 and Accuracy Scores :")
print("F1 score {:.4}%".format(f1_score(y_test, y_pred, average='macro') * 100))
print("Accuracy score {:.4}%".format(accuracy_score(y_test, y_pred) * 100))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
