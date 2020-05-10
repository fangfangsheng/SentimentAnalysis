import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegressionCV

np.set_printoptions(precision=2)

# Data Preparation, remove tag and emoticons
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text


# Loading the dataset
df = pd.read_csv('sampledata/movie_data.csv')

# Transforming documents into feature vectors
count = CountVectorizer()

# Word relevancy using term frequency-inverse document frequency
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)

# Tokenization of documents
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# Transform Text Data into TF-IDF Vectors
tfidf = TfidfVectorizer(strip_accents=None,
                       lowercase=False,
                       preprocessor=preprocessor,
                       tokenizer=tokenizer_porter,
                       use_idf=True,
                       norm='l2',
                       smooth_idf=True)

# Data Preparation
y = df.sentiment.values
x = tfidf.fit_transform(df.review)

# Document Classification using Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.7, shuffle=False)

clf = LogisticRegressionCV(cv=5,
                          scoring='accuracy',
                          n_jobs=-1,
                          verbose=3,
                          max_iter=300).fit(X_train, y_train)

saved_model = open('save_model.sav', 'wb')
pickle.dump(clf, saved_model)
saved_model.close()

# Model Evaluation
filename = 'save_model.sav'
saved_clf = pickle.load(open(filename, 'rb'))
accuracy = saved_clf.score(X_test, y_test)

print(f'The accuracy of this logistic regression model is {accuracy * 100:.3f} %.')