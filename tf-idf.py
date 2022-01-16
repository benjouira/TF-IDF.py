from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())


print(X.shape)
print(X)

# ************************************************

from scipy.sparse import hstack
import numpy as np
X_train_dtm = hstack((X,np.array(l)[:,None]))

print(X.shape)
print(X_train_dtm.shape)
print(X_train_dtm)

