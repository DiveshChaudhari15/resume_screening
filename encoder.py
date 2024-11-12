import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Example data
documents = ["This is the first document.", "This document is the second document."]
labels = [0, 1]  # Example labels (0 or 1)

# Create and fit the TF-IDF vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documents)

# Create and fit the classifier
clf = RandomForestClassifier()
clf.fit(X, labels)

# Step 2: Save the model and vectorizer
with open('clf.pkl', 'wb') as clf_file:
    pickle.dump(clf, clf_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)
