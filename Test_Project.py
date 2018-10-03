from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import os.path

data_folder = 'text/'
all_data = load_files(data_folder)

docs_train, docs_test, y_train, y_test = train_test_split(
    all_data.data, all_data.target, test_size=0.5)

classifier_one = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])


classifier_one.fit(docs_train, y_train)

classifier_two = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', SGDClassifier(penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                          ])

classifier_two.fit(docs_train, y_train)

predicted_one = classifier_one.predict(docs_test)

predicted_proba = classifier_one.predict_proba(docs_test)

predicted_two = classifier_two.predict(docs_test)

predicted_confidence_score = classifier_two.decision_function(docs_test)
