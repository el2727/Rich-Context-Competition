
# coding: utf-8

# In[2]:


# Importing necessary libraries

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import pandas 
import csv

# Loading data using sklearn load_files function

data_folder = 'data_structure_for_classification/'
all_data = load_files(data_folder)

# Performing train and test datasets splitting using train_test_split function

docs_train, docs_test, y_train, y_test = train_test_split(
    all_data.data, all_data.target, test_size=0.5)

# Training a classifier using sklearn Pipeline
# Text preprocessing, tokenizing and filtering of stopwords are included in CountVectorizer

classifier_svm = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier(penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None, loss='log')),
                          ])

classifier_svm.fit(docs_train, y_train)

# Making predictions and getting probabilities for predictions

predicted_svm = classifier_svm.predict(docs_test)
predicted_proba_svm = classifier_svm.predict_proba(docs_test)


# In[59]:


# Converting probability predictions to dataframe with label names

dataframe = pandas.DataFrame(predicted_proba_svm, columns=classifier_svm.classes_)
print(dataframe)

# Reference link: https://stackoverflow.com/questions/16858652/how-to-find-the-corresponding-class-in-clf-predict-proba


# In[63]:


# Converting to csv

dataframe.to_csv('test_results.csv')


# In[7]:


# Converting csv to dictionary

reader_csv = csv.reader(open('test_results.csv'))

result_list = {}
for i in reader_csv:
    key = i[0]
    result_list[key] = i[1:]
print(result_list)

# Reference link: https://stackoverflow.com/questions/42825102/how-to-save-python-dictionary-into-json-files


# In[80]:


# Converting dictionary to JSON

results = json.dumps(result_list)
file_json = open('test_results.json', 'a')
file_json.write(results)
file_json.close()

# Reference link: https://stackoverflow.com/questions/42825102/how-to-save-python-dictionary-into-json-files


# In[ ]:


# Label names are available in classification_report.target_names
# Also optional - zipping two lists together 
# predictions_list = list(zip(predicted_svm, predicted_proba_svm))

