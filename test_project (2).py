
# coding: utf-8

# In[42]:


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
    all_data.data, all_data.target, shuffle=None)

# Training a classifier using sklearn Pipeline
# Text preprocessing, tokenizing and filtering of stopwords are included in CountVectorizer

classifier_svm = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                          ('clf', SGDClassifier(penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None, loss='log')),
                          ])

classifier_svm.fit(docs_train, y_train)

# Making predictions and getting probabilities for predictions

predicted_svm = classifier_svm.predict(docs_test)
predicted_proba_svm = classifier_svm.predict_proba(docs_test)[:,1]


# In[59]:


# Get names of publications in the docs_test (last 83 publications, or 0.25 of test size)

test_file_names = list(all_data.filenames[83:])


# In[61]:


# Perform regex to bring the name of publications to just a number

import re
test_names_list = []
for i in test_file_names:
    i = re.sub(r'data_structure_for_classification/', '', i)
    test_names_list.append(i)


# In[64]:


list_cleaned = []
for i in test_names_list:
    i = re.sub(r'.txt', '', i)
    list_cleaned.append(i)


# In[89]:


list_finalized = []
for i in list_cleaned:
    i = re.sub(r'[0-9][0-9][0-9][0-9]/''', '', i)
    list_finalized.append(i)


# In[104]:


# Zip lists with publications names, predicted dataset and score

output_file = list(zip(predicted_svm, list_finalized, predicted_proba_svm))


# In[110]:


# Create a dictionary in the output format

list_output = []
for i in output_file:
    dictionary = {'data_set_id': i[0], "publication_id": i[1], "score": i[2]}
    list_output.append(dictionary)
    
# Reference link: https://stackoverflow.com/questions/31181830/adding-item-to-dictionary-within-loop


# In[115]:


print(list_output)


# In[30]:


# Perform probability calibration
# Reference links: http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py
# http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV.fit
# http://scikit-learn.org/stable/modules/calibration.html

from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

calibrated_classifier = CalibratedClassifierCV(classifier_svm, method='sigmoid')
calibrated_classifier.fit(docs_train, y_train)
predicted_calibrated = calibrated_classifier.predict_proba(docs_test)[:,1]

classifier_svm_score = brier_score_loss(y_test, predicted_proba_svm)
calibrated_classifier_score = brier_score_loss(y_test, predicted_calibrated)

print(classifier_svm_score, calibrated_classifier_score)


# In[59]:


# Converting probability predictions to dataframe with label names

#dataframe = pandas.DataFrame(predicted_proba_svm, columns=classifier_svm.classes_)
#print(dataframe)

# Reference link: https://stackoverflow.com/questions/16858652/how-to-find-the-corresponding-class-in-clf-predict-proba


# In[63]:


# Converting to csv

#dataframe.to_csv('test_results.csv')


# In[7]:


# Converting csv to dictionary

#reader_csv = csv.reader(open('test_results.csv'))

#result_list = {}
#for i in reader_csv:
#    key = i[0]
#    result_list[key] = i[1:]
#print(result_list)

# Reference link: https://stackoverflow.com/questions/42825102/how-to-save-python-dictionary-into-json-files


# In[80]:


# Converting dictionary to JSON

#results = json.dumps(result_list)
#file_json = open('test_results.json', 'a')
#file_json.write(results)
#file_json.close()

# Reference link: https://stackoverflow.com/questions/42825102/how-to-save-python-dictionary-into-json-files


# In[ ]:


# Label names are available in classification_report.target_names
# Also optional - zipping two lists together 
# predictions_list = list(zip(predicted_svm, predicted_proba_svm))

