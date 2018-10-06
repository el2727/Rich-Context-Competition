
# coding: utf-8

# In[1]:


# Importing necessary libraries

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import json

# Loading data using sklearn load_files function

data_folder = 'data_structure_for_classification/'
all_data = load_files(data_folder)

# Performing train and test datasets splitting using train_test_split function

docs_train, docs_test, y_train, y_test = train_test_split(
    all_data.data, all_data.target, shuffle=None)

# Training a classifier using sklearn Pipeline
# Text preprocessing, tokenizing and filtering of stopwords are included in CountVectorizer

classifier_MLP = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                          ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30,), learning_rate='adaptive', random_state=1)),
                          ])

classifier_MLP.fit(docs_train, y_train)

# Making predictions and getting probabilities for predictions

predicted_MLP = classifier_MLP.predict(docs_test)
predicted_proba_MLP = classifier_MLP.predict_proba(docs_test)[:,1]

# Reference link: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html


# In[2]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predicted_MLP))
#print(classification_report(y_test, predicted_MLP, target_names=all_data.target_names))


# In[3]:


# Get names of publications in the docs_test (last 83 publications, or 0.25 of test size)

test_file_names = list(all_data.filenames[83:])


# In[4]:


# Perform regex to bring the name of publications to just a number

import re
test_names_list = []
for i in test_file_names:
    i = re.sub(r'data_structure_for_classification/', '', i)
    test_names_list.append(i)
    
list_cleaned = []
for i in test_names_list:
    i = re.sub(r'.txt', '', i)
    list_cleaned.append(i)
    
list_finalized = []
for i in list_cleaned:
    i = re.sub(r'[0-9][0-9][0-9][0-9]/''', '', i)
    list_finalized.append(i)


# In[6]:


# Zip lists with publications names, predicted dataset and score

output_file = list(zip(predicted_MLP, list_finalized, predicted_proba_MLP))


# In[7]:


# Create a dictionary in the output format

list_output = []
for i in output_file:
    dictionary = {'data_set_id': i[0], "publication_id": i[1], "score": i[2]}
    list_output.append(dictionary)
    
# Reference link: https://stackoverflow.com/questions/31181830/adding-item-to-dictionary-within-loop


# In[20]:


# Saving output JSON file

with open('test_results_finalized.json', 'w') as file:
    file.write(json.dumps(list_output, default=str))
    
# Reference link: https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python

