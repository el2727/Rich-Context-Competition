
# coding: utf-8

# In[ ]:


# Importing necessary libraries

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
import json
import pickle

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

# Reference link: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html


# In[ ]:


# Saving model to a pickle

pickle_file = 'classifier_test.pkl'
pickle_model_test = open(pickle_file, 'wb')
pickle.dump(classifier_MLP, pickle_model_test)
pickle_model_test.close()

# Reference link: http://dataaspirant.com/2017/02/13/save-scikit-learn-models-with-python-pickle/


# In[ ]:


# Open the model from the pickle

model_classifier = open('classifier_test.pkl', 'rb')
classifier_test = pickle.load(model_classifier)

# Import text files for test (10 current files in dev fold)

import glob   
path = '/home/ekaterina/text/text/*.txt'   
files=glob.glob(path)   
for file in files:     
    f=open(file, 'r')  
    content = f.readlines()   
    f.close()

# Make predictions

predicted_MLP = classifier_test.predict(content)
predicted_proba_MLP = classifier_test.predict_proba(content)[:,1]

# Reference link: https://stackoverflow.com/questions/34734714/ipython-jupyter-uploading-folder


# In[ ]:


# Get names of publications in the docs_test (last 83 publications, or 0.25 of test size)

import os
filenames = os.listdir('/home/ekaterina/text/text/') 


# In[ ]:


# Perform regex to bring the name of publications to just a number
 
import re
list_cleaned = []
for i in filenames:
    i = re.sub(r'.txt', '', i)
    list_cleaned.append(i)
    


# In[ ]:


# Zip lists with publications names, predicted dataset and score

output_file = list(zip(predicted_MLP, list_cleaned, predicted_proba_MLP))


# In[ ]:


# Create a dictionary in the output format

list_output = []
for i in output_file:
    dictionary = {'data_set_id': i[0], "publication_id": i[1], "score": i[2]}
    list_output.append(dictionary)
    
# Reference link: https://stackoverflow.com/questions/31181830/adding-item-to-dictionary-within-loop


# In[ ]:


# Saving output JSON file

with open('test_results_updated.json', 'w') as file:
    file.write(json.dumps(list_output, default=str))
    
# Reference link: https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python

