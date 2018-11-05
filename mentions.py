#!/usr/bin/env python
# coding: utf8
# Module for neuralcoref: https://github.com/huggingface/neuralcoref

# A full pipeline for running a coreference resolution neural network model for mentions of other datasets and 
# filtering the results by keywords to preserve those with the mention of datasets. Outputs a JSON format file.

import spacy
import json
import glob
import os
import re
from sklearn.datasets import load_files

nlp = spacy.load('en_coref_md')


# Loading data using sklearn load_files function

data_folder = 'data_structure_for_classification/'
all_data = load_files(data_folder)

# Saving and loading all text documents

documents = all_data.data
        
# Converting bytes-object into string

text = []
for i in documents:
    i = i.decode("utf8")
    text.append(i)
    
# Reference link: https://stackoverflow.com/questions/606191/convert-bytes-to-a-string

# Loading neuralcoref module

nlp = spacy.load('en_coref_md')

# Performing coreference resolution and extracting mention clusters for every text document through a for loop

mentions_list = []
for i in text:
	doc = nlp(i)
	mentions = doc._.coref_clusters
	mentions_list.append(mentions)

# Reference link:  https://github.com/huggingface/neuralcoref

# Saving the output as text files

filenames = all_data.filenames

filenames_list = []
for i in filenames:
    y = str(i) + '_mentions.txt'
    filenames_list.append(y)

for i, y in zip(mentions_list, filenames_list):
    with open(y, 'w') as output:
        output.write(i)

# Reference link: https://stackoverflow.com/questions/6673092/printing-out-elements-of-list-into-separate-text-files-in-python
# Reference link: https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters

# Filtering of dataset mentions

# Using the following command to create a folder with just mentions of other datasets and a publication ID:

# find . -name "*_mentions.txt" -exec mv "{}" ~/mentions_other_datasets \;

# Read in multiple text files

files = glob.glob(os.path.join(os.getcwd(), "mentions_other_datasets_finalized/mentions_other_datasets/", "*.txt"))

content_files = []

for i in files:
    with open(i) as y:
        content_files.append(y.read())
	
# Reference link: https://stackoverflow.com/questions/42407976/loading-multiple-text-files-from-a-folder-into-a-python-list-variable

# Create a keywords list to filter by

keywords_list = ['data,survey data,polls,dataset,study,files,source,microdata,data tapes,questionnaire,numbers,trend,information,time-series,series,module,database,sample,Sample,publication,Publication,Data,Survey Data,Polls,Dataset,Study,Files,Source,Microdata,Data Tapes,Questionnaire,Information,Time-series,Series,Module,Findings,Responses,Database']

# Pre-process the keywords list to create a list of string tokens

string_keywords = str(keywords_list)
tokenized_keywords = string_keywords.split(',')

# Process the text file contents

processed_content = []
for i in content_files:
    content = i.replace('],', '],\n')
    content_read = content.split('\n')
    processed_content.append(content_read)

# Get the keys list and filter

keys_list = []
for i in processed_content:
    for y in i:
        keys_list.append(y.split(':')[0])
	
filtered_keys = []
for b in keys_list:
    for c in tokenized_keywords:
        if c in b:
            filtered_keys.append(b)
	

# Index the text files content

index_list = range(1, len(processed_content) + 1)
zipped = list(zip(index_list, processed_content))

# Filter text files content by keywords

results_list = []
for d in filtered_keys:
    for w in zipped:
        for y, x in [w]:
            if d in x:
                results_list.append(w)
		
# Convert results to dictionary and further JSON format

list_for_dictionary = []
for f in results_list:
    list_for_dictionary.append(f[1])

list_output = []
for i in list_for_dictionary:
    for y in i:
        list_output.append({y})
	
# Get the filenames with publication ID

filenames = os.listdir('mentions_other_datasets_finalized/mentions_other_datasets/')
filenames_processed = []
for i in filenames:
    i = re.sub('_mentions.txt', '', i)
    filenames_processed.append(i)
	
# Output as JSON with filtered mentions and publication ID in the JSON file name

filenames_list = []
for i in filenames_processed:
    y = str(i) + '_filtered_mentions.json'
    filenames_list.append(y)

for i, y in zip(list_output, filenames_list):
    with open(os.path.join('mentions_json_output/', y), 'w') as file:
        file.write(json.dumps(list_output, default=str))

