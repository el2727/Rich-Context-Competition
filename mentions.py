#!/usr/bin/env python
# coding: utf8
# Module for neuralcoref: https://github.com/huggingface/neuralcoref

import spacy
nlp = spacy.load('en_coref_md')

from sklearn.datasets import load_files

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

# Import and pre-process a text file with mentions as output for string tokens split by ":"

content = [i.split('\n') for i in open('1166.txt_mentions.txt')]
content_tokenized = str(content).split(':')


# Create a keywords list to filter by

keywords_list = ['data,survey data,polls,dataset,study,files,source,microdata,data tapes,questionnaire,response rate,numbers,trend,information,items,estimates,time-series,series,module,findings,responses,database,Data,Survey Data,Polls,Dataset,Study,Files,Source,Microdata,Data Tapes,Questionnaire,Response Rate,Numbers,Trend,Information,Items,Estimates,Time-series,Series,Module,Findings,Responses,Database']


# Pre-process the keywords list to create a list of string tokens

string_keywords = str(keywords_list)
tokenized_keywords = string_keywords.split(',')


# Loop through the keywords list, if a keyword is in a text file (string tokens), return a string token that contains the keyword

mentions_datasets = []
for i in tokenized_keywords:
    for y in content_tokenized:
        if i in y:
            mentions_datasets.append(y)
	
# Write-out to text files

filenames_datasets_list = []
for i in filenames:
    y = str(i) + '_other_datasets.txt'
    filenames_datasets_list.append(y)

for i, y in zip(mentions_datasets, filenames_datasets_list):
    with open(y, 'w') as output_datasets:
        output_datasets.write(i)
	
