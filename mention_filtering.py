
# coding: utf-8

# In[81]:


# Import and pre-process a text file for tokens split by ":"

content = [i.split('\n') for i in open('1166.txt_mentions.txt')]
content_tokenized = str(content).split(':')


# In[83]:


# Create a keywords list to filter by

keywords_list = ['data,survey data,polls,dataset,study,files,source,microdata,data tapes,questionnaire,response rate,numbers,trend,information,items,estimates,time-series,series,module,findings,responses,database,Data,Survey Data,Polls,Dataset,Study,Files,Source,Microdata,Data Tapes,Questionnaire,Response Rate,Numbers,Trend,Information,Items,Estimates,Time-series,Series,Module,Findings,Responses,Database']


# In[91]:


# Pre-process the keywords list to create a list of string tokens

string_keywords = str(keywords_list)
tokenized_keywords = string_keywords.split(',')


# In[111]:


# Loop through the keywords list, if a keyword is in a text file (string tokens), return a string token that contains the keyword

for i in tokenized_keywords:
    for y in content_tokenized:
        if i in y:
            print(y)

