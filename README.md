### Test_Project

This script describes two pipelines - for classification and for document similarity.

A. Classification

1. I've restructured the dataset for the classification approach, where subfolder names are dataset indices with relevant publication texts inside them (this data structure is easy to load with sklearn.datasets.load_files function). I did the restructuring with a combination of python and vba code in excel.

2. Created training and test data by using train_test_split function - 50%-50%.

3. Performed TF-IDF transformation on data, training classifiers through Pipeline from Sklearn - Multinomial Naive Bayes and SGDClassifier

4. Made predictions on test data, using both probabilities and confidence scores.

5. Evaluated for accuracy - the current sample is small, and accuracy results are not that high. Will try with a larger sample.

B. DOCUMENT SIMILARITY

1. First, extracting 'title', 'description' and 'mention_list' fields from description of dataset JSON and concatenate this information into one string per dataset.

2. Then, creating TD-IDF representations of dataset strings and documents (articles). 

3. Calculating similarity


