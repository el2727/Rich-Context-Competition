A. Classification

1. As mentioned, I've restructured the dataset for the classification approach - I uploaded it as a .tar.gz here, if it's needed: https://app.box.com/folder/52766229995, where subfolder names are dataset indices with relevant publication texts inside them (this data structure is easy to load with sklearn.datasets.load_files function). I did the restructuring with a combination of python and vba code in excel, the process is described in the notebook.

2. Created training and test data by using train_test_split function - 50%-50%.

3. After creating TF-IDF matrices, I planned to run two classifiers that usually perform quite well in multi-label classification - Multinomial Naïve Bayes and Linear SVC. At the stage of creating matrices, I ran out of memory - I freed most of space that I could on my machine, but it looks like would need more at any rate, as creating TF-IDF matrices is quite expensive in this case.. 

3. Nevertheless, I built out the rest of the pipeline - next step is evaluation where there are several options - one can get an option of getting probabilities (predict_proba), as well as confidence scores (decision_function). I thought it would also be interesting to see the results of a regular predict function (which just picks the top results without mentioning probabilities for other cases), so would be able to compare three outputs (just top results, all probabilities and all confidence scores). As I understand, the difference between probabilities and confidence score is marginal, but these are all still separate functions. In terms of evaluation, a full metrics report is included and confusion matrices for both Multinomial Naïve Bayes and Linear SVC.

What is interesting in terms of probabilities output is that it is said that Multinomial Naïve Bayes classifier, since it's making a "naïve" assumption of independence between all variables, tends to push probabilities either towards zero or 1, while other classifiers, such as support vector machines (Linear SVC) make more calibrated predictions in terms of probabilities (more on this here).

B. Information retrieval (document relevance)

In terms of information retrieval or document relevance approach, I've been thinking to try the cosine similarity between the query and the text documents. The query would be a concatenated string of all information on the dataset, such as its title, metadata description and a list of all mentions - this process is described in the notebook.

Then, it's a matter of running a similarity measure between a TF-IDF matrix of a query (which can basically be considered as a short text document as mentioned here) and a TF-IDF matrix of text documents. I'm looking at using cosine similarity in Sklearn.

# Test_Project
