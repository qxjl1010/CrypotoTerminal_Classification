# -*- coding: utf-8 -*-
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))


# reading headlines
def readheadlines():
    try:
        with open("data/keywords_analysis/headlines_list_3classes") as train_headline, open("data/keywords_analysis/test_headlines_3classes") as test_headline, open("data/keywords_analysis/labels_list_3classes") as train_label, open("data/keywords_analysis/test_labels_3classes") as test_label:
            clean_train_h = []
            clean_test_h = []
            clean_train_l = []
            clean_test_l = []
            for train_h, train_l in zip(train_headline, train_label):
                
                words = train_h.split()        
                tmp_list = []
                for word in words:                
                    if word[0] == '@' or word[0] == '#':
                        continue
                    word = word.replace(',','')
                    word = word.replace('?','')
                    word = word.replace('.','')
                    if word != '':
                        clean_word = lemmatizer.lemmatize(word)
                        # clean_word = stemmer.stem(clean_word)
                    if clean_word.lower() in stop:
                        continue
                    tmp_list.append(clean_word)
                
                headline = ""
                for w in tmp_list:
                    headline = headline + w + " "
                headline = headline[:-1]
                clean_train_h.append(headline)
                clean_train_l.append(train_l)


            for test_h, test_l in zip(test_headline, test_label):
                test_words = test_h.split()        
                tmp_list = []
                for word in test_words:                
                    if word[0] == '@' or word[0] == '#':
                        continue
                    word = word.replace(',','')
                    word = word.replace('?','')
                    word = word.replace('.','')
                    if word != '':
                        clean_word = lemmatizer.lemmatize(word)
                        # clean_word = stemmer.stem(clean_word)
                    if clean_word.lower() in stop:
                        continue
                    tmp_list.append(clean_word)
                
                headline = ""
                for w in tmp_list:
                    headline = headline + w + " "
                headline = headline[:-1]     
                clean_test_h.append(headline)        
                clean_test_l.append(test_l[:-1])
            train = [clean_train_h, clean_train_l]
            test = [clean_test_h, clean_test_l]
            
            print("**************************************************************")
            print(len(test[0]))
            print("**************************************************************")
            
        return train, test
    except Exception as inst:
        with open("error.log", 'a') as outfile:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)  
            outfile.write(type(inst))
            outfile.write(inst.args)
            outfile.write(inst)
        outfile.close()

train, test = readheadlines()
train_content = train[0]
train_opinion = train[1]
test_content = test[0]
test_opinion = test[1]


# weight calculating
vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  
print(tfidf.shape)

def SVM_eval():
    try:
        train,test = readheadlines()
        train_content = train[0]
        train_opinion = train[1]
        test_content = test[0]
        test_opinion = test[1]
        print("test_opinion is:")
        print(test_opinion)
        # weight calculating
        vectorizer = CountVectorizer()
        tfidftransformer = TfidfTransformer()
        tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))  

        # train and test
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear', probability=True))])
        text_clf = text_clf.fit(train_content, train_opinion)

        joblib.dump(text_clf,'runs/SVM_model')

        clf = joblib.load('runs/SVM_model')

        predicted = clf.predict(test_content)

        pred_proba = clf.predict_proba(test_content)

        # print("SVM pred_proba is:")
        # print(pred_proba)

        tmp_pred = []
        correct = 0
        for pred, opi in zip(predicted, test_opinion):
            tmp_pred.append(pred[:-1])
            if pred[:-1] == opi:
                correct += 1
        predicted = tmp_pred
        accuracy = correct / len(test_opinion)
        print("SVM predicted is:")
        print(predicted)

        # analyse result:
        each_result = len(predicted) / 3
        i = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for rst in predicted:
            lab = 0
            if i < each_result:
                lab = 0
            elif i < each_result * 2:
                lab = 1
            else:
                lab = 2
            if lab == 2 and rst == '2':
                true_positive += 1
            elif lab != 2 and rst == '2':
                false_positive += 1
            elif lab == 2 and rst != '2':
                false_negative += 1
            else:
                true_negative += 1
            i += 1

        
        with open("result", "a") as outfile:
            outfile.write("SVM Result:\n")
            outfile.write("total number:" + str(len(predicted)) + "\n")
            outfile.write("true_positive:" + str(true_positive) + "(" + str(round(true_positive/len(predicted)*100, 2)) + "%)\n")
            outfile.write("false_positive:" + str(false_positive) + "(" + str(round(false_positive/len(predicted)*100, 2)) + "%)\n")
            outfile.write("false_negative:" + str(false_negative) + "(" + str(round(false_negative/len(predicted)*100, 2)) + "%)\n")
            outfile.write("true_negative:" + str(true_negative) + "(" + str(round(true_negative/len(predicted)*100, 2)) + "%)\n\n")
        outfile.close()



        print("SVM result:")
        print("total number:" + str(len(predicted)))
        print("true_positive:" + str(true_positive))
        print("false_positive:" + str(false_positive))
        print("false_negative" + str(false_negative))
        print("true_negative" + str(true_negative))
        


        print("SVM accuracy is: ")
        print(accuracy)

        return pred_proba
    except Exception as inst:
        with open("error.log", 'a') as outfile:
            '''
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)  
            outfile.write(type(inst))
            outfile.write(inst.args)
            outfile.write(inst)
            '''
        outfile.close()

# predict single sentence
'''
word = vectorizer.get_feature_names()
weight = tfidf.toarray()

clf = MultinomialNB().fit(tfidf, opinion)
docs = ["Hello world!", "This is a headline."]
new_tfidf = tfidftransformer.transform(vectorizer.transform(docs))
predicted = clf.predict(new_tfidf)
print predicted
'''


# train and test
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=0.99, kernel = 'linear', probability=True))])
text_clf = text_clf.fit(train_content, train_opinion)
'''
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("test_content is:")
print(test_content)
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
'''
predicted = text_clf.predict(test_content)

pred_proba = text_clf.predict_proba(test_content)

tmp_pred = []
correct = 0
for pred, opi in zip(predicted, test_opinion):
    tmp_pred.append(pred[:-1])
    if pred[:-1] == opi:
        correct += 1
predicted = tmp_pred
accuracy = correct / len(test_opinion)

print("pred_proba is:")
print(pred_proba)
print("predicted is:")
print(predicted)
print("test_opinion is:")
print(test_opinion)
# print('SVC',np.mean(predicted == test_opinion))
print("accuracy is: ")
print(accuracy)
#print metrics.confusion_matrix(test_opinion,predicted) # confusion matrix

# pred_proba = SVM_eval()

# tune parameters
'''
parameters = {'vect__max_df': (0.4, 0.5, 0.6, 0.7),'vect__max_features': (None, 5000, 10000, 15000),
              'tfidf__use_idf': (True, False)}
grid_search = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
grid_search.fit(content, opinion)
best_parameters = dict()
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

'''