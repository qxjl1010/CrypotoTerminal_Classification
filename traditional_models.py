import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))


def readheadlines():
    try:
        with open("data/keywords_analysis/headlines_list") as train_headline, open("data/keywords_analysis/test_headlines") as test_headline, open("data/keywords_analysis/labels_list") as train_label, open("data/keywords_analysis/test_labels") as test_label:
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
train_texts = train[0]
train_labels = train[1]
test_texts = test[0]
test_labels = test[1]

single_class = len(test_labels) / 4
single_class = int(single_class)


print(len(train_texts),len(test_texts))
 



# SVM
text_clf = Pipeline([('vect', TfidfVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=1, kernel = 'linear', probability=True))])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("SVM accuracy："+str(right/(4*single_class)))
# print("SVM accuracy：",np.mean(predicted==test_labels))


text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',LinearSVC())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("LinearSVC accuracy："+str(right/(4*single_class)))
# print("LinearSVC accuracy：",np.mean(predicted==test_labels))

'''
# Bayes
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',MultinomialNB())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("MultinomialNB accuracy："+str(right/(4*single_class)))
# print("MultinomialNB accuracy：",np.mean(predicted==test_labels))
 
# SGD
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',SGDClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("SGDClassifier accuracy："+str(right/(4*single_class)))
# print("SGDClassifier accuracy：",np.mean(predicted==test_labels))
 
# LogisticRegression
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',LogisticRegression())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("LogisticRegression accuracy："+str(right/(4*single_class)))
# print("LogisticRegression accuracy：",np.mean(predicted==test_labels))
 

 

 

 
# MLPClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',MLPClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("MLPClassifier accuracy："+str(right/(4*single_class)))
# print("MLPClassifier accuracy：",np.mean(predicted==test_labels))
 

# KNeighborsClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',KNeighborsClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("KNeighborsClassifier accuracy："+str(right/(4*single_class)))
# print("KNeighborsClassifier accuracy：",np.mean(predicted==test_labels))
 
# RandomForestClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',RandomForestClassifier(n_estimators=8))])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("RandomForestClassifier accuracy："+str(right/(4*single_class)))
# print("RandomForestClassifier accuracy：",np.mean(predicted==test_labels))
 
# GradientBoostingClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',GradientBoostingClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("GradientBoostingClassifier accuracy："+str(right/(4*single_class)))
# print("GradientBoostingClassifier accuracy：",np.mean(predicted==test_labels))
 
# AdaBoostClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',AdaBoostClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("AdaBoostClassifier accuracy："+str(right/(4*single_class)))
# print("AdaBoostClassifier accuracy：",np.mean(predicted==test_labels))
 
# DecisionTreeClassifier
text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=10000)),
                   ('clf',DecisionTreeClassifier())])
text_clf=text_clf.fit(train_texts,train_labels)
predicted=text_clf.predict(test_texts)
right = 0
for i in range(single_class):
    if predicted[i] == '0\n':
        right += 1
for i in range(single_class):
    if predicted[i+single_class] == '1\n':
        right += 1
for i in range(single_class):
    if predicted[i+2*single_class] == '2\n':
        right += 1
for i in range(single_class):
    if predicted[i+3*single_class] == '3\n':
        right += 1
print("DecisionTreeClassifier accuracy："+str(right/(4*single_class)))
# print("DecisionTreeClassifier accuracy：",np.mean(predicted==test_labels))
'''