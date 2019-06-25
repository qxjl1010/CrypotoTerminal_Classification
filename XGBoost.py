# -*- coding: utf-8 -*-
import xgboost as xgb
import csv
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))

def readtrain():
    content_train = []
    opinion_train = []
    with open('data/headlines/rank0') as rank0, open('data/headlines/rank1') as rank1, open('data/headlines/rank2') as rank2, open('data/headlines/rank3') as rank3:
        for line in rank0:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_train.append(tmp_headline[:-1])
            opinion_train.append("0")
        for line in rank1:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_train.append(tmp_headline)
            opinion_train.append("1")
        for line in rank2:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_train.append(tmp_headline)
            opinion_train.append("2")
        for line in rank3:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_train.append(tmp_headline)
            opinion_train.append("3")
    rank0.close()
    rank1.close()
    rank2.close()
    rank3.close()

    li = list(range(len(content_train)))
    random.shuffle(li)

    content_train = [x for _,x in sorted(zip(li, content_train))]
    opinion_train = [x for _,x in sorted(zip(li, opinion_train))]

    content_test = []
    opinion_test = []
    with open('data/headlines/rank0_test') as rank0, open('data/headlines/rank1_test') as rank1, open('data/headlines/rank2_test') as rank2, open('data/headlines/rank3_test') as rank3:
        for line in rank0:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_test.append(tmp_headline)
            opinion_test.append("0")
        for line in rank1:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_test.append(tmp_headline)
            opinion_test.append("1")
        for line in rank2:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_test.append(tmp_headline)
            opinion_test.append("2")
        for line in rank3:
            words = line.split()        
            tmp_headline = ''
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
                tmp_headline = tmp_headline + clean_word + ' '
            content_test.append(tmp_headline)
            opinion_test.append("3")
    rank0.close()
    rank1.close()
    rank2.close()
    rank3.close()
    '''
    li = list(range(len(content_test)))
    random.shuffle(li)

    content_test = [x for _,x in sorted(zip(li, content_test))]
    opinion_test = [x for _,x in sorted(zip(li, opinion_test))]
    '''

    with open("data/keywords_analysis/headlines_list","w") as train_content, open("data/keywords_analysis/labels_list","w") as train_opinion, open("data/keywords_analysis/test_headlines","w") as test_content, open("data/keywords_analysis/test_labels","w") as test_opinion:
        for line in content_train:
            train_content.write(line)
            train_content.write('\n')
        for line in opinion_train:
            train_opinion.write(line)
            train_opinion.write('\n')
        for line in content_test:
            test_content.write(line)
            test_content.write('\n')
        for line in opinion_test:
            test_opinion.write(line)
            test_opinion.write('\n')
    train_content.close()
    train_opinion.close()
    test_content.close()
    test_opinion.close()


    return content_train, opinion_train, content_test, opinion_test

def xgb_test(bst, dtest):
    return bst.predict(dtest)

def main():
    train_content, train_opinion, test_content, test_opinion = readtrain()
    # content = segmentWord(train[0])
    # opinion = transLabel(train[1])  
    train_opinion = np.array(train_opinion)
    test_opinion = np.array(test_opinion)   



    vectorizer = CountVectorizer()
    tfidftransformer = TfidfTransformer()
    print(type(train_content[0]))

    tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))
    weight = tfidf.toarray()
    print( tfidf.shape)
    test_tfidf = tfidftransformer.transform(vectorizer.transform(test_content))
    test_weight = test_tfidf.toarray()
    print( test_weight.shape)


    dtrain = xgb.DMatrix(weight, label=train_opinion)
    dtest = xgb.DMatrix(test_weight, label=test_opinion)
    param = {'nthread':1, 'max_depth':8, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':4}  
    evallist  = [(dtrain,'train'), (dtest,'test')]
    num_round = 1
    bst = xgb.train(param, dtrain, num_round, evallist)

    # bst.save_model('runs/XGBoost_model_core')
    
    # joblib.dump(bst,'runs/XGBoost_model')

    preds = bst.predict(dtest,output_margin= True)
    print("preds is:")
    print(preds)


    

    


    print("finally pred:")
    match = 0
    for pred, opinion in zip(preds, test_opinion):
        if float(pred) == float(opinion):
            match += 1
    print(match/len(test_opinion))


if __name__ == '__main__':
    main()