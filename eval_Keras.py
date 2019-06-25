import tensorflow as tf
import numpy as np
import os
import data_helpers
import Keywords_analysis
import SVM

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional, Flatten, Dropout

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from ddrop.layers import DropConnect

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# In eval, Keras load model and predict



def readheadlines():
    with open("data/keywords_analysis/headlines_list") as train_headline, open("data/keywords_analysis/test_headlines") as test_headline, open("data/keywords_analysis/labels_list") as train_label, open("data/keywords_analysis/test_labels") as test_label:
        clean_train_h = []
        clean_test_h = []
        clean_train_l = []
        clean_test_l = []
        for train_h, train_l in zip(train_headline, train_label):
            
            words = train_h.split()        
            tmp_list = []
            for word in words:                
                if word[0] == '@' or word[0] == '#' or 'http' in word:
                    continue
                word = word.replace(',','')
                word = word.replace('?','')
                word = word.replace('.','')
                if word != '':
                    clean_word = lemmatizer.lemmatize(word)
                    # clean_word = stemmer.stem(clean_word)
                '''
                if clean_word.lower() in stop:
                    continue
                '''
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
                if word[0] == '@' or word[0] == '#' or 'http' in word:
                    continue
                word = word.replace(',','')
                word = word.replace('?','')
                word = word.replace('.','')
                if word != '':
                    clean_word = lemmatizer.lemmatize(word)
                    # clean_word = stemmer.stem(clean_word)
                '''
                if clean_word.lower() in stop:
                    continue
                '''
                tmp_list.append(clean_word)
            
            headline = ""
            for w in tmp_list:
                headline = headline + w + " "
            headline = headline[:-1]     
            clean_test_h.append(headline)        
            clean_test_l.append(test_l[:-1])
        train = [clean_train_h, clean_train_l]
        test = [clean_test_h, clean_test_l]
        
        
    return train, test



def eval():
    print('Loading data...')
    train, test = readheadlines()

    x_train = train[0]
    y_train = list(map(int,train[1]))
    x_test = test[0]
    y_test = list(map(int,test[1]))

    train_text = x_train + x_test

    tok = Tokenizer()
    tok.fit_on_texts(train_text)
    x_train = tok.texts_to_sequences(x_train)
    x_test = tok.texts_to_sequences(x_test)

    y_train = np.asarray(y_train, dtype=np.int64)
    y_test = np.asarray(y_test, dtype=np.int64)

    model = load_model('model.h5', custom_objects={'DropConnect':DropConnect})


    print('Predict result:')
    pred_possibilities = model.predict(x_test)
    y_eval = pred_possibilities.argmax(axis=-1)


    return y_eval, pred_possibilities

def change_ensemble_weight(logits_list, SVM_pred_list, y_eval):
    best_ratio_neural = 0.5
    best_ratio_SVM = 0.5
    ratio_neural = 1.1
    ratio_SVM = -0.1

    best_result = 0

    ensemble_pred_list = []
    ensemble_label_list = []
    
    # print("logits_list is:")
    # print(logits_list)
    print("SVM_pred_list is:")
    print(SVM_pred_list)
    print("opinion is:")
    print(y_eval)
    for i in range(11):
        ratio_neural -= 0.1
        ratio_SVM += 0.1
        correct_num = 0
        tmp_list = []
        label_list = []
        print("nn ratio:" + str(ratio_neural))
        print("SVM ratio:" + str(ratio_SVM))
        for pred_neural, pred_SVM, label in zip(logits_list, SVM_pred_list, y_eval):
            final_min1 = pred_neural[0] * ratio_neural + pred_SVM[0] * ratio_SVM
            final_pos0 = pred_neural[1] * ratio_neural + pred_SVM[1] * ratio_SVM
            final_pos1 = pred_neural[2] * ratio_neural + pred_SVM[2] * ratio_SVM
            final_pos2 = pred_neural[3] * ratio_neural + pred_SVM[3] * ratio_SVM
            final_pred = [final_min1, final_pos0, final_pos1, final_pos2]
            tmp_list.append(final_pred)
            label_pred = final_pred.index(max(final_pred))
            label_list.append(label_pred)
            '''
            # can print the pred here if need
            print("---------------------------------------------------")
            print("label is:" + str(label))
            print("label_pred is:" + str(label_pred))
            print("correct_num is:" + str(correct_num))
            print("---------------------------------------------------")
            '''
            if label_pred == label:
                correct_num += 1
        
        rst = correct_num / len(y_eval)
        print("label_list is:")
        print(label_list)
        print("rst is:" + str(rst))
        print("best_result is:" + str(best_result))
        if rst > best_result:
            best_result = rst
            best_ratio_neural = ratio_neural
            best_ratio_SVM = ratio_SVM
            ensemble_pred_list = tmp_list
            ensemble_label_list = label_list
    
    print("best ratio is:")
    print("ratio neural:")
    print(best_ratio_neural)
    print("ratio SVM:")
    print(best_ratio_SVM)

    print("Final prediction is:")
    print(ensemble_label_list)

    # analyse result:
    each_result = len(ensemble_label_list) / 4
    i = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    for rst in ensemble_label_list:
        lab = 0
        if i < each_result:
            lab = 0
        elif i < each_result * 2:
            lab = 1
        elif i < each_result * 3:
            lab = 2
        else:
            lab = 3
        if lab == 3 and rst == 3:
            true_positive += 1
        elif lab != 3 and rst == 3:
            false_positive += 1
        elif lab == 3 and rst != 3:
            true_negative += 1
        i += 1
    

    with open("result", "a") as outfile:
        outfile.write("Total Result:\n")
        outfile.write("total number:" + str(len(ensemble_label_list)) + "\n")
        outfile.write("true_positive:" + str(true_positive) + "(" + str(round(true_positive/len(ensemble_label_list)*100, 2)) + "%)\n")
        outfile.write("false_positive:" + str(false_positive) + "(" + str(round(false_positive/len(ensemble_label_list)*100, 2)) + "%)\n")
        outfile.write("true_negative:" + str(true_negative) + "(" + str(round(true_negative/len(ensemble_label_list)*100, 2)) + "%)\n")
        outfile.write("=======================================================================================\n")
        outfile.write("=======================================================================================\n")
    outfile.close()

    print("total result:")
    print("total number:" + str(len(ensemble_label_list)))
    print("true_positive:" + str(true_positive))
    print("false_positive:" + str(false_positive))
    print("true_negative:" + str(true_negative))

    print("best ensemble result is:")
    print(best_result)

    return best_ratio_neural, best_ratio_SVM, ensemble_pred_list

def main(_):
    y_eval, logits_list = eval()    



    # analyse result:
    each_result = len(logits_list) / 4
    i = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for rst in logits_list:
        lab = 0
        rst = rst.tolist()
        rst = rst.index(max(rst))
        if i < each_result:
            lab = 0
        elif i < each_result * 2:
            lab = 1
        elif i < each_result * 3:
            lab = 2
        else:
            lab = 3
        if lab == 3 and rst == 3:
            true_positive += 1
        elif lab != 3 and rst == 3:
            false_positive += 1
        elif lab == 3 and rst != 3:
            true_negative += 1
        i += 1

    with open("result", "a") as outfile:
        outfile.write("NN Result:\n")
        outfile.write("total number:" + str(len(logits_list)) + "\n")
        outfile.write("true_positive:" + str(true_positive) + "(" + str(round(true_positive/len(logits_list)*100, 2)) + "%)\n")
        outfile.write("false_positive:" + str(false_positive) + "(" + str(round(false_positive/len(logits_list)*100, 2)) + "%)\n")
        outfile.write("true_negative:" + str(true_negative) + "(" + str(round(true_negative/len(logits_list)*100, 2)) + "%)\n")
    outfile.close()


    print("NN result:")
    print("total number:" + str(len(logits_list)))
    print("true_positive:" + str(true_positive))
    print("false_positive:" + str(false_positive))
    print("true_negative:" + str(true_negative))



    # ensemble with SVM
    SVM_pred_list = SVM.SVM_eval()

    ratio_neural, ratio_SVM, ensemble_pred_list = change_ensemble_weight(logits_list, SVM_pred_list, y_eval)

    with open("runs/ratio", "w") as outfile:
        outfile.write(str(ratio_neural))
        outfile.write(" ")
        outfile.write(str(ratio_SVM))
    outfile.close()


if __name__ == "__main__":
    tf.app.run()