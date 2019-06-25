
'''
#Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
**Notes**
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Bidirectional, Flatten, Dropout
from keras.datasets import imdb
import tensorflow as tf
import numpy as np

from ddrop.layers import DropConnect

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
# stop = set(stopwords.words('english'))



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





max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 10
batch_size = 32

print('Loading data...')
train, test = readheadlines()

x_train = train[0]
y_train = list(map(int,train[1]))
x_test = test[0]
y_test = list(map(int,test[1]))


max_document_length = max([len(x.split(" ")) for x in x_train])


train_text = x_train + x_test

tok = Tokenizer()
tok.fit_on_texts(train_text)
x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test)

y_train = np.asarray(y_train, dtype=np.int64)
y_test = np.asarray(y_test, dtype=np.int64)


'''
text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
x_train = np.array(list(text_vocab_processor.fit_transform(x_train)))

x_test = np.array(list(text_vocab_processor.fit_transform(x_test)))
'''

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)



print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_document_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_document_length)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# RNN-BiLSTM
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_document_length))

# bi-lstm
model.add(Bidirectional(LSTM(128)))
# normal lstm
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=1))

# dropout
model.add(Dropout(0.2))
# drop connect
model.add(DropConnect(Dense(4, activation='exponential'), prob=0.2))

# model.add(Dense(4, activation='exponential'))

# try using different optimizers and different optimizer configs
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# CNN
'''
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_document_length))
'''


print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# pred_possibilities = model.predict(x_test)
# print(pred_possibilities)

model.save('model.h5')
del model

model = load_model('model.h5', custom_objects={'DropConnect':DropConnect})


print('Predict result:')
pred_possibilities = model.predict(x_test)
print(pred_possibilities)