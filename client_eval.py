import tensorflow as tf
import numpy as np
import os

import sys
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import numbers

import threading

from sklearn.externals import joblib

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import time
from datetime import datetime

import socketserver
import collections
# import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

# Parameters
# ==================================================
# Eval Parameters



# used on server
tf.flags.DEFINE_string("checkpoint_dir", "/home/jialong/chris-ai/runs/1550514222/checkpoints", "Checkpoint directory from training run")

# used on local
# tf.flags.DEFINE_string("checkpoint_dir", "/home/jialong/lstm_version4/runs/1550514222/checkpoints", "Checkpoint directory from training run")



# tf.flags.DEFINE_string("headline", "", "input headline")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))




# Map data into vocabulary

# used on server
text_path = "/home/jialong/chris-ai/runs/1550514222/text_vocab"

# used on local
# text_path = "/home/jialong/lstm_version4/runs/1550514222/text_vocab"


#text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

channel = grpc.insecure_channel('0.0.0.0:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    

request = predict_pb2.PredictRequest()
request.model_spec.name = 'lstm'



class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = 400
    self._concurrency = 1
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.
  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()

      response = np.array(result_future.result().outputs['classes'].int64_val)
      scores = np.array(result_future.result().outputs['scores'].float_val)
      '''
      print("Prediction: ")
      print(scores[0:4])      
      for r,l in zip(response, label):
        if r != l:
          result_counter.inc_error()
      '''
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback

def final_pred(ensemble_pred, keywords_layers, weight_list, headlines):
    try:
        #print('enter final_pred')

        keywords_list = []
        # used on server
        with open("/home/jialong/chris-ai/data/keywords_analysis/keywords_list") as infile:
            # used on local
            # with open("/home/jialong/lstm_version4/data/keywords_analysis/keywords_list") as infile:
            for line in infile:
                words = line.split(" ")
                for word in words:
                    word = word.replace(',','')
                    word = word.replace('\n','')
                    if word != '':
                        
                        try :
                            word = lemmatizer.lemmatize(word)
                            # word = stemmer.stem(word)
                            keywords_list.append(word)
                        except Exception as e :
                            print( e )
            infile.close()

            
        keyword_num = 0
        First3_num = 0
        word_num = 0
        i = 1
        words = headlines.split()

        clean_headline = ""

        at_tag = False
        hash_tag = False
        double_tag = False
        for word in words:
            if word[0] == '@':
                at_tag = True
                continue
            if word[0] == '#':
                hash_tag = True
                continue
            word = word.replace(',','')
            word = word.replace('?','')
            word = word.replace('.','')
            if word != '':
                clean_word = lemmatizer.lemmatize(word)
                # clean_word = stemmer.stem(clean_word)
            if clean_word.lower() in stop:
                continue
            # print("left word is:")
            # print(word)
            if clean_word in keywords_list:
                keyword_num += 1
                if i <= 3:
                    First3_num += 1
            clean_headline = clean_headline + clean_word + " "
            word_num += 1
            i += 1
        if at_tag == True and hash_tag == True:
            double_tag = True
        clean_headline = clean_headline[:-1]
        
        enhance_pred_min1 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][0]) + ensemble_pred[0]
        enhance_pred_pos0 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][1]) + ensemble_pred[1]
        enhance_pred_pos1 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][2]) + ensemble_pred[2]
        enhance_pred_pos2 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][3]) + ensemble_pred[3]
        '''
        print("enhance_pred_min1 is:")
        print(enhance_pred_min1)
        print("enhance_pred_pos0 is:")
        print(enhance_pred_pos0)
        print("enhance_pred_pos1 is:")
        print(enhance_pred_pos1)
        print("enhance_pred_pos2 is:")
        print(enhance_pred_pos2)
        print("clean_headline is:")
        print(clean_headline)
        '''
        # last filter:
        # 1. price analysis headlines down weight:
        price_related_list = ["analysis price", "price change", "technical analysis", "price analysis", "price: usd"]
        if any(element in clean_headline for element in price_related_list):
            
            enhance_pred_pos2 = 0
            enhance_pred_pos1 = 0
            enhance_pred_pos0 = 0
            enhance_pred_min1 = 1
        
        # 2. event/conference headlines down weight:
        if "event" in clean_headline or "conference" in clean_headline:
            
            enhance_pred_pos2 *= 0.6
            enhance_pred_pos1 *= 0.7
            enhance_pred_pos0 *= 1.3
            enhance_pred_min1 *= 1.4
        
        # 3. double tag down weight:
        if double_tag == True:
            
            enhance_pred_pos2 *= 0 
            enhance_pred_pos1 *= 0
            enhance_pred_pos0 *= 0
            enhance_pred_min1 *= 1

        # 4. chart headlines down weight:
        if "chart" in clean_headline:
            
            enhance_pred_pos2 *= 0.6
            enhance_pred_pos1 *= 0.7
            enhance_pred_pos0 *= 1.3
            enhance_pred_min1 *= 1.4

        # 5. bullish/bearish headlines down weight:
        if "bullish" in clean_headline or "bearish" in clean_headline:
            
            enhance_pred_pos2 *= 0.6
            enhance_pred_pos1 *= 0.7
            enhance_pred_pos0 *= 1.3
            enhance_pred_min1 *= 1.4
        
        # 6. a word appears 3 times at the same time: down weight:
        clean_words = clean_headline.split()
        m = collections.Counter(clean_words)
        for w in clean_words:
            if m[w] >= 3:                
                enhance_pred_pos2 *= 0.6
                enhance_pred_pos1 *= 0.7
                enhance_pred_pos0 *= 1.3
                enhance_pred_min1 *= 1.4
                break

        # 7. important words:
        important_list = ["relase"]
        if any(element in clean_headline for element in important_list):
            
            enhance_pred_pos2 = 1
            enhance_pred_pos1 = 0
            enhance_pred_pos0 = 0
            enhance_pred_min1 = 0

        enhance_pred = [enhance_pred_min1, enhance_pred_pos0, enhance_pred_pos1, enhance_pred_pos2]
        '''
        print("enhance_pred_min1 is:")
        print(enhance_pred_min1)
        print("enhance_pred_pos0 is:")
        print(enhance_pred_pos0)
        print("enhance_pred_pos1 is:")
        print(enhance_pred_pos1)
        print("enhance_pred_pos2 is:")
        print(enhance_pred_pos2)
        '''
        fi_pred = enhance_pred.index(max(enhance_pred))
        '''
        print('final prediction is:')
        print(fi_pred)
        '''
        return fi_pred

    except Exception as inst:
        '''
        with open("error.log", 'a') as outfile:
            # print(type(inst))    # the exception instance
            # print(inst.args)     # arguments stored in .args
            # print(inst)  
            #outfile.write(inst.args)
            #outfile.write(inst)
        outfile.close()
        '''

class ThreadedTCPRequestHandler( socketserver.BaseRequestHandler ) :
    def handle( self ) :
        while 1==1:
            try :
                self.data = self.request.recv( 10240 ).strip()

                result_counter = _ResultCounter(400, 1)
        
                headline = str(self.data)
                '''
                print("start time:")
                print(datetime.utcnow())
                '''
                headline = headline.lower()
                headline = headline[2:-1]
                words = headline.split()
                print("headline is:")
                print(headline)
                if len(words) < 3:
                    print("short headline!!!!!")
                    return
                x_text = [headline, "PADDING", "PADDING", "PADDING"]

                x_eval = np.array(list(text_vocab_processor.transform(x_text)))
                y_eval = np.argmax([0,0,0,0])


                x_eval = x_eval.astype(np.int32)
                request.inputs['inputs'].CopyFrom(
                    tf.contrib.util.make_tensor_proto(x_eval)
                )

                result_counter.throttle()

                result_future = stub.Predict.future(request, 5.0)

                result_future.add_done_callback(_create_rpc_callback(y_eval, result_counter))
                print("before class prediction")
                # class prediction
                # response = np.array(result_future.result().outputs['classes'].int64_val)
                # score prediction
                '''
                print("00000000000000000000000")
                print(str(self.data))
                print("11111111111111111111111")
                print(result_future)
                print("22222222222222222222222")
                print(result_future.result())
                print("33333333333333333333333")
                print(result_future.result().outputs['scores'])
                print("44444444444444444444444")
                print(result_future.result().outputs['scores'].float_val)
                print("55555555555555555555555")
                '''
                
                response2 = np.array(result_future.result().outputs['scores'].float_val)
                '''
                print("response2 is:")
                print(response2[0:4])
                '''
                
                # used on server
                clf = joblib.load('/home/jialong/chris-ai/runs/SVM_model')

                # used on local
                # clf = joblib.load('/home/jialong/lstm_version4/runs/SVM_model')
                SVM_pred = clf.predict_proba([headline])
                ratio_neural = 0
                ratio_SVM = 0
                # print("before ratio stuff")


                # used on local
                #with open("/home/jialong/lstm_version4/runs/ratio") as infile:
                # used on server
                with open("/home/jialong/chris-ai/runs/ratio") as infile:

                    for line in infile:
                        ratios = line.split()
                        ratio_neural = ratios[0]
                        ratio_SVM = ratios[1]
                '''
                print("SVM_pred is:")
                print(SVM_pred)
                '''


                

                ensemble_pred = [response2[0] * float(ratio_neural) + SVM_pred[0][0] * float(ratio_SVM), response2[1] * float(ratio_neural) + SVM_pred[0][1] * float(ratio_SVM), response2[2] * float(ratio_neural) + SVM_pred[0][2] * float(ratio_SVM), response2[3] * float(ratio_neural) + SVM_pred[0][3] * float(ratio_SVM)]

                # used on server
                weight_list = np.load("/home/jialong/chris-ai/runs/weight_list.npy")
                keywords_layers = np.load("/home/jialong/chris-ai/runs/keywords_layers.npy")

                # used on local
                # weight_list = np.load("/home/jialong/lstm_version4/runs/weight_list.npy")
                # keywords_layers = np.load("/home/jialong/lstm_version4/runs/keywords_layers.npy")
                '''

                print("before final_pred")
                print("ensemble_pred is:")
                print(ensemble_pred)
                '''
                fi_pred = final_pred(ensemble_pred, keywords_layers, weight_list, headline)

                fi_pred = str(fi_pred)
                # print(fi_pred)
                fi_pred = bytes(fi_pred, encoding="utf8")
                '''
                print("end time:")
                print(datetime.utcnow())
                '''
                # self.request.send(str(1))
                self.request.send( fi_pred )
            except Exception as inst:
                # print(str(self.data))
                '''
                with open("error.log", 'a') as outfile:
                    # print(type(inst))    # the exception instance
                    # print(inst.args)     # arguments stored in .args
                    # print(inst)  
                    # outfile.write(inst.args)
                    # outfile.write(inst)
                outfile.close()
                '''


class ThreadedTCPServer( socketserver.ThreadingMixIn, socketserver.TCPServer ) :
    pass



def main(_):
    try:
        # listen to TCP message
        HOST = ""
        PORT = 42526
        server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
        server.serve_forever()
        server_thread = threading.Thread( target=server.serve_forever )
        server_thread.setDaemon( True )
        server_thread.start()
        while True :
            pass
    
    except Exception as inst:
        # print(str(self.data))
        '''
        with open("error.log", 'a') as outfile:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)  
        '''
    return


if __name__ == "__main__":
    tf.app.run()
