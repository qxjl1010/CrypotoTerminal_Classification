import tensorflow as tf
import numpy as np
import os
import data_helpers

from sklearn.externals import joblib
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1542052783/checkpoints", "checkpoint directory")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")

tf.flags.DEFINE_string("headline", "", "input headline")

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))

def final_pred(ensemble_pred, keywords_layers, weight_list, headlines):
    keywords_list = []

    with open("data/keywords_analysis/keywords_list") as infile:
        for line in infile:
            words = line.split( " ")
            for word in words:
                word = word.replace(',','')
                word = word.replace('\n','')
                if word != '':
                    clean_word = lemmatizer.lemmatize(word)
                    clean_word = stemmer.stem(clean_word)
                    keywords_list.append(clean_word)
        infile.close()

        
    keyword_num = 0
    First3_num = 0
    word_num = 0
    i = 1

    words = headlines.split()
    for word in words:
        if word[0] == '@' or word[0] == '#':
            continue
        word = word.replace(',','')
        word = word.replace('?','')
        word = word.replace('.','')
        if word != '':
            clean_word = lemmatizer.lemmatize(word)
            clean_word = stemmer.stem(clean_word)
        if clean_word.lower() in stop:
            continue
        # print("left word is:")
        # print(word)
        if clean_word in keywords_list:
            keyword_num += 1
            if i <= 3:
                First3_num += 1
        word_num += 1
        i += 1
    enhance_pred_min1 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][0]) + ensemble_pred[0]
    enhance_pred_pos0 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][1]) + ensemble_pred[1]
    enhance_pred_pos1 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][2]) + ensemble_pred[2]
    enhance_pred_pos2 = float(weight_list[word_num][keyword_num][First3_num]) * float(keywords_layers[word_num][keyword_num][First3_num][3]) + ensemble_pred[3]
    enhance_pred = [enhance_pred_min1, enhance_pred_pos0, enhance_pred_pos1, enhance_pred_pos2]
    fi_pred = enhance_pred.index(max(enhance_pred))
    
    print("ensemble_pred is:")
    print(ensemble_pred)
    print("keywords_layers is:")
    print(keywords_layers[word_num][keyword_num][First3_num])
    print("enhance_pred_min1 is:")
    print(enhance_pred_min1)
    print("enhance_pred_pos0 is:")
    print(enhance_pred_pos0)
    print("enhance_pred_pos1 is:")
    print(enhance_pred_pos1)
    print("enhance_pred_pos2 is:")
    print(enhance_pred_pos2)
    
    print("==============================================")
    print("final_predictions is:")
    print(fi_pred)
    print("==============================================")
    


with tf.device('/gpu:0'):
    x_test = [FLAGS.headline, "PADDING", "PADDING", "PADDING"]

    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_test)))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logits_list = []
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_text = graph.get_operation_by_name("input_text").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            logits = graph.get_operation_by_name("output/logits").outputs[0]

            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)
            
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch, dropout_keep_prob: FLAGS.dropout_keep_prob})
                batch_logits = sess.run(logits, {input_text: x_batch, dropout_keep_prob: FLAGS.dropout_keep_prob})
                
                for i in range(len(batch_logits)):
                    logits_list.append(batch_logits[i])
    neural_pred = logits_list[0]

    '''
    print("neural_pred is:")
    print(neural_pred)
    '''
    clf = joblib.load('runs/SVM_model')
    SVM_pred = clf.predict_proba([FLAGS.headline])
    '''
    print("SVM_pred is:")
    print(SVM_pred)
    '''
    ratio_neural = 0
    ratio_SVM = 0
    with open("runs/ratio") as infile:
        for line in infile:
            ratios = line.split()
            ratio_neural = ratios[0]
            ratio_SVM = ratios[1]

    ensemble_pred = [neural_pred[0] * float(ratio_neural) + SVM_pred[0][0] * float(ratio_SVM), neural_pred[1] * float(ratio_neural) + SVM_pred[0][1] * float(ratio_SVM), neural_pred[2] * float(ratio_neural) + SVM_pred[0][2] * float(ratio_SVM), neural_pred[3] * float(ratio_neural) + SVM_pred[0][3] * float(ratio_SVM)]

    weight_list = np.load("runs/weight_list.npy")
    keywords_layers = np.load("runs/keywords_layers.npy")

    final_pred(ensemble_pred, keywords_layers, weight_list, FLAGS.headline)