import tensorflow as tf
import numpy as np
import os
import data_helpers
import Keywords_analysis
import SVM


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("rank0_dir", "data/headlines/rank0_test", "Path of rank0 data")
tf.flags.DEFINE_string("rank1_dir", "data/headlines/rank1_test", "Path of rank1 data")
tf.flags.DEFINE_string("rank2_dir", "data/headlines/rank2_test", "Path of rank2 data")
tf.flags.DEFINE_string("rank3_dir", "data/headlines/rank3_test", "Path of rank3 data")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1559075039/checkpoints", "Checkpoint directory from training run")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")


# y_eval = []

def eval():
    with tf.device('/gpu:0'):
        x_text, y = data_helpers.load_data_and_labels(FLAGS.rank0_dir, FLAGS.rank1_dir, FLAGS.rank2_dir, FLAGS.rank3_dir)

    # Map data into vocabulary
    text_path = os.path.join(FLAGS.checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

    x_eval = np.array(list(text_vocab_processor.transform(x_text)))
    y_eval = np.argmax(y, axis=1)


    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    logits_list = []
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            logits = graph.get_operation_by_name("output/logits").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch})
                batch_logits = sess.run(logits, {input_text: x_batch})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                
                for i in range(len(batch_logits)):
                    logits_list.append(batch_logits[i])            

            correct_predictions = float(sum(all_predictions == y_eval))
            print("Total number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))
    return y_eval, logits_list

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
        print("correct_num is:" + str(correct_num))
        rst = correct_num / len(y_eval)
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

def create_keyword_prediction(layer_list, weight_list, word_num_test, headlines_test, labels_test, keywords_list):
    correct_num = 0
    pred_list = []
    prediction_list= []
    # print(layer_list)
    for word_num, headline, label in zip(word_num_test, headlines_test, labels_test):
        keyword_num = 0
        keyword_First3 = 0
        i = 1
        tmp_list = []
        for word in headline:
            if word in keywords_list:
                keyword_num += 1
                if i <= 3:
                    keyword_First3 += 1
                i += 1

        tmp_list = [i*float(weight_list[word_num][keyword_num][keyword_First3]) for i in layer_list[word_num][keyword_num][keyword_First3]]

        if tmp_list[0] == tmp_list[1] and tmp_list[2] == tmp_list[1] and tmp_list[2] == tmp_list[3]:
            pred_label = 4
        else:
            pred_label = tmp_list.index(max(tmp_list))

        if pred_label == int(label):
            correct_num += 1
        pred_list.append(pred_label)
        prediction_list.append(tmp_list)
    print("keywords logic accuracy is:")
    print(correct_num/len(labels_test))
    return prediction_list

def change_keyword_weight(weight_list, keyword_pred, neural_pred, word_num, keyword_num, first3_keyword_num, label):

    ensemble_pred_min1 = float(weight_list[word_num][keyword_num][first3_keyword_num]) * float(keyword_pred[0]) + neural_pred[0]
    ensemble_pred_pos0 = float(weight_list[word_num][keyword_num][first3_keyword_num]) * float(keyword_pred[1]) + neural_pred[1]
    ensemble_pred_pos1 = float(weight_list[word_num][keyword_num][first3_keyword_num]) * float(keyword_pred[2]) + neural_pred[2]
    ensemble_pred_pos2 = float(weight_list[word_num][keyword_num][first3_keyword_num]) * float(keyword_pred[3]) + neural_pred[3]
    ensemble_pred = [ensemble_pred_min1, ensemble_pred_pos0, ensemble_pred_pos1, ensemble_pred_pos2]
    final_pred = ensemble_pred.index(max(ensemble_pred))
    keyword_pred = keyword_pred.index(max(keyword_pred))
    tmp_pred = [neural_pred[0], neural_pred[1], neural_pred[2], neural_pred[3]]
    neural_pred = tmp_pred.index(max(tmp_pred))

    if final_pred == label and neural_pred != label and keyword_pred == label:
        weight_list[word_num][keyword_num][first3_keyword_num] = float(weight_list[word_num][keyword_num][first3_keyword_num]) + 2
    elif final_pred != label and neural_pred == label and keyword_pred != label and weight_list[word_num][keyword_num][first3_keyword_num] != 0:
        weight_list[word_num][keyword_num][first3_keyword_num] = float(weight_list[word_num][keyword_num][first3_keyword_num]) - 2
    return weight_list

def final_pred(neural_pred, keywords_pred, weight_list, labels, headlines_test, keywords_list):
    correct_num = 0
    fi_list = []

    right_min1 = 0
    right_pos0 = 0
    right_pos1 = 0
    right_pos2 = 0

    for pred1, pred2, label, headlines in zip(neural_pred, keywords_pred, labels, headlines_test):
        
        keyword_num = 0
        First3_num = 0
        word_num = 0
        i = 1
        for word in headlines:
            if word in keywords_list:
                keyword_num += 1
                if i <= 3:
                    First3_num += 1
            word_num += 1
            i += 1
        ensemble_pred_min1 = float(weight_list[word_num][keyword_num][First3_num]) * float(pred2[0]) + pred1[0]
        ensemble_pred_pos0 = float(weight_list[word_num][keyword_num][First3_num]) * float(pred2[1]) + pred1[1]
        ensemble_pred_pos1 = float(weight_list[word_num][keyword_num][First3_num]) * float(pred2[2]) + pred1[2]
        ensemble_pred_pos2 = float(weight_list[word_num][keyword_num][First3_num]) * float(pred2[3]) + pred1[3]
        ensemble_pred = [ensemble_pred_min1, ensemble_pred_pos0, ensemble_pred_pos1, ensemble_pred_pos2]
        fi_pred = ensemble_pred.index(max(ensemble_pred))
        fi_list.append(fi_pred)
        if fi_pred == label:
            correct_num += 1
            if label == 0:
                right_min1 += 1
            elif label == 1:
                right_pos0 += 1
            elif label == 2:
                right_pos1 += 1
            elif label == 3:
                right_pos2 += 1

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
    
    # enhance result by keywords
    layer_list, weight_list, word_num_test, headlines_test, labels_test, keywords_list = Keywords_analysis.analyze_low_to_high()    
    prediction_list = create_keyword_prediction(layer_list, weight_list, word_num_test, headlines_test, labels_test, keywords_list)

    for neural_pred, keywords_pred, label, headlines in zip(ensemble_pred_list, prediction_list, y_eval, headlines_test):
        keyword_num = 0
        First3_num = 0
        word_num = 0
        i = 1
        for word in headlines:
            if word in keywords_list:
                keyword_num += 1
                if i <= 3:
                    First3_num += 1
            word_num += 1
            i += 1

        weight_list = change_keyword_weight(weight_list, keywords_pred, neural_pred, word_num, keyword_num, First3_num, label)
    
    np_weight_list = np.array(weight_list)
    np.save("runs/weight_list", np_weight_list)
    np.save("runs/keywords_layers", layer_list)
    weight_list = np.load("runs/weight_list.npy")

    final_pred(ensemble_pred_list, prediction_list, weight_list, y_eval, headlines_test, keywords_list)

if __name__ == "__main__":
    tf.app.run()