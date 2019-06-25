from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from pylab import *
import matplotlib.pyplot as plt

keywords_list = []
headlines_list = []
labels_list = []
word_num_list = []

headlines_test = []
labels_test = []
word_num_test = []

prediction_list = []

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop = set(stopwords.words('english'))

# new logic: tree structure according to 3 layers
# logic 1: from high to low(don't need balanced samples)
def analyze_high_to_low():
    layer_list = []
    for i in range(50):
        keyword_num_list = []
        for j in range(15):
            first3_list = []
            for k in range(4):
                label_list = []
                for l in range(4):
                    label_list.append("0")
                first3_list.append(label_list)
            keyword_num_list.append(first3_list)
        layer_list.append(keyword_num_list)

    min1_num = 0
    pos0_num = 0
    pos1_num = 0
    pos2_num = 0

    for word_num, headline, label in zip(word_num_list, headlines_list, labels_list):
        keyword_num = 0
        keyword_First3 = 0
        i = 1
        for word in headline:
            if word in keywords_list:
                keyword_num += 1
                if i <= 3:
                    keyword_First3 += 1
                i += 1
        
        if label == '-1':
            min1_num += 1
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][0])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][0] = tmp_i
        if label == '0':
            pos0_num += 1
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][1])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][1] = tmp_i
        if label == '1':
            pos1_num += 1
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][2])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][2] = tmp_i
        if label == '2':
            pos2_num += 1
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][3])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][3] = tmp_i

    for i in range(len(layer_list)):
        for j in range(len(layer_list[i])):
            for k in range(len(layer_list[i][j])):
                tmp_0 = int(layer_list[i][j][k][0])
                layer_list[i][j][k][0] = tmp_0 / min1_num
                tmp_1 = int(layer_list[i][j][k][1])
                layer_list[i][j][k][1] = tmp_1 / pos0_num
                tmp_2 = int(layer_list[i][j][k][2])
                layer_list[i][j][k][2] = tmp_2 / pos1_num
                tmp_3 = int(layer_list[i][j][k][3])
                layer_list[i][j][k][3] = tmp_3 / pos2_num

    # weight
    weight_list = []
    for i in range(50):
        tmp_weight1 = []
        for j in range(15):
            tmp_weight2 = []
            for k in range(4):
                tmp_weight2.append("2")
            tmp_weight1.append(tmp_weight2)
        weight_list.append(tmp_weight1)

    # print(weight_list)    
    # print(layer_list)
    return layer_list, weight_list


# logic 2: from low to up(need balanced samples)
def analyze_low_to_high():
    layer_list = []
    for i in range(50):
        keyword_num_list = []
        for j in range(15):
            first3_list = []
            for k in range(4):
                label_list = []
                for l in range(4):
                    label_list.append("0")
                first3_list.append(label_list)
            keyword_num_list.append(first3_list)
        layer_list.append(keyword_num_list)


    for word_num, headline, label in zip(word_num_list, headlines_list, labels_list):
        keyword_num = 0
        keyword_First3 = 0
        i = 1
        for word in headline:
            if word in keywords_list:
                keyword_num += 1
                if i <= 3:
                    keyword_First3 += 1
                i += 1
        
        if label == '-1':
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][0])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][0] = tmp_i
        if label == '0':
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][1])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][1] = tmp_i
        if label == '1':
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][2])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][2] = tmp_i
        if label == '2':
            tmp_i = int(layer_list[word_num-3][keyword_num][keyword_First3][3])
            tmp_i += 1
            layer_list[word_num-3][keyword_num][keyword_First3][3] = tmp_i

    for i in range(len(layer_list)):
        for j in range(len(layer_list[i])):
            for k in range(len(layer_list[i][j])):
                tmp_0 = int(layer_list[i][j][k][0])
                tmp_1 = int(layer_list[i][j][k][1])
                tmp_2 = int(layer_list[i][j][k][2])
                tmp_3 = int(layer_list[i][j][k][3])
                if tmp_0 + tmp_1 + tmp_2 + tmp_3 > 0:
                    layer_list[i][j][k][0] = tmp_0 / (tmp_0 + tmp_1 + tmp_2 + tmp_3)
                    layer_list[i][j][k][1] = tmp_1 / (tmp_0 + tmp_1 + tmp_2 + tmp_3)
                    layer_list[i][j][k][2] = tmp_2 / (tmp_0 + tmp_1 + tmp_2 + tmp_3)
                    layer_list[i][j][k][3] = tmp_3 / (tmp_0 + tmp_1 + tmp_2 + tmp_3)
                else:
                    layer_list[i][j][k][0] = tmp_0
                    layer_list[i][j][k][1] = tmp_1
                    layer_list[i][j][k][2] = tmp_2
                    layer_list[i][j][k][3] = tmp_3
    
    # weight
    weight_list = []
    for i in range(50):
        tmp_weight1 = []
        for j in range(15):
            tmp_weight2 = []
            for k in range(4):
                tmp_weight2.append("1")
            tmp_weight1.append(tmp_weight2)
        weight_list.append(tmp_weight1)

    # print(layer_list)
    return layer_list, weight_list, word_num_test, headlines_test, labels_test, keywords_list

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

with open("data/keywords_analysis/headlines_list") as headlines_infile, open("data/keywords_analysis/labels_list") as labels_infile:
    for headlines, labels in zip(headlines_infile, labels_infile):
        words = headlines.split()        
        if len(words) <= 4:
            continue
        tmp_list = []
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
            tmp_list.append(clean_word)
        headlines_list.append(tmp_list)
        labels_list.append(labels[:-1])
        word_num_list.append(len(tmp_list))
headlines_infile.close()
labels_infile.close()

# layer_list, weight_list = analyze_low_to_high()


with open("data/keywords_analysis/test_headlines") as headlines_infile, open("data/keywords_analysis/test_labels") as labels_infile:
    for headlines, labels in zip(headlines_infile, labels_infile):
        words = headlines.split()
        if len(words) <= 4:
            continue
        tmp_list = []
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
            tmp_list.append(clean_word)
        headlines_test.append(tmp_list)
        labels_test.append(str(int(labels[:-1]) + 1))
        word_num_test.append(len(tmp_list))
headlines_infile.close()
labels_infile.close()
# create_prediction()
        


# old logic: analysis according to label
'''
keyword_num_list_min1 = {}
keyword_num_list_pos0 = {}
keyword_num_list_pos1 = {}
keyword_num_list_pos2 = {}
keyword_inFirst3_min1 = {}
keyword_inFirst3_pos0 = {}
keyword_inFirst3_pos1 = {}
keyword_inFirst3_pos2 = {}
word_num_dic_min1 = {}
word_num_dic_0 = {}
word_num_dic_1 = {}
word_num_dic_2 = {}
for i in range(30):
    word_num_dic_min1[i+3] = 0
    word_num_dic_0[i+3] = 0
    word_num_dic_1[i+3] = 0
    word_num_dic_2[i+3] = 0

for i in range(15):
    keyword_num_list_min1[i] = 0
    keyword_num_list_pos0[i] = 0
    keyword_num_list_pos1[i] = 0
    keyword_num_list_pos2[i] = 0


for i in range(15):
    keyword_inFirst3_min1[i] = 0
    keyword_inFirst3_pos0[i] = 0
    keyword_inFirst3_pos1[i] = 0
    keyword_inFirst3_pos2[i] = 0
    

for headline, label, word_num in zip(headlines_list, labels_list, word_num_list):
    keyword_num = 0
    keyword_First3 = 0
    i = 1
    for word in headline:
        if word in keywords_list:
            keyword_num += 1
            if i <= 3:
                keyword_First3 += 1
            i += 1


    if label == '-1':
        word_num_dic_min1[word_num] += 1
        keyword_num_list_min1[keyword_num] += 1
        keyword_inFirst3_min1[keyword_First3] += 1
    elif label == '0':
        word_num_dic_0[word_num] += 1
        keyword_num_list_pos0[keyword_num] += 1
        keyword_inFirst3_pos0[keyword_First3] += 1
    elif label == '1':
        word_num_dic_1[word_num] += 1
        keyword_num_list_pos1[keyword_num] += 1
        keyword_inFirst3_pos1[keyword_First3] += 1
    else:
        word_num_dic_2[word_num] += 1
        keyword_num_list_pos2[keyword_num] += 1
        keyword_inFirst3_pos2[keyword_First3] += 1    

min1_num = 0
pos0_num = 0
pos1_num = 0
pos2_num = 0

for i in range(30):
    min1_num += word_num_dic_min1[i+3]
    pos0_num += word_num_dic_0[i+3]
    pos1_num += word_num_dic_1[i+3]
    pos2_num += word_num_dic_2[i+3]

for i in range(30):
    word_num_dic_min1[i+3] /= min1_num
    word_num_dic_0[i+3] /= pos0_num
    word_num_dic_1[i+3] /= pos1_num
    word_num_dic_2[i+3] /= pos2_num

for i in range(15):
    keyword_num_list_min1[i] /= min1_num
    keyword_num_list_pos0[i] /= pos0_num
    keyword_num_list_pos1[i] /= pos1_num
    keyword_num_list_pos2[i] /= pos2_num

for i in range(15):
    keyword_inFirst3_min1[i] /= min1_num
    keyword_inFirst3_pos0[i] /= pos0_num
    keyword_inFirst3_pos1[i] /= pos1_num
    keyword_inFirst3_pos2[i] /= pos2_num

min1_x = []
min1_y = []
pos0_x = []
pos0_y = []
pos1_x = []
pos1_y = []
pos2_x = []
pos2_y = []
for i in range(30):
    min1_x.append(i+3)
    min1_y.append(word_num_dic_min1[i+3])
    pos0_x.append(i+3)
    pos0_y.append(word_num_dic_0[i+3])
    pos1_x.append(i+3)
    pos1_y.append(word_num_dic_1[i+3])
    pos2_x.append(i+3)
    pos2_y.append(word_num_dic_2[i+3])

print(min1_y)
print(pos0_y)
print(pos1_y)
print(pos2_y)

keyword_min1_x = []
keyword_min1_y = []
keyword_pos0_x = []
keyword_pos0_y = []
keyword_pos1_x = []
keyword_pos1_y = []
keyword_pos2_x = []
keyword_pos2_y = []
for i in range(15):
    keyword_min1_x.append(i)
    keyword_min1_y.append(keyword_num_list_min1[i])
    keyword_pos0_x.append(i)
    keyword_pos0_y.append(keyword_num_list_pos0[i])
    keyword_pos1_x.append(i)
    keyword_pos1_y.append(keyword_num_list_pos1[i])
    keyword_pos2_x.append(i)
    keyword_pos2_y.append(keyword_num_list_pos2[i])

keyword_First3_min1_x = []
keyword_First3_min1_y = []
keyword_First3_pos0_x = []
keyword_First3_pos0_y = []
keyword_First3_pos1_x = []
keyword_First3_pos1_y = []
keyword_First3_pos2_x = []
keyword_First3_pos2_y = []
for i in range(15):
    keyword_First3_min1_x.append(i)
    keyword_First3_min1_y.append(keyword_inFirst3_min1[i])
    keyword_First3_pos0_x.append(i)
    keyword_First3_pos0_y.append(keyword_inFirst3_pos0[i])
    keyword_First3_pos1_x.append(i)
    keyword_First3_pos1_y.append(keyword_inFirst3_pos1[i])
    keyword_First3_pos2_x.append(i)
    keyword_First3_pos2_y.append(keyword_inFirst3_pos2[i])
'''



# paint result
'''
figure(1)
subplot(221)
plot(min1_x,min1_y,'r')
xlabel('length of headline')
ylabel('word numbers')
title('rating = -1')

subplot(222)
plot(pos0_x,pos0_y,'r')
xlabel('length of headline')
ylabel('word numbers')
title('rating = 0')

subplot(223)
plot(pos1_x,pos1_y,'r')
xlabel('length of headline')
ylabel('word numbers')
title('rating = 1')

subplot(224)
plot(pos2_x,pos2_y,'r')
xlabel('length of headline')
ylabel('word numbers')
title('rating = 2')
show()

figure(2)
subplot(221)
plot(keyword_min1_x,keyword_min1_y,'r')
xlabel('keyword numbers')
ylabel('word numbers')
title('rating = -1')

subplot(222)
plot(keyword_pos0_x,keyword_pos0_y,'r')
xlabel('keyword numbers')
ylabel('word numbers')
title('rating = 0')

subplot(223)
plot(keyword_pos1_x,keyword_pos1_y,'r')
xlabel('keyword numbers')
ylabel('word numbers')
title('rating = 1')

subplot(224)
plot(keyword_pos2_x,keyword_pos2_y,'r')
xlabel('keyword numbers')
ylabel('word numbers')
title('rating = 2')
show()

x = list(range(len(keyword_min1_x)))
plt.bar(x, keyword_min1_y, width=0.2, label='rating = -1', fc='black')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, keyword_pos0_y, width=0.2, label='rating = 0', fc='green')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, keyword_pos1_y, width=0.2, label='rating = 1', fc='orange')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, keyword_pos2_y, width=0.2, label='rating = 2', fc='r')
plt.legend()
plt.show()


x = list(range(len(min1_x)))
plt.bar(x, min1_y, width=0.2, label='rating = -1', fc='black')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, pos0_y, width=0.2, label='rating = 0', fc='green')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, pos1_y, width=0.2, label='rating = 1', fc='orange')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, pos2_y, width=0.2, label='rating = 2', fc='r')
plt.legend()
plt.show()

x = list(range(len(keyword_First3_min1_x)))
plt.bar(x, keyword_First3_min1_y, width=0.2, label='rating = -1', fc='black')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, keyword_First3_pos0_y, width=0.2, label='rating = 0', fc='green')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, keyword_First3_pos1_y, width=0.2, label='rating = 1', fc='orange')

for i in range(len(x)):
    x[i] = x[i] + 0.2

plt.bar(x, keyword_First3_pos2_y, width=0.2, label='rating = 2', fc='r')
plt.legend()
plt.show()
'''
