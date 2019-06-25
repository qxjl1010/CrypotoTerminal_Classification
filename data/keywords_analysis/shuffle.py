import random

li=list(range(3612*3))
random.shuffle(li)

sentence_list = []
label_list = []

with open("headlines_list_3classes") as infile:
    for line in infile:
        sentence_list.append(line)
infile.close()

with open("labels_list_3classes") as infile:
    for line in infile:
        label_list.append(line)
infile.close()

Z1 = [x for _,x in sorted(zip(li,sentence_list))]
Z2 = [x for _,x in sorted(zip(li,label_list))]

with open("headlines_list_3classes", "w") as outfile:
    for item in Z1:
        outfile.write(item)
outfile.close()

with open("labels_list_3classes", "w") as outfile:
    for item in Z2:
        outfile.write(item)
outfile.close()
