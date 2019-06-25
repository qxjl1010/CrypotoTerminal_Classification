rank0_list = []
with open("rank0.txt", "r") as infile:
    for line in infile:
        if len(line) > 1:
            rank0_list.append(line)
infile.close()

with open("rank0.txt", "w") as outfile:
    for line in rank0_list:
        outfile.write(line)
outfile.close()

rank1_list = []
with open("rank1.txt", "r") as infile:
    for line in infile:
        if len(line) > 1:
            rank1_list.append(line)
infile.close()

with open("rank1.txt", "w") as outfile:
    for line in rank1_list:
        outfile.write(line)
outfile.close()

rank2_list = []
with open("rank2.txt", "r") as infile:
    for line in infile:
        if len(line) > 1:
            rank2_list.append(line)
infile.close()

with open("rank2.txt", "w") as outfile:
    for line in rank2_list:
        outfile.write(line)
outfile.close()

rank3_list = []
with open("rank3.txt", "r") as infile:
    for line in infile:
        if len(line) > 1:
            rank3_list.append(line)
infile.close()

with open("rank3.txt", "w") as outfile:
    for line in rank3_list:
        outfile.write(line)
outfile.close()
