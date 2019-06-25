with open("test_labels", "w") as infile:
    for i in range(200):
        infile.write("0\n")
    for i in range(200):
        infile.write("1\n")
    for i in range(200):
        infile.write("2\n")
    for i in range(200):
        infile.write("3\n")
infile.close()

with open("labels_list", "w") as infile:
    for i in range(2285):
        infile.write("0\n")
    for i in range(2285):
        infile.write("1\n")
    for i in range(2285):
        infile.write("2\n")
    for i in range(2285):
        infile.write("3\n")
infile.close()


