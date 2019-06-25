import numpy as np
import re


import mysql.connector



def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# train our word embedding, not using word2vec now
def load_embedding_vectors_word2vec(vocabulary, filename):
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))

        binary_len = np.dtype('float32').itemsize * vector_size
        for line_no in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'':
                    raise EOFError("unexpected end of input, is count incorrect or file other wise damaged?")
                if ch != b'\n':
                    word.append(ch)
            word = str(b''.join(word), encoding=encoding, errors='strict')
            idx = vocabulary.get(word)
            if idx != 0:
                embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.seek(binary_len, 1)
        f.close()
        return embedding_vectors


def load_data_and_labels(rank0_data_file, rank1_data_file, rank2_data_file, rank3_data_file):


    # connect to DB
    mysql_user = 'root'
    mysql_password = 'Bitcoin2018!'
    mysql_database = 'cryptote_db2'
    mysql_host = '35.203.23.69'

    cnx = mysql.connector.connect(user = mysql_user, password = mysql_password, host = mysql_host, database = mysql_database )
    crsr = cnx.cursor(buffered=True)
    print("connect to DB successful!")
    # get placeholder list
    command = "SELECT name,context_name FROM cryptote_db2.instrument_groups"
    crsr.execute(command)
    rows = crsr.fetchall()

    placeholder_list = []
    print("creating placeolder list...")
    for row in rows:
        if row[0] not in placeholder_list:
            placeholder_list.append(row[0])
        if row[1] not in placeholder_list:
            placeholder_list.append(row[1])
    print("loading data...")
    # Load data from files
    rank0_examples = list(open(rank0_data_file, "r", encoding="UTF8").readlines())

    clean0 = []

    for line in rank0_examples:
        words_list = ''
        line = line.split()
        for word in line:
            if 'http' in word:
                continue
            if word in placeholder_list:
                words_list = words_list + 'instrument '
            else:
                words_list = words_list + word + ' '
        clean0.append(words_list[:-1])
        

    rank0_examples = [s.strip() for s in clean0]



    rank1_examples = list(open(rank1_data_file, "r", encoding="UTF8").readlines())


    clean1 = []

    for line in rank1_examples:
        words_list = ''
        line = line.split()
        for word in line:
            if 'http' in word:
                continue
            if word in placeholder_list:
                words_list = words_list + 'instrument '
            else:
                words_list = words_list + word + ' '
        clean1.append(words_list[:-1])


    rank1_examples = [s.strip() for s in clean1]
    rank2_examples = list(open(rank2_data_file, "r", encoding="UTF8").readlines())


    clean2 = []

    for line in rank2_examples:
        words_list = ''
        line = line.split()
        for word in line:
            if 'http' in word:
                continue
            if word in placeholder_list:
                words_list = words_list + 'instrument '
            else:
                words_list = words_list + word + ' '
        clean2.append(words_list[:-1])


    rank2_examples = [s.strip() for s in clean2]
    rank3_examples = list(open(rank3_data_file, "r", encoding="UTF8").readlines())


    clean3 = []

    for line in rank3_examples:
        words_list = ''
        line = line.split()
        for word in line:
            if 'http' in word:
                continue
            if word in placeholder_list:
                words_list = words_list + 'instrument '
            else:
                words_list = words_list + word + ' '
        clean3.append(words_list[:-1])



    rank3_examples = [s.strip() for s in clean3]
    print("data loaded!")
    # Split by words
    x_text = rank0_examples + rank1_examples + rank2_examples + rank3_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    rank0_labels = [[1, 0, 0, 0] for _ in rank0_examples]
    rank1_labels = [[0, 1, 0, 0] for _ in rank1_examples]
    rank2_labels = [[0, 0, 1, 0] for _ in rank2_examples]
    rank3_labels = [[0, 0, 0, 1] for _ in rank3_examples]
    y = np.concatenate([rank0_labels, rank1_labels, rank2_labels, rank3_labels], 0)
    return [x_text, y]


def load_data_and_labels2(rank1_data_file, rank2_data_file, rank3_data_file):
    # Load data from files
    rank1_examples = list(open(rank1_data_file, "r", encoding="UTF8").readlines())
    rank1_examples = [s.strip() for s in rank1_examples]
    rank2_examples = list(open(rank2_data_file, "r", encoding="UTF8").readlines())
    rank2_examples = [s.strip() for s in rank2_examples]
    rank3_examples = list(open(rank3_data_file, "r", encoding="UTF8").readlines())
    rank3_examples = [s.strip() for s in rank3_examples]
    # Split by words
    x_text = rank1_examples + rank2_examples + rank3_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    rank1_labels = [[1, 0, 0] for _ in rank1_examples]
    rank2_labels = [[0, 1, 0] for _ in rank2_examples]
    rank3_labels = [[0, 0, 1] for _ in rank3_examples]
    y = np.concatenate([rank1_labels, rank2_labels, rank3_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    rank0_dir = "data/headlines/rank0"
    rank1_dir = "data/headlines/rank1"
    rank2_dir = "data/headlines/rank2"
    rank3_dir = "data/headlines/rank3"

    load_data_and_labels(rank0_dir, rank1_dir, rank2_dir, rank3_dir)
