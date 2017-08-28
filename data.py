

def read_data(filepath):

    batch_size = 32
    sequece_length = 20

    with open(filepath, encoding='utf-8') as f:
    	data = f.read()

    words = list(set(data))
    words.sort()

    char_id_map = dict(enumerate(words))
    id_char_map = dict((v,k) for k,v in index_to_phoneme.items())

    train_dataset = []
    train_labels = []

    index = 0

    for i in range(batch_size):
        features = data[index:index + sequece_length]
        labels = data[index + 1:index + sequece_length + 1]
        index += sequece_length

        features = [char_id_map[word] for word in features]
        labels = [char_id_map[word] for word in labels]

        train_dataset.append(features)
        train_labels.append(labels)

    return train_dataset , train_labels



