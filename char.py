import numpy as np
import random, os, sys

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM

from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file



class charRNN():

    def __init__(self, sample_length, vocabulary_size, hidden_states=128, learning_rate=0.01):
        self.model = Sequential()
        self.model.add(LSTM(hidden_states, input_length=sample_length, input_dim=vocabulary_size, return_sequences=True))
        self.model.add(LSTM(vocabulary_size, return_sequences=True))
        self.model.add(Activation('softmax'))
        self.optimizer = RMSprop(lr=learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer

    def update_parameter(self, lr_decrease=None):
        if lr_decrease is not None:
            new_lr = np.float32(self.model.optimizer.lr.get_value() * lr_decrease)
            self.model.optimizer.lr.set_value(new_lr)
            return new_lr



    def train(self, X, y, batch_size=128, epochs=1, verbose=1):

        try:
            return self.model.fit(X, y, batch_size=batch_size, nb_epoch=epochs, verbose=verbose)
        except Exception as err:
            print("Error training network: ", err)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("Error :", exc_type, fname, exc_tb.tb_lineno)

    def evaluate(self, X, y):

        try:
            return self.model.evaluate(X, y, verbose=0)
        except Exception as err:
            print("Error training network: ", err)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("Error :", exc_type, fname, exc_tb.tb_lineno)

    def predict(self, x):
        try:
            return self.model.predict(x, verbose=0)[0]
        except Exception as err:
            print("Error predicting: ", err)

    def save(self, file_name):
        try:
            self.model.save(file_name)
        except Exception as err:
            print("Error saving network: ", err)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("Error :", exc_type, fname, exc_tb.tb_lineno)


    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


class Text:

    def __init__(self):
        pass

    def load_char_level(self, path, verbose=False):
        text = open(path).read().lower()
        chars = sorted(list(set(text)))
        char_idx = dict((c, i) for i, c in enumerate(chars))
        ixd_char = dict((i, c) for i, c in enumerate(chars))
        if verbose:
            print('Total corpus length:', len(text))
            print('total chars:', len(chars))
        return text, char_idx, ixd_char

    def parse_dataset(self, text, parcel_size=40, step=3, output_level=False, verbose=False):
        if not output_level:
            output_level = parcel_size
        j = len(text) - parcel_size
        x_set = []
        y_set = []
        for i in range(0, j, step):
            x_set.append(text[i: i + parcel_size])
            y_set.append(text[(i + output_level):(i + 1 + parcel_size)])

            if verbose and (i %100 == 0):
                print("[x: '%s' - y: '%s']" % (x_set[-1], y_set[-1]))
        if verbose:
            print('Number of sequences:', len(x_set))
        return x_set, y_set

    def vectorization(self, v_set, char_idx):
        V = np.zeros((len(v_set), len(v_set[0]), len(char_idx)), dtype=np.bool)
        for i, x in enumerate(v_set):
            for t, char in enumerate(x):
                V[i, t, char_idx[char]] = 1
        return V
    def split_dataset(self, number_samples, test_percentage):
        indices = range(number_samples)
        random.shuffle(indices)
        tsh_id = number_samples - int(number_samples * test_percentage)
        train_index = np.array(indices[:tsh_id])
        test_index = np.array(indices[tsh_id:])
        return train_index, test_index

    def save_split(self, split_file_name, split_dataset):
        np.save(split_file_name, split_dataset)

    def load_split(self, split_file_name):
        return np.load(split_file_name)











