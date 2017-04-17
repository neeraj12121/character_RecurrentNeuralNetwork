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


