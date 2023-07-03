import os
import numpy as np
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self.build_model(num_layers, width)


    def build_model(self, num_layers, width):
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    # 單一狀態預測action
    def predict_one(self, state):
        state = np.reshape(state, [1, self._input_dim])
        return self.model.predict(state)

    # 多狀態預測action
    def predict_batch(self, states):
        return self.model.predict(states)

    # new_Q value訓練neural network
    def train_batch(self, states, q_sa):
        self.model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        self.model.save(os.path.join(path, 'trained_model.h5'))

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self.model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        state = np.reshape(state, [1, self._input_dim])
        return self.model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim