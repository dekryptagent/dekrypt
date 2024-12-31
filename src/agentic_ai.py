import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Dropout, BatchNormalization, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

class AgenticAI:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(10, 10, 1))
        x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Reshape((128, 1))(x)
        x = LSTM(64, return_sequences=False)(x)
        output = Dense(1)(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        return model

    def train(self, data, labels, epochs=10, batch_size=32):
        self.model.fit(data, labels, epochs=epochs, batch_size=batch_size)

    def predict(self, data):
        return self.model.predict(data)
