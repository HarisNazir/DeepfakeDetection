from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import LSTM, Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)
        
class RNN(Classifier):
    def __init__(self, time_steps, num_features, num_classes):
        super().__init__()
        inputs = Input(shape=(time_steps, num_features))
        lstm = LSTM(128, return_sequences=True)(inputs)
        lstm = Dropout(0.5)(lstm)
        lstm = LSTM(128, return_sequences=False)(lstm)
        lstm = Dropout(0.5)(lstm)
        outputs = Dense(num_classes, activation='softmax')(lstm)
        self.model = KerasModel(inputs, outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])