from typing import List
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from random import randint, uniform


GenReturn = tuple(list())

class GeneticNeuralNet(tf.keras.Sequential):
    def __init__(self, is_child: bool, hidden_layer_weights: List, output_weights: List, x_train, y_train, epochs: int = 10) -> None:
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs

        if is_child:
            self.add(tf.keras.layers.Dense(12, weights=[hidden_layer_weights[0], np.zeros(12)], activation='sigmoid'))
            self.add(tf.keras.layers.Dense(6, weights=[hidden_layer_weights[1], np.zeros(6)],activation='sigmoid'))
            self.add(tf.keras.layers.Dense(1, weights=[output_weights, np.zeros(1)],activation='sigmoid'))
        else:
            self.add(tf.keras.layers.Dense(12, activation='sigmoid'))
            self.add(tf.keras.layers.Dense(6, activation='sigmoid'))
            self.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    def forward_prop(self) -> float:
        y_hat = self.predict(self.x_train.values)
        self.fitness_value = accuracy_score(self.y_train.values, y_hat.round())
        return self.fitness_value
    
    def train(self) -> None:
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.fit(self.x_train.values, self.y_train.values, epochs=self.epochs)
    
    def accuracy(self, x_test, y_test) -> float:
        y_hat = self.predict(x_test.values)
        acc = accuracy_score(y_test.values, y_hat.round())
        return acc


def mutation(weights):
    mutated_layer = randint(0, len(weights) - 1)
    mutated_probability = uniform(0, 1)
    if mutated_probability >= .5:
        weights[mutated_layer] *= randint(2, 5)
    else:
        pass

def crossover(parent_1, parent_2, x_train, y_train):
    p1_weights = []
    p2_weights = []
    result_weights = []

    for layer in parent_1.layers:
        p1_weights.append(layer.get_weights()[0])

    for layer in parent_2.layers:
        p2_weights.append(layer.get_weights()[0])

    layer_size = len(p1_weights)

    for i in range(0, layer_size):
        split = randint(0, np.shape(p1_weights[i])[1]-1)

        for j in range(split, np.shape(p1_weights[i])[1]-1):
            p1_weights[i][:, j] = p2_weights[i][:, j]

        result_weights.append(p1_weights[i])

    mutation(result_weights)

    child = GeneticNeuralNet(is_child=True, hidden_layer_weights=result_weights[:-1], 
                                output_weights=result_weights[-1], x_train=x_train, y_train=y_train)
    return child
