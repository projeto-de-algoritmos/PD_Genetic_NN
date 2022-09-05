import pandas as pd
from gen_nn import *
from sklearn.model_selection import train_test_split
from typing import List
from random import choice
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score


def init_and_split_dataset(df: pd.DataFrame, test_size: float = 0.2):
    x = df.drop('y', axis=1)
    y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def initiate_neural_nets(x_train, y_train, initial_population: int = 20) -> List:
    neural_nets = []
    for _ in range(initial_population):
        neural_nets.append(GeneticNeuralNet(is_child=False, hidden_layer_weights=None,
                           output_weights=None, x_train=x_train, y_train=y_train))

    return neural_nets
