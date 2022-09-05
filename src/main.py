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

def genetic_algorithm(neural_nets: List, x, y) -> GenReturn:
    fitness = 0
    best_weights = []
    output_layer_weights = []
    computed_nns = []
    gen = 0
    log_file = open('log_file.txt', 'w')

    while fitness < .85:
        print(f"Generation {gen}")
        gen += 1
        for idx, neural_net in enumerate(neural_nets):
            neural_net_fp_value = neural_net.forward_prop()
            print(f"Neural Net {idx} score -> {neural_net_fp_value}")
            computed_nns.append(neural_net)
            log_file.write(str(neural_net.fitness_value) + '\n')

        neural_nets.clear()

        computed_nns = sorted(
            computed_nns, key=lambda x: x.fitness_value, reverse=True)

        for i in range(0, len(computed_nns)):
            if computed_nns[i].fitness_value > fitness:
                fitness = computed_nns[i].fitness_value
                print(f"Gen {gen}")
                print(f"Max fitness value -> {fitness}")
                best_weights = []
                output_layer_weights = []
                for idx, layer in enumerate(computed_nns[i].layers):
                    if idx != len(computed_nns) - 1:
                        best_weights.append(layer.get_weights()[0])
                    else:
                        output_layer_weights.append(layer.get_weights()[0])

        for i in range(0, 3):
            for _ in range(0, 2):
                tmp = crossover(computed_nns[i], 
                                choice(computed_nns), 
                                x_train=x, 
                                y_train=y)
                neural_nets.append(tmp)
    log_file.close()
    return (best_weights, output_layer_weights)

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    y = []

    x_train, x_test, y_train, y_test = init_and_split_dataset(df=df)

    nns = initiate_neural_nets(x_train=x_train, y_train=y_train)
    (best_weights, output_layer_weights) = genetic_algorithm(
        neural_nets=nns, x=x_train, y=y_train)

    gnn = GeneticNeuralNet(is_child=True, hidden_layer_weights=best_weights,
                           output_weights=output_layer_weights, epochs=20,
                           x_train=x_train, y_train=y_train)
    gnn.train()
    acc = gnn.accuracy(x_test, y_test)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train.values, y_train.values, epochs=10)
    y_hat = model.predict(x_test.values)
    nn_acc = accuracy_score(y_test.values, y_hat.round())

    print(f'GNN acc: {acc}')
    print(f'Normal NN acc: {nn_acc}')

    for idx, line in enumerate(open('log_file.txt', 'r')):
        y.append(float(line))
    plt.title("Value")
    plt.plot(y)
    plt.show()
