import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt


def knearestneighbors(X, Y, x, k):
    distances = []
    for muestra, etiqueta, in zip(X, Y):
        dist = np.linalg.norm(muestra - x)
        distances.append((etiqueta, dist))
    sorted_distances = sorted(distances, key=lambda x: x[1])[:k]
    occurrences = {}
    for etiqueta, dist in sorted_distances:
        if etiqueta in occurrences:
            occurrences[etiqueta] += 1
        else:
            occurrences[etiqueta] = 1
    best_y = max(occurrences, key=occurrences.get)
    return best_y



def main():
    # Load Iris dataset
    k = 11
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    # Randomize
    datos = [(x, y) for x, y in zip(X, Y)]
    random.shuffle(datos)
    X, Y = zip(*datos)
    # Split
    X_train = X[:100]
    Y_train = Y[:100]
    X_test = X[100:]
    Y_test = Y[100:]
    print(Y_train)
    buenas = 0
    # Predict
    for muestra, etiqueta in zip(X_test, Y_test):
        best_y = knearestneighbors(X_train, Y_train, muestra, k)
        if best_y == etiqueta:
            buenas += 1
        print("Etiqueta real: ", etiqueta, "Etiqueta predicha: ", best_y)
    print("Porcentaje de aciertos: ", buenas / len(Y_test) * 100, "%")


if __name__ == "__main__":
    main()

