import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets


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
    etiquetas = list(iris.target_names)
    # Randomize
    datos = [(x, y) for x, y in zip(X, Y)]
    random.shuffle(datos)
    X, Y = zip(*datos)
    # Split
    X_train = X[:100]
    Y_train = Y[:100]

    X_test = X[100:]
    Y_test = Y[100:]
    Y_obteined = []
    buenas = 0
    # Predict
    for muestra, etiqueta in zip(X_test, Y_test):
        best_y = knearestneighbors(X_train, Y_train, muestra, k)
        Y_obteined.append(best_y)
        if best_y == etiqueta:
            buenas += 1
        print("Etiqueta real: ", etiquetas[etiqueta], "Etiqueta predicha: ", etiquetas[best_y])
    print("Porcentaje de aciertos: ", buenas / len(Y_test) * 100, "%")

    sepalo_test = [(x[0], x[1], etiqueta) for x, etiqueta in zip(X_test, Y_obteined)]
    petalo_test = [(x[2], x[3], etiqueta) for x, etiqueta in zip(X_test, Y_obteined)]
    sepalo_train = [(x[0], x[1], etiqueta) for x, etiqueta in zip(X_train, Y_train)]
    petalo_train = [(x[2], x[3], etiqueta) for x, etiqueta in zip(X_train, Y_train)]
    colores = ['red', 'green', 'blue']

    for x, y, etiqueta in sepalo_train:
        plt.scatter(x, y, color=colores[etiqueta], alpha=0.2)
    for x, y, etiqueta in sepalo_test:
        plt.scatter(x, y, color=colores[etiqueta])
    plt.show()
    plt.clf()

    for x, y, etiqueta in petalo_train:
        plt.scatter(x, y, color=colores[etiqueta], alpha=0.2)
    for x, y, etiqueta in petalo_test:
        plt.scatter(x, y, color=colores[etiqueta])
    plt.show()
    plt.clf()



if __name__ == "__main__":
    main()

