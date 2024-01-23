import sys
import numpy as np
import pandas as pd

from typing import Tuple, List
from pathlib import Path
from sklearn.metrics import classification_report

# Festlegung des Seeds für die Generierung der Zufallszahlen (nicht ändern!)
np.random.seed(2357)

# -------------------------------
# Abgabegruppe: 99
# Personen: Kerim Özkara, Anel Begic
# -------------------------------


def to_one_hot_vectors(y: np.ndarray) -> np.ndarray:
    """
    Diese Funktion wandelt die Goldstandard-Labels in One-Hot-Vektoren um.

    :param y:
    :return:
    """
    n = y.shape[0]
    categorical = np.zeros((n, 2), dtype="float32")
    categorical[np.arange(n), y] = 1
    output_shape = y.shape + (2,)
    categorical = np.reshape(categorical, output_shape)

    return categorical


def create_batches(input: np.ndarray, n: int) -> List[np.ndarray]:
    """
    Teilt das übergebene NumPy-Array in Teilarrays der Größe n ein.

    :param input: Eingabe NumPy-Array
    :param n: Größe eines Batches
    :return: Liste mit allen Teilarrays
    """
    l = len(input)
    batches = []
    for ndx in range(0, l, n):
        batches.append(input[ndx:min(ndx + n, l)])

    return batches


def load_diabetes_dataset(train_size=0.5) -> Tuple[Tuple[np.ndarray, np.ndarray],Tuple[np.ndarray, np.ndarray]]:
    """
    Lädt die Daten aus der Eingabedatei.

    :param train_size: Anteil der Trainingsdaten
    :return: Tupel mit zwei Tupeln welches die Trainingsdaten und -klassen sowie die Testdaten und -klassen
             als NumPy-Array erfasst.
    """
    df = pd.read_csv("diabetes.csv")
    train_size = int(df.shape[0] * train_size)

    X, y = df.iloc[:,:10].to_numpy(), df.iloc[:,10].to_numpy()

    x_train, y_train = X[:train_size,:], y[:train_size]
    x_test, y_test = X[train_size:,:], y[train_size:]

    return (x_train, y_train), (x_test, y_test)


class FeedforwardNetwork:
    """
        Diese Klasse implementiert ein einfaches, einschichtiges neuronales Netzwerk.
    """

    def __init__(self, input_size: int, output_size: int):
        # Membervariablen zur Erfassung der Gewichte und Bias-Werte der linearen
        # Transformation. Die Gewichte und Bias-Werte werden mit zufälligen Werten
        # initialisiert.
        self.weights = np.random.rand(input_size, output_size)  # Dimensionalität (<Anzahl-Features>, <Anzahl-Klassen>)
        self.bias = np.random.rand(output_size)  # Dimensionalität (<Anzahl-Klassen>)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Diese Methode berechnet den Forward-Pass, d.h. die lineare Transformation der Eingabedaten und die Anwendung
        der Aktivierungsfunktion, für die in x gegebenen n Trainingsbeispiele. Als Aktivierungsfunktion soll die
        **Softmax-Funktion** genutzt werden.

        Die Eingabe der n Trainingsbeispiele erfolgt als NumPy-Array der Dimensionalität (n, 10). Als Rückgabe wird
        die Netzwerkausgabe erwartet, d.h. die lineare Transformation mit der Aktivierung 
        (nach Anwendung der Softmax-Funktion). Die Rückgabe soll als NumPy-Float-Array der Dimensionalität (n, 2) erfolgen.

        :param x: Feature-Werte der n Eingabebeispiele (Dim. (n, 10))
        :return: Netzwerkausgabe als NumPy-Array (Dim (n, 2))
        """
        ######### Anfang: hier Code einfügen #########
        z = np.dot(x, self.weights) + self.bias
        return self.softmax(z)
        ######### Ende #########

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(x)
        return exp / np.sum(exp)

    def predict_labels(self, p: np.ndarray) -> np.ndarray:
        """
        Diese Methode berechnet die Vorhersage, d.h. die vom Netzwerk geschätzte Klasse bzw. Ziffer, basierend auf den
        in p gegebenen (Softmax-) Aktivierungen von n Trainingsbeispielen.

        Die Eingabe von p erfolgt als NumPy-Float-Array der Dimensionalität (n, 2). Die Rückgabe der Labels soll
        als NumPy-Int-Array der Länge n erfolgen.

        :param p: Aktivierungszustände von n Beispielen (Dim (n, 2))
        :return: Vorhersage des Netzwerks (Dim (n))
        """
        ######### Anfang: hier Code einfügen #########
        return np.argmax(p, axis=1)
        ######### Ende #########

    def loss(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Diese Methode berechnet den Wert der **Cross-Entropy-Fehlerfunktion** für n Trainingsbeispiele.

        Der Eingabeparameter p der Methode erfasst die Aktivierungen für die n Trainingsbeispiele und ist als
        NumPy-Float-Array der Dimensionalität (n, 10) gegeben. Die Übergabe der Goldstandard-Labels erfolgt mittels
        One-Hot-Vektoren über den Parameter y. y wird als NumPy-Int-Array der Dimensionalität (n, 2) übergeben.

        Die Rückgabe soll als NumPy-Float-Array der Dimensionalität (n, 1), welches den Fehlerwert der jeweiligen
        Trainingsbeispiele enthält, erfolgen.

        :param y: Goldstandard-Klassen für n Beispiele repräsentiert als One-Hot-Vektoren (Dim (n, 2))
        :param p: Aktivierungszustände von n Beispielen (Dim (n, 2))
        :return: Fehlerwerte der n Trainingsbeispiele (Dim (n))
        """
        ######### Anfang: hier Code einfügen #########
        return -np.sum(y * np.log(p))
        ######### Ende #########

    def backward(self, x: np.ndarray, p: np.ndarray, y: np.ndarray):
        """
        Diese Methode berechnet mittels Backpropagation die Gradienten der Gewichte und Bias-Werte für
        n Trainingsbeispiele.

        Die Eingabeparameter x und p repräsentieren hierbei die Features der n Trainingsbeispiele sowie die
        entsprechenden (Softmax) Aktivierungen des Netzwerks. Die Übergabe der Goldstandard-Klassen erfolgt als
        One-Hot-Vektoren über den Parameter y. y wird als NumPy-Int-Array der Dimensionalität (n, 2) übergeben.

        Als Rückgabe wird ein Tupel erwartet, welches die Gradienten der Gewichte (NumPy-Float-Array der
        Dimensionalität (768, 2)) und der Bias-Werte (NumPy-Float-Array der Dimensionalität (2)) erfasst.

        :param x: Feature-Werte der n Trainingsbeispiele (Dim (n, 10))
        :param p: (Softmax) Aktivierungszustände des Netzwerks der n Trainingsbeispiele (Dim (n, 2))
        :param y: Goldstandard-Klassen für n Beispiele repräsentiert als One-Hot-Vektoren (Dim (n, 2))

        :return: Tupel mit den Gradienten der Gewichte (Dim (10, 2)) und der Bias-Werte (Dim (2))
        """
        ######### Anfang: hier Code einfügen #########
        n = x.shape[0]
        weight_grad = np.dot(x.T, p - y) / n
        bias_grad = np.mean(p - y, axis=0)
        return weight_grad, bias_grad
        ######### Ende #########

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, learning_rate: float):
        # Erzeuge Batches mit jeweils batch_size Trainingsbeispielen
        x_batches = create_batches(x, batch_size)
        y_batches = create_batches(y, batch_size)
        num_batches = len(x_batches)

        # Trainingsschleife
        for i in range(epochs):
            sum_loss = 0

            # In jeder Epoche wird jeder Batch einmal betrachtet
            for b_x, b_y in zip(x_batches, y_batches):
                # Berechne den Forward-Pass
                p = self.forward(b_x)

                # Berechne die Gradienten
                grad_weights, grad_bias = self.backward(b_x, p, b_y)

                # Aktualisiere die Gewichte und Bias-Werte anhand des Gradienten
                self.weights -= learning_rate * grad_weights / b_x.shape[0]
                self.bias -= learning_rate * grad_bias / b_x.shape[0]

                # Berechne den aktuellen Fehlerwert (zum Debugging des Lernprozesses)
                loss = self.loss(b_y, p).mean()
                sum_loss += loss

            # Evaluiere das aktuelle Modell auf den Trainingsdaten (nur zum Debugging)
            y_pred = self.predict(x, batch_size)
            accuracy = (y_pred == y.argmax(axis=-1)).sum() / y.shape[0]
            accuracy = round(accuracy, 6)

            # Berechne den durchschnittlichen Fehlerwert
            loss = sum_loss / num_batches
            loss = round(loss, 6)

            print(f" Epoche: {i + 1}/{epochs}   Fehler={loss}   Trainingsgenauigkeit={accuracy}")

        print("Training ist abgeschlossen\n")

    def predict(self, x: np.ndarray, batch_size: int):
        # Führe die Vorhersage auch in Batches durch
        x_batches = create_batches(x, batch_size)

        predictions = []
        for b_x in x_batches:
            # Berechne den Forward-Pass
            p_x = self.forward(b_x)

            # Bilde die Vorhersage anhand der Aktivierungen
            y_pred = self.predict_labels(p_x)
            predictions.append(y_pred)

        # Fasse die Vorhersage der einzelnen Batches in einem Array zusammen
        y_pred = np.concatenate(predictions)

        return y_pred

    def evaluate(self, x: np.ndarray, y_gold: np.ndarray, batch_size: int):
        y_pred = self.predict(x, batch_size)

        print(classification_report(y_gold, y_pred, digits=4, zero_division=1))


if __name__ == "__main__":
    num_epochs = 20000
    batch_size = 24
    learning_rate = 0.1

    # Laden der Daten aus Eingabedatei
    (x_train, y_train), (x_test, y_test) = load_diabetes_dataset(train_size=0.5)

    print(f"#Trainingsinstanzen: {x_train.shape[0]}")
    print(f"#Testinstanzen     : {x_test.shape[0]}\n")

    # Umwandlung der Goldstandard-Labels in ein One-Hot-Vektor;
    # (bspw. das Label 1 wird zu [0, 1])
    y_train_enc = to_one_hot_vectors(y_train)
    y_test_enc = to_one_hot_vectors(y_test)

    # Baue das Netzwerk auf
    ff_network = FeedforwardNetwork(10, 2)

    # Trainiere das Modell anhand der Trainingsdaten
    print(f"Starte das Training des Modells mit {len(x_train)} Beispielen")
    ff_network.fit(x_train, y_train_enc, batch_size=batch_size, epochs=num_epochs, learning_rate=learning_rate)

    # Evaluiere das Modell
    print("Starte die Evaluation des Modells")
    ff_network.evaluate(x_test, y_test, batch_size=batch_size)
