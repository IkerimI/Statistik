# Bitte nicht ändern
import random
random.seed(2357)

# Bitte nicht ändern
import numpy as np
np.random.seed(2357)

# Tipp: diese Funktionen / Bibliotheken könnten nützlich sein zum Lösen der Aufgaben
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Abgabegruppe: 99
# Personen: Kerim Özkara, Anel Begic
# -------------------------------


# Das ist eine Beispiel-Funktion. Hier muss nichts bearbeitet oder gecodet werden! Sie dient nur zur Unterstützung.
def example_function(beispiel_input: str) -> tuple[str, str]:
    """
    Hier steht die Aufgabenstellung. "Output" gibt den Variablennamen sowie den Rückgabetypen an.
    Euer Ergebnis zu der Aufgabe soll sowohl den Rückgabetypen als auch Variablennamen besitzen
    und mit "return" von der funktion ausgegeben werden.
    Tipp: löscht die Zeile "raise NotImplementedError" sobald ihr mit dem implementieren beginnt.

    Input:
        beispiel_input[str]     = eine Beispielvariable die als Eingabe für die Funktion dient.
    Output:
        ergebnis1[str]          = das erste Ergebnis
        ergebnis2[str]          = das zweite Ergebnis
    """
    # das Dataframe wird geladen
    df = load_electricity_dataset()

    ######### Anfang: hier Code  einfügen #########
    ergebnis1 = "Hier steht meine Lösung als string"

    # mit den Variablen kann in einer Funktion gearbeitet werden
    ergebnis2 = df[0] + beispiel_input

    return ergebnis1, ergebnis2
    ######### Ende #########


def load_electricity_dataset():
    """
    Liest die Verbrauchswerte als NumPy-Array ein.
    """
    consumption = np.loadtxt("electricity.txt")
    return consumption


def compute_bootstrapping_means(samples: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Bildet basierend auf dem übergebenen Werten eines Samples n Bootstrapping-Samples mit jeweils
    m Werten und berechnet jeweils das mittlere Element dieser Bootstrapping-Samples. Die Funktion
    gibt die Liste der berechneten mittleren Elemente zurück.

    Input:
        samples[np.ndarray] = NumPy-Array der Werte des ursprünglichen Examples
        n[int]              = Anzahl der zu bildenden Bootstrapping-Samples
        m[int]              = Anzahl der Elemente je Bootstrapping-Sample

    Output:
        means[np.ndarray]   = Numpy-Array mit den mittleren Elementen der n Bootstrapping-Samples
    """
    means = np.zeros(n, np.float64)

    ######### Anfang: hier Code einfügen #########
    for i in range(n):
        #Auswahl von Werten mit Zuruecklegen
        pick = np.random.choice(samples, size=m, replace=True)
        mean = np.mean(pick)
        means[i] = mean
    return means
    ######### Ende #########


def calculate_standard_error(means: np.ndarray) -> float:
    """
    Gibt den Stichprobenfehler / Standardfehler der übergebenen Liste von mittleren Elemente
    des Bootstrappings zurück.

    Input:
        means[np.ndarray]   = Liste der berechneten Mittelwerte aus dem Bootstrapping

    Output:
        std_error[float]    = Standardfehler der Bootstrapping-Mittelwerte
    """

    ######### Anfang: hier Code einfügen #########
    std = np.std(means, ddof=1)
    std_error = std/np.sqrt(len(means))
    return std_error
    ######### Ende #########


def calculate_confidence_interval(means: np.ndarray, p: float) -> tuple[float, float]:
    """
    Berechnet das p-% Konfidenzintervall der übergebenen Liste an Bootstrapping-Mittelwerte.

    Input:
        means[np.ndarray]   = Numpy-Array der berechneten Mittelwerte aus dem Bootstrapping
        p[flaot]            = Zu ermittelndes Konfidenzintervall im Bereich (0,1), float

    Output:
        start_conf[float]        = Startwerte des Konfidenzintervalls
        end_conf[float]          = Endwerte des Konfidenzintervalls
    """
    if p >= 1.0 or p <= 0.0:
        raise AssertionError("p muss im Intervall (0,1) liegen")

    ######### Anfang: hier Code einfügen #########
    means.sort()
    lower_index = ((1-p)/2)*len(means)
    start_conf = means[int(lower_index)]
    end_conf = means[-int(lower_index)]
    return start_conf, end_conf
    ######### Ende #########


def teilaufgabe_b1() -> tuple[float, tuple[float, float]]:
    """
    Berechnen Sie den Standardfehler und das 95%-Konfidenzintervall mit 1000 Bootstrapping-Stichproben,
    wobei jede Stichprobe 100 Verbrauchswerte enthält, basierend auf den gegebenen Daten.

    Output:
        standard_fehler[float]                  = Standardfehler
        095_intervall[tuple(float, float)]      = 95 % Konfidenzintervall
    """
    consumption = load_electricity_dataset()

    ######### Anfang: hier Code einfügen #########
    standard_fehler = calculate_standard_error(compute_bootstrapping_means(consumption, 1000, 100))
    intervall = calculate_confidence_interval(compute_bootstrapping_means(consumption, 1000, 100), 0.95)
    return standard_fehler, intervall
    ######### Ende #########


def teilaufgabe_b2():
    """
    Erstellen Sie einen Boxplot, der die Durchschnittsverbrauchswerte der in Teilaufgabe (b1) erzeugten
    Bootstrapping-Stichproben visualisiert.

    Output:
        Nichts! Speichern Sie den Boxplot und reichen ihn als JPEG-Datei ein!
    """
    consumption = load_electricity_dataset()

    ######### Anfang: hier Code einfügen #########
    means = compute_bootstrapping_means(consumption, 1000, 100)
    bootstrapped_means = compute_bootstrapping_means(means, 1000, 100)
    plt.boxplot(bootstrapped_means)
    plt.savefig('boxplot.jpeg')
    ######### Ende #########


if __name__ == "__main__":
    print(teilaufgabe_b1())
    teilaufgabe_b2()