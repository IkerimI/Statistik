# Bitte nicht ändern
import numpy as np

# Tipp: diese Bibliothek könnte nützlich sein zum Lösen der Aufgaben
import itertools

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

    nuetzliche_variable = "dies könnte eine nuetzliche Variable sein, die man beim Implementieren verwenden koennte"

    ######### Anfang: hier Code  einfügen #########
    ergebnis1 = "Hier steht meine Lösung als string"

    # mit den Variablen kann in einer Funktion gearbeitet werden
    ergebnis2 = nuetzliche_variable + beispiel_input

    return ergebnis1, ergebnis2
    ######### Ende #########


def compute_permutationstest(
    samples1: np.ndarray, samples2: np.ndarray
) -> tuple[float, float]:
    """
    Berechnen Sie den Permutationstest, welcher ermittelt, ob die beiden Stichproben aus der gleichen Population stammen könnten.
    Implementieren Sie das zweiseitige Testproblem unter der Annahme, dass die beiden Arrays stets die gleiche Anzahl an Werten besitzen.

    Input:
        samples1[np.ndarray]    = NumPy-Array der Werte der ersten Stichprobe
        samples2[np.ndarray]    = NumPy-Array der Werte der zweiten Stichprobe

    Output:
        stichproben_diff[float] = die Differenz der Erwartungswerte der Stichproben
        p_wert[float]           = p-Wert für das zweiseitige Testproblem
    """

    ######### Anfang: hier Code einfügen #########
    mean1 = np.mean(samples1)
    mean2 = np.mean(samples2)
    stichproben_diff = np.abs(mean1 - mean2)
    
######### Ende #########


def teilaufgabe_c() -> tuple[float, bool]:
    """
    Führen Sie den Permutationstest für die beiden Stichproben mit einem Signifikanzniveau von 5\% durch.
    Kann die Nullhypothese zugunsten der Alternativhypothese verworfen werden?

    Output:
        p_wert[float]           = p-Wert für das zweiseitige Testproblem
        test_bestanden[bool]    = wahr, wenn die Nullhypothese zugunsten der Alternativhypothese verworfen werden kann
    """

    samples1 = np.array([4.1, 4.6, 3.9, 4.2, 4.0], dtype=np.float64)
    samples2 = np.array([4.5, 4.3, 4.7, 4.8, 4.5], dtype=np.float64)
    alpha = 0.05

    ######### Anfang: hier Code einfügen #########
    compute_permutationstest(samples1, samples2)
    ######### Ende #########


if __name__ == "__main__":
    print(teilaufgabe_c())
