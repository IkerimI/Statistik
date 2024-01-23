from operator import itemgetter
import sys
import pandas as pd
import numpy as np
import scipy

# -------------------------------
# Abgabegruppe: 99
# Personen: Kerim Özkara, Anel Begic
# -------------------------------


# Das ist eine Beispiel-Funktion. Hier muss nichts bearbeitet oder gecodet werden! Sie dient nur zur Unterstützung.
def example_function():
    """
    Hier steht die Aufgabenstellung. "Output" gibt den Variablennamen sowie den Rückgabetypen an.
    Euer Ergebnis zu der Aufgabe soll sowohl den Rückgabetypen als auch Variablennamen besitzen
    und mit "return" von der funktion ausgegeben werden.
    Tipp: löscht die Zeile "raise NotImplementedError" sobald ihr mit dem implementieren beginnt.

    Output:
        ergebnis = str
    """
    # das Dataframe wird geladen
    df = load_tornado_dataset()

    ######### Anfang: hier Code einfügen #########
    ergebnis = "Hier steht meine Lösung als string"
    return ergebnis
    ######### Ende #########


def load_tornado_dataset():
    """
    Liest den Tornado Datensatz als Pandas Dataframe ein.
    """
    df = pd.read_csv("tornado.csv")
    return df


def teilaufgabe_a1():
    """
    Berechnen Sie den Mittelwert, die Standardabweichung und den unverzerrten Fisher-Pearson Schiefekoeffizienten der jährlichen Anzahl von Tornados.

    Output:
        mean = float,
        std = float,
        skew = float
    """
    df = load_tornado_dataset()

    ######### Anfang: hier Code einfügen #########
    counterList = []
    counter = 0
    column = df["jahr"]
    year = column[0]
    # Anzahl der Tornados fuer jedes Jahr zaehlen und speichern
    for element in column:
        if element == year:
            counter += 1
        else:
            year += 1
            counterList.append(counter)
            counter = 1
    mean = np.mean(counterList)
    std = np.std(counterList)
    skew = scipy.stats.skew(counterList)
    return mean, std, skew
    ######### Ende #########


def teilaufgabe_a2():
    """
    Ermitteln Sie die minimale und maximale Anzahl von Verletzten und Todesfällen, die durch einen Tornado der Stärke F3 verursacht wurden.

    Output:
        min_verletzte = int,
        max_verletzte = int,
        min_todesfaelle = int,
        max_todesfaelle = int
    """
    df = load_tornado_dataset()

    ######### Anfang: hier Code einfügen #########
    min_verletzte = sys.maxsize
    min_todesfaelle = sys.maxsize
    max_verletzte = 0
    max_todesfaelle = 0
    staerke = df["magnitude"]
    verletzte = df["verletzte"]
    todesopfer = df["todesopfer"]
    counter = 0
    for element in staerke:
        if element == 3:
            if verletzte[counter] < min_verletzte:
                min_verletzte = verletzte[counter]
            if verletzte[counter] > max_verletzte:
                max_verletzte = verletzte[counter]
            if todesopfer[counter] > max_todesfaelle:
                max_todesfaelle = todesopfer[counter]
            if todesopfer[counter] < min_todesfaelle:
                min_todesfaelle = todesopfer[counter]
        counter += 1
    return min_verletzte, max_verletzte, min_todesfaelle, max_todesfaelle
    ######### Ende #########


def teilaufgabe_a3():
    """
    Berechnen Sie den Interquartilsabstand und das 95% Perzentil für die Fläche (in Quadratkilometern) von F2-Tornados, die in Florida aufgetreten sind.

    Output:
        iqr = float,
        percentile_95 = float
    """
    df = load_tornado_dataset()

    ######### Anfang: hier Code einfügen #########
    staerke = df["magnitude"]
    laenge = df["streckenlaenge"]
    breite = df["breite"]
    flaechen = []
    counter = 0
    # alle Flaechen berechnen
    for element in staerke:
        if element == 2:
            flaechen.append(laenge[counter] * breite[counter])
        counter += 1
    iqr = scipy.stats.iqr(flaechen)
    percentile_95 = np.percentile(flaechen, 95)
    return iqr, percentile_95
    ######### Ende #########


def teilaufgabe_b1():
    """
    Welcher Monat hat im Durchschnitt die höchste Anzahl an Tornados pro Bundesstaat? Hat sich dieser Monat im Zeitraum von 2010 bis 2015 in den meisten Bundesstaaten geändert?
    Hinweis: Geben Sie an, in wie viel Prozent der Bundesstaaten sich der Monat in der Zeitspanne geändert hat.

    Output:
        bundesstaaten = dict(str: int),
        aenderung = float
    """
    df = load_tornado_dataset()
    bundesstaaten = {}

    ######### Anfang: hier Code einfügen #########
    original_states = df["bundesstaat"]
    # alle Duplikate aus der Liste der Staaten entfernen
    states = list(dict.fromkeys(original_states))
    monate = df["monat"]
    for state in states:
        monatszahl = {}
        counter = 0
        # fuer jeden Monat die Anzahl an Tornados speichern
        for original_state in original_states:
            if original_state == state:
                if monatszahl.get(monate[counter]) == None:
                    monatszahl[monate[counter]] = 1
                else:
                    monatszahl[monate[counter]] = monatszahl.get(monate[counter]) + 1
            counter += 1
        for element in monatszahl:
            monatszahl[element] = monatszahl[element] / 71
        result_monat = max(monatszahl, key=monatszahl.get)
        bundesstaaten[state] = result_monat
    return bundesstaaten
    ######### Ende #########


def teilaufgabe_b2():
    """
    Wie hoch ist die Korrelation zwischen der Magnitude eines Tornados und der Anzahl der Verletzten bzw. Todesopfer?
    Welche der beiden Korrelationen ist stärker? Hinweis: Schließen Sie Einträge aus Ihrer Analyse aus, die keine Magnitudeangaben enthalten.

    Output:
        corr_verletzte = float,
        corr_todesopfer = float
    """
    df = load_tornado_dataset()

    ######### Anfang: hier Code einfügen #########
    staerke = []
    verletzte = []
    todesopfer = []
    magnitude = df["magnitude"]
    original_verletzte = df["verletzte"]
    original_todesopfer = df["todesopfer"]
    counter = 0
    for magnitude in magnitude:
        if magnitude != -9:
            staerke.append(magnitude)
            verletzte.append(original_verletzte[counter])
            todesopfer.append(original_todesopfer[counter])
        counter += 1
    corr_verletzte = pd.Series(staerke).corr(pd.Series(verletzte))
    corr_todesopfer = pd.Series(staerke).corr(pd.Series(todesopfer))
    return corr_verletzte, corr_todesopfer
    ######### Ende #########


def teilaufgabe_b3():
    """
    Welche Magnitude weisen die Tornados im Median auf, die 5\% der größten Fläche bezüglich aller Tornados repräsentieren?
    Hinweis: Berechnen Sie die Tornados mit den 5\% der größten Flächen und geben sie von diesen den Median der Magnitude an.

    Output:
        median_magnitude = float
    """
    df = load_tornado_dataset()

    ######### Anfang: hier Code einfügen #########
    magnitude = df["magnitude"]
    streckenlaenge = df["streckenlaenge"]
    breite = df["breite"]
    flaeche = []
    flaeche_magnitude = [[]]
    result_magnitude = []
    counter = 0
    for breite in breite:
        flaeche.append(streckenlaenge[counter] * breite)
        temp = []
        temp.append(streckenlaenge[counter] * breite)
        temp.append(magnitude[counter])
        flaeche_magnitude.append(temp)
    flaeche.sort()
    kleinste_flaeche = flaeche[int(np.ceil(len(flaeche) * (1 / 20)))]
    for flaeche_magnitude in flaeche_magnitude:
        if flaeche_magnitude != []:
            if flaeche_magnitude[0] >= kleinste_flaeche:
                result_magnitude.append(flaeche_magnitude[1])
    median_magnitude = np.median(result_magnitude)
    return median_magnitude
    ######### Ende #########


if __name__ == "__main__":
    print(f"Teilaufgabe a1: {teilaufgabe_a1()}")
    print(f"Teilaufgabe a2: {teilaufgabe_a2()}")
    print(f"Teilaufgabe a3: {teilaufgabe_a3()}")

    print(f"Teilaufgabe b1: {teilaufgabe_b1()}")
    print(f"Teilaufgabe b2: {teilaufgabe_b2()}")
    print(f"Teilaufgabe b3: {teilaufgabe_b3()}")
