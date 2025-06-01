import requests
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import random
import os

def nacti_data():
    url = os.getenv("DATA_ENDPOINT")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return [radek for radek in response.json() if radek and len(radek) == 7]

def natrenuj_model(tahy, N=3):
    X = []
    y = []
    for i in range(len(tahy) - N):
        vstup = tahy[i:i+N]
        vystup = tahy[i+N]
        X.append(sum(vstup, []))
        y.append(vystup)

    mlb = MultiLabelBinarizer(classes=list(range(1, 50)))
    y_bin = mlb.fit_transform([set(yy) for yy in y])

    X = np.array(X)
    print("Trénuji model...")
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
    model.fit(X, y_bin)
    print("Model natrénován.")

    return model, mlb

def vytvor_ai_tip(tahy, pocet=6, N=3):
    model, mlb = natrenuj_model(tahy, N)
    vstup_pred = sum(tahy[-N:], [])
    y_pred_bin = model.predict([vstup_pred])[0]
    pravdepodobnosti = list(zip(mlb.classes_, y_pred_bin))
    vybrane = [cislo for cislo, aktivni in pravdepodobnosti if aktivni == 1]
    if len(vybrane) > pocet:
        tip = sorted(random.sample(vybrane, pocet))
    elif len(vybrane) < pocet:
        zbytek = list(set(range(1, 50)) - set(vybrane))
        tip = sorted(vybrane + random.sample(zbytek, pocet - len(vybrane)))
    else:
        tip = sorted(vybrane)
    return tip
