import requests
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import random
import os
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
import logging
import traceback

# FastAPI server
app = FastAPI()

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
    X_train, _, y_train, _ = train_test_split(X, y_bin, test_size=0.2, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

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

def nacti_data():
    try:
        url = os.getenv("DATA_ENDPOINT")  # např. https://mojedomena.cz/sportka_json.php
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return [radek for radek in response.json() if radek and len(radek) == 7]
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })

@app.get("/tip")
def generuj_tipy(n: int = Query(5, ge=1, le=20)):
    try:
        tahy = nacti_data()
        if isinstance(tahy, JSONResponse):
            return tahy
        tipy = [vytvor_ai_tip(tahy) for _ in range(n)]
        spojena = sum(tipy, [])
        from collections import Counter
        ult_tip = [cislo for cislo, _ in Counter(spojena).most_common(6)]
        return JSONResponse(content={"tips": tipy, "ultimate": sorted(ult_tip)})
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "error": str(e),
            "traceback": traceback.format_exc()
        })

if __name__ == "__main__":
    uvicorn.run("ai_tip:app", host="0.0.0.0", port=8000, reload=False)
