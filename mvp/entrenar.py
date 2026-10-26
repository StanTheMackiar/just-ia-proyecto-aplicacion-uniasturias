"""
JustIA - Proyecto de aplicacion - Fine-tuning y evaluacion de BETO
Stanly Leon Calle Samper

Entrena BETO con los 196 fragmentos de entrenamiento de la Unidad 2 (misma division,
semilla 42) mas las 60 consultas coloquiales, y lo evalua en los 50 fragmentos de
prueba de la Unidad 2 y en 40 consultas que nunca vio (prueba ciega).

Ejecutar: python mvp/entrenar.py
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging

from justia import DISPOSITIVO, MODELO_DIR, RAIZ, UMBRAL, VERSION, anonimizar, limpiar, predecir

BASE = "dccuchile/bert-base-spanish-wwm-cased"
SEMILLA, EPOCAS, LR, LOTE = 42, 5, 3e-5, 8
torch.manual_seed(SEMILLA)
logging.set_verbosity_error()
logging.disable_progress_bar()

df = pd.read_csv(RAIZ / "datos/fragmentos_juridicos.csv")
df["texto"] = df["texto"].apply(limpiar)
entrenamiento, prueba = train_test_split(df, test_size=0.2, stratify=df["categoria"], random_state=SEMILLA)
entrenamiento = pd.concat([entrenamiento, pd.read_csv(RAIZ / "datos/consultas_adicionales.csv")])
ciega = pd.read_csv(RAIZ / "datos/prueba_ciega.csv")
areas = sorted(df["categoria"].unique())
print(f"Entrenamiento: {len(entrenamiento)} | Prueba U2: {len(prueba)} | Prueba ciega: {len(ciega)}")

tokenizer = AutoTokenizer.from_pretrained(BASE)
modelo = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(areas), id2label=dict(enumerate(areas)), label2id={a: i for i, a in enumerate(areas)}
).to(DISPOSITIVO)
optimizador = torch.optim.AdamW(modelo.parameters(), lr=LR)
textos = entrenamiento["texto"].tolist()
etiquetas = torch.tensor(entrenamiento["categoria"].map(modelo.config.label2id).values)

modelo.train()
for epoca in range(EPOCAS):
    orden, perdida = torch.randperm(len(textos)), 0.0
    for i in range(0, len(textos), LOTE):
        idx = orden[i:i + LOTE]
        lote = tokenizer([textos[j] for j in idx], truncation=True, max_length=128, padding=True,
                         return_tensors="pt").to(DISPOSITIVO)
        salida = modelo(**lote, labels=etiquetas[idx].to(DISPOSITIVO))
        salida.loss.backward()
        optimizador.step()
        optimizador.zero_grad()
        perdida += salida.loss.item() * len(idx)
    print(f"Epoca {epoca + 1}/{EPOCAS} - perdida {perdida / len(textos):.4f}")
modelo.eval()

filas = []
figura, ejes = plt.subplots(1, 2, figsize=(15, 6.5))
for eje, (nombre, datos) in zip(ejes, {"prueba_U2": prueba, "prueba_ciega": ciega}.items()):
    probs = predecir(modelo, tokenizer, [anonimizar(t) for t in datos["texto"]])
    pred, certeza = [areas[i] for i in probs.argmax(1)], probs.max(1).values.numpy()
    real, arriba = datos["categoria"].tolist(), certeza >= UMBRAL
    errores = [p != r for p, r in zip(pred, real)]
    filas.append({"conjunto": nombre, "casos": len(real), "accuracy": round(accuracy_score(real, pred), 3),
                  "f1_macro": round(f1_score(real, pred, average="macro"), 3),
                  "cobertura_umbral": round(float(arriba.mean()), 3),
                  "errores_atajados": f"{sum(e and not a for e, a in zip(errores, arriba))}/{sum(errores)}"})
    print(f"\n===== {nombre} ({len(real)} casos) =====")
    print(classification_report(real, pred, labels=areas, zero_division=0))
    for r, p, c, t, e in zip(real, pred, certeza, datos["texto"], errores):
        if e:
            print(f"  {r:<19} -> {p:<19} {c:.0%}  {t[:60]}")
    ConfusionMatrixDisplay.from_predictions(real, pred, labels=areas, ax=eje, cmap="Blues",
                                            colorbar=False, xticks_rotation=45)
    eje.set_title(f"BETO v{VERSION} - {nombre} ({len(real)} casos)")

resumen = pd.DataFrame(filas)
print(f"\n===== Resumen (umbral {UMBRAL:.0%}) =====\n{resumen.to_string(index=False)}")
(RAIZ / "resultados").mkdir(exist_ok=True)
resumen.to_csv(RAIZ / "resultados/metricas.csv", index=False)
figura.tight_layout()
figura.savefig(RAIZ / "resultados/matriz_confusion.png", dpi=150)
modelo.save_pretrained(MODELO_DIR)
tokenizer.save_pretrained(MODELO_DIR)
ficha = {"modelo_base": BASE, "version": VERSION, "epocas": EPOCAS, "lr": LR, "lote": LOTE,
         "textos_entrenamiento": len(textos), "umbral": UMBRAL, "metricas": filas}
(MODELO_DIR / "ficha.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nModelo guardado en {MODELO_DIR.relative_to(RAIZ)}")
