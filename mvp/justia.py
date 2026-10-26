"""
JustIA - Proyecto de aplicacion - MVP de clasificacion de consultas con BETO
Stanly Leon Calle Samper

Verifica el hash del modelo contra el registro de la Unidad 3, anonimiza la consulta,
sugiere el area del derecho con BETO, aplica el umbral y las reglas de urgencia y de
violencia economica y patrimonial, explica la sugerencia por oclusion y guarda una
traza en trazas/trazas_justia.jsonl. Ninguna sugerencia llega al usuario sin un practicante.

Uso: python mvp/justia.py "texto" | --demo | -i
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

import torch

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "gobernanza"))
from registro_modelos import REGISTRO, sha256  # noqa: E402

ID_MODELO, VERSION = "justia-clf-beto", "2.0.0"
MODELO_DIR = RAIZ / f"modelos/beto_justia_v{VERSION}"
TRAZAS = RAIZ / "trazas/trazas_justia.jsonl"
UMBRAL = 0.70
PISO_DOMINIO = 0.35
VACIAS = {"este", "esta", "estos", "estas", "nuestra", "nuestro", "para", "pero", "como", "cuando", "donde",
          "desde", "porque", "tambien", "también", "sobre", "entre", "todo", "toda", "muy", "tengo", "tiene"}
DISPOSITIVO = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

ANONIMIZAR = [
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "[CORREO]"),
    (r"\b3\d{2}[\s-]?\d{3}[\s-]?\d{4}\b", "[TELEFONO]"),
    (r"(?i)\b(c[eé]dula|c\.?c\.?)\s*(n[o°º.]*\s*)?[\d.]{6,13}", r"\1 [DOCUMENTO]"),
    (r"(?i)\b(calle|carrera|cra|avenida|diagonal|transversal)\.?\s*\d+\w?\s*(#|no\.?)\s*\d+\w?\s*-\s*\d+",
     "[DIRECCION]"),
    (r"(?i)\b(me llamo|mi nombre es)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}",
     r"\1 [NOMBRE]"),
    (r"\b\d{1,3}(\.\d{3}){2,3}\b|\b\d{7,10}\b", "[NUMERO]"),
]
PAREJA = r"(esposo|marido|pareja|companero|novio|\bex(esposo|marido|pareja|companero|novio)?\b|(^|[.,;¿]\s*)el\b)"
REGLAS = [
    ("URGENCIA", r"(me (va|van|quiere|quieren) a matar|me va a hacer dano|esta afuera|tiene un arma"
                 r"|me esta pegando)", None, "Posible riesgo inminente: atencion humana inmediata, Linea 155 o 123."),
    ("VIOLENCIA_ECONOMICA", r"(controla|maneja|retiene|quita|quito|niega|no me da|no me deja)",
     r"(plata|dinero|salario|sueldo|tarjeta|cuenta|mercado)",
     "Posible violencia economica (Ley 1257 de 2008): revisar en el area de genero."),
    ("VIOLENCIA_PATRIMONIAL", r"(quito|retiene|escondio|rompio|destruyo|vendio|se llevo)",
     r"(cedula|documentos|papeles|herramientas|celular|moto)",
     "Posible violencia patrimonial (Ley 1257 de 2008): revisar en el area de genero."),
]


def limpiar(texto):
    texto = re.sub(r"\((simulad[oa]|adaptado|texto adaptado)\)|https?://\S+", " ", texto, flags=re.I)
    return re.sub(r"\s+", " ", texto).strip()


def anonimizar(texto):
    for patron, reemplazo in ANONIMIZAR:
        texto = re.sub(patron, reemplazo, texto)
    return texto


def aplicar_reglas(texto):
    t = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return [{"regla": n, "accion": a} for n, verbo, objeto, a in REGLAS
            if re.search(verbo, t) and (objeto is None or (re.search(objeto, t) and re.search(PAREJA, t)))]


@torch.no_grad()
def predecir(modelo, tokenizer, textos):
    lote = tokenizer(textos, truncation=True, max_length=128, padding=True, return_tensors="pt")
    return torch.softmax(modelo(**lote.to(modelo.device)).logits, dim=1).cpu()


def cargar_modelo():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
    logging.disable_progress_bar()
    if not MODELO_DIR.exists():
        sys.exit("No existe el modelo. Primero: python mvp/entrenar.py")
    with open(REGISTRO, encoding="utf-8") as f:
        fila = next((r for r in csv.DictReader(f) if r["id_modelo"] == ID_MODELO and r["version"] == VERSION), None)
    hash_actual = sha256(MODELO_DIR)
    if fila is None:
        print(f"AVISO: {ID_MODELO} v{VERSION} no esta registrado. Modo evaluacion, no usar con usuarios.")
    elif fila["hash_modelo"] != hash_actual:
        sys.exit("ALERTA: el modelo cambio despues de registrarse (hash distinto). No se usa.")
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_DIR).to(DISPOSITIVO).eval()
    return modelo, AutoTokenizer.from_pretrained(MODELO_DIR), hash_actual


def explicar(modelo, tokenizer, texto, clase, top=5):
    """Oclusion: cuanto baja la probabilidad del area si se quita cada palabra."""
    palabras = sorted({p for p in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñÜü]{4,}", texto) if p.lower() not in VACIAS},
                      key=str.lower)
    if not palabras:
        return []
    variantes = [texto] + [re.sub(rf"\b{re.escape(p)}\b", " ", texto) for p in palabras]
    probs = predecir(modelo, tokenizer, variantes)[:, clase]
    aportes = sorted(zip(palabras, (probs[0] - probs[1:]).tolist()), key=lambda x: -x[1])
    return [(p, round(a, 3)) for p, a in aportes[:top] if a > 0.005]


def clasificar(consulta, modelo, tokenizer, hash_modelo):
    areas = modelo.config.id2label
    texto = anonimizar(limpiar(consulta))
    probs = predecir(modelo, tokenizer, [texto])[0]
    orden = probs.argsort(descending=True).tolist()
    clase, certeza = orden[0], float(probs[orden[0]])
    alertas = aplicar_reglas(texto)
    a_favor = explicar(modelo, tokenizer, texto, clase)
    if any(a["regla"] == "URGENCIA" for a in alertas):
        decision = "atencion_inmediata"
    elif certeza < PISO_DOMINIO:
        decision = "posible_fuera_de_alcance"
    elif certeza >= UMBRAL and not alertas:
        decision = "sugerencia_para_validar"
    else:
        decision = "revision_manual"
    palabras = ", ".join(f"'{p}'" for p, _ in a_favor[:3]) or "ninguna palabra en particular"
    traza = {
        "id_traza": f"JIA-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6]}",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": ID_MODELO, "version": VERSION, "hash_modelo": hash_modelo[:16],
        "hash_consulta": hashlib.sha256(consulta.encode()).hexdigest()[:16],
        "texto_anonimizado": texto,
        "top3": [{"area": areas[i], "probabilidad": round(float(probs[i]), 3)} for i in orden[:3]],
        "area_sugerida": areas[clase], "certeza": round(certeza, 3), "umbral": UMBRAL,
        "alertas": alertas, "decision": decision, "explicacion": a_favor,
        "justificacion": f"Se sugiere {areas[clase]} con {certeza:.1%} (umbral {UMBRAL:.0%}). "
                         f"Lo que mas pesa hacia esa area: {palabras}.",
        "revision_humana": {"estado": "pendiente", "practicante": None, "motivo": None},
    }
    TRAZAS.parent.mkdir(exist_ok=True)
    with open(TRAZAS, "a", encoding="utf-8") as f:
        f.write(json.dumps(traza, ensure_ascii=False) + "\n")
    return traza


def mostrar(consulta, t):
    print("=" * 78 + f"\nConsulta : {consulta}")
    if t["texto_anonimizado"] != consulta:
        print(f"Anonimiz.: {t['texto_anonimizado']}")
    print("Top 3    : " + " | ".join(f"{x['area']} {x['probabilidad']:.1%}" for x in t["top3"]))
    print(f"A favor  : {t['explicacion']}")
    for a in t["alertas"]:
        print(f"ALERTA   : {a['regla']} -> {a['accion']}")
    print(f"Decision : {t['decision'].upper()}  (traza {t['id_traza']}, v{t['version']} {t['hash_modelo']})")
    print(f"Por que  : {t['justificacion']}")


DEMO = [
    "Me despidieron estando incapacitado y no me pagaron las vacaciones",
    "Mi exmarido no me deja ver a los niños y tampoco paga la cuota",
    "La Fiscalía legalizó la captura del sospechoso del atraco",
    "Nos desplazaron de la vereda y queremos volver a nuestra tierra",
    "Él controla mi plata y me quitó la tarjeta del banco",
    "Me llamo Carlos Pérez, cédula 1045678901, celular 300 123 4567. Me echaron sin pagarme la liquidación",
    "Necesito ayuda ya, mi expareja está afuera de la casa gritando que me va a matar",
    "¿Cuál es el mejor celular para comprar este año?",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JustIA - clasificador de consultas con BETO (MVP)")
    parser.add_argument("consulta", nargs="?")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("-i", "--interactivo", action="store_true")
    args = parser.parse_args()
    modelo, tokenizer, hash_modelo = cargar_modelo()
    consultas = DEMO if args.demo else [args.consulta] if args.consulta else iter(
        lambda: input("\nConsulta ('salir' para terminar): ").strip(), "salir")
    for c in filter(None, consultas):
        mostrar(c, clasificar(c, modelo, tokenizer, hash_modelo))
    print(f"\nTrazas guardadas en {TRAZAS.relative_to(RAIZ)}")
