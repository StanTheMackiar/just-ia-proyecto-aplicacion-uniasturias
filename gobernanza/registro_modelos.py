"""
JustIA - Unidad 3 - Actividad 2
Stanly Leon Calle Samper

Registro de versiones de los modelos de JustIA. Cada version entrenada queda
como una fila en control_versiones_modelos.csv con su fecha, el corpus con el
que se entreno, el objetivo, el riesgo principal y la validacion que se hizo
antes de desplegarla. El archivo solo crece: no se editan ni se borran filas,
si algo cambia se registra una version nueva.

Para que la trazabilidad no dependa de la buena fe, el script calcula el
SHA-256 del corpus y del modelo. Si alguien reentrena o cambia un archivo y no
lo registra, el comando "verificar" lo detecta.

Uso:
  python registro_modelos.py registrar --id justia-clf-beto --version 1.0.0 ...
  python registro_modelos.py listar
  python registro_modelos.py verificar --id justia-clf-beto --version 1.0.0 --artefacto ruta/modelo
  python registro_modelos.py exportar-md
"""

import argparse
import csv
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
REGISTRO = CARPETA / "control_versiones_modelos.csv"
TABLA_MD = CARPETA / "control_versiones_modelos.md"

CAMPOS = ["id_modelo", "version", "fecha_entrenamiento", "objetivo", "tecnica",
          "corpus_fuente", "hash_corpus", "hash_modelo", "metricas",
          "riesgo_potencial", "detalle_riesgo", "validacion_previa",
          "revisado_por", "estado", "version_anterior", "fecha_registro"]
OBJETIVOS = {"clasificacion", "extraccion_entidades", "preguntas_respuestas", "busqueda_semantica"}
RIESGOS = {"sesgo", "falsos_positivos", "falsos_negativos", "opacidad", "alucinacion", "privacidad"}
ESTADOS = {"en_revision", "aprobado_piloto", "rechazado", "retirado"}
SIN_ARCHIVO = "pendiente"


def sha256(ruta):
    """Hash de un archivo o de todos los archivos de una carpeta (en orden)."""
    ruta = Path(ruta)
    if not ruta.exists():
        sys.exit(f"No existe: {ruta}")
    archivos = [ruta] if ruta.is_file() else sorted(p for p in ruta.rglob("*")
                                                            if p.is_file() and not p.name.startswith("."))
    h = hashlib.sha256()
    for archivo in archivos:
        with open(archivo, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
    return h.hexdigest()


def leer_registro():
    if not REGISTRO.exists():
        return []
    with open(REGISTRO, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def version_a_tupla(v):
    return tuple(int(x) for x in v.split("."))


def validar(args, filas):
    errores = []
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        errores.append("La version debe tener formato MAYOR.MENOR.PARCHE (ej. 1.0.0)")
    if args.objetivo not in OBJETIVOS:
        errores.append(f"Objetivo no valido. Opciones: {sorted(OBJETIVOS)}")
    riesgos = set(args.riesgo.split("|"))
    if not riesgos <= RIESGOS:
        errores.append(f"Riesgo no valido. Opciones: {sorted(RIESGOS)}")
    if args.estado not in ESTADOS:
        errores.append(f"Estado no valido. Opciones: {sorted(ESTADOS)}")
    if args.estado == "aprobado_piloto":
        # nadie aprueba su propio modelo sin dejar evidencia
        if not args.validacion or not args.revisado_por:
            errores.append("Para aprobar hay que dejar la validacion y quien la reviso")
        if not args.artefacto:
            errores.append("No se aprueba un modelo sin el archivo entrenado (sin hash)")
    previas = [f for f in filas if f["id_modelo"] == args.id]
    if any(f["version"] == args.version for f in previas):
        errores.append("Esa version ya esta registrada; el registro no se sobreescribe")
    elif previas and not errores:
        ultima = max(previas, key=lambda f: version_a_tupla(f["version"]))
        if version_a_tupla(args.version) <= version_a_tupla(ultima["version"]):
            errores.append(f"La version debe ser mayor que la ultima ({ultima['version']})")
    return errores


def registrar(args):
    filas = leer_registro()
    errores = validar(args, filas)
    if errores:
        sys.exit("No se registro:\n - " + "\n - ".join(errores))
    previas = sorted((f for f in filas if f["id_modelo"] == args.id),
                     key=lambda f: version_a_tupla(f["version"]))
    fila = {
        "id_modelo": args.id, "version": args.version, "fecha_entrenamiento": args.fecha,
        "objetivo": args.objetivo, "tecnica": args.tecnica,
        "corpus_fuente": args.corpus_desc or Path(args.corpus).name,
        "hash_corpus": sha256(args.corpus) if args.corpus else SIN_ARCHIVO,
        "hash_modelo": sha256(args.artefacto) if args.artefacto else SIN_ARCHIVO,
        "metricas": args.metricas, "riesgo_potencial": args.riesgo,
        "detalle_riesgo": args.detalle_riesgo, "validacion_previa": args.validacion,
        "revisado_por": args.revisado_por, "estado": args.estado,
        "version_anterior": previas[-1]["version"] if previas else "",
        "fecha_registro": date.today().isoformat(),
    }
    nuevo = not REGISTRO.exists()
    with open(REGISTRO, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)
    print(f"Registrado {args.id} v{args.version} ({args.estado}) hash={fila['hash_modelo'][:12]}")


def listar(_):
    for f in leer_registro():
        print(f"{f['id_modelo']:<22} v{f['version']:<7} {f['fecha_entrenamiento']}  "
              f"{f['estado']:<16} riesgo={f['riesgo_potencial']}")


def verificar(args):
    fila = next((f for f in leer_registro()
                 if f["id_modelo"] == args.id and f["version"] == args.version), None)
    if fila is None:
        sys.exit("Esa version no esta en el registro")
    actual = sha256(args.artefacto)
    if actual == fila["hash_modelo"]:
        print("OK: el modelo es el mismo que se registro y se valido")
    else:
        print("ALERTA: el modelo cambio despues de registrarse. No desplegar hasta registrar "
              "una version nueva y validarla otra vez.")
        sys.exit(1)


def exportar_md(_):
    columnas = ["id_modelo", "version", "fecha_entrenamiento", "objetivo", "corpus_fuente",
                "riesgo_potencial", "validacion_previa", "estado"]
    lineas = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    for f in leer_registro():
        lineas.append("| " + " | ".join(f[c].replace("|", ", ") for c in columnas) + " |")
    TABLA_MD.write_text("# Control de versiones de modelos JustIA\n\n" + "\n".join(lineas) + "\n",
                        encoding="utf-8")
    print(f"Tabla exportada en {TABLA_MD.name}")


def main():
    parser = argparse.ArgumentParser(description="Registro de versiones de modelos JustIA")
    sub = parser.add_subparsers(dest="comando", required=True)
    r = sub.add_parser("registrar")
    r.add_argument("--id", required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--fecha", required=True, help="fecha de entrenamiento AAAA-MM-DD")
    r.add_argument("--objetivo", required=True)
    r.add_argument("--tecnica", required=True)
    r.add_argument("--corpus", help="archivo o carpeta del corpus (se calcula el hash)")
    r.add_argument("--corpus-desc", default="", help="descripcion del corpus")
    r.add_argument("--artefacto", help="archivo o carpeta del modelo entrenado")
    r.add_argument("--metricas", default="")
    r.add_argument("--riesgo", required=True, help="uno o varios separados por |")
    r.add_argument("--detalle-riesgo", default="")
    r.add_argument("--validacion", default="")
    r.add_argument("--revisado-por", default="")
    r.add_argument("--estado", default="en_revision")
    r.set_defaults(func=registrar)
    sub.add_parser("listar").set_defaults(func=listar)
    v = sub.add_parser("verificar")
    v.add_argument("--id", required=True)
    v.add_argument("--version", required=True)
    v.add_argument("--artefacto", required=True)
    v.set_defaults(func=verificar)
    sub.add_parser("exportar-md").set_defaults(func=exportar_md)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
