# JustIA – Proyecto de Aplicación (IA Aplicada al Desarrollo de Software)
Stanly Leon Calle Samper · Uniasturias · Especialización en Desarrollo de Software

## Alcance
MVP del consultorio jurídico virtual JustIA para el reto de **clasificación automática de consultas por área del derecho**. Usa BETO (`dccuchile/bert-base-spanish-wwm-cased`) ajustado con los fragmentos de la Unidad 2 y 60 consultas coloquiales. Por cada consulta:

- verifica que el hash del modelo coincida con el registro de versiones de la Unidad 3,
- anonimiza documento, teléfono, correo, dirección y nombre,
- sugiere un área con su certeza y aplica el umbral de 0,70,
- activa reglas de urgencia y de violencia económica o patrimonial (Ley 1257 de 2008),
- explica la sugerencia por oclusión y guarda una traza en `trazas/trazas_justia.jsonl`.

El sistema solo sugiere: toda consulta pasa por un practicante. Todos los fragmentos, consultas, nombres y documentos son **simulados con fines académicos**, y el revisor del registro es un rol simulado.

Continúa los repositorios de la [Unidad 2](https://github.com/StanTheMackiar/just-ia-unidad2-uniasturias) y la [Unidad 3](https://github.com/StanTheMackiar/just-ia-unidad3-uniasturias).

## Cómo ejecutar
Requiere Python 3.10–3.12. El modelo ajustado (~420 MB) no se sube al repositorio, por lo que primero hay que entrenarlo.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python mvp/entrenar.py                     # ajusta BETO y lo evalúa (unos minutos)
python mvp/justia.py --demo                # 8 consultas de ejemplo
python mvp/justia.py "Él controla mi plata y me quitó la tarjeta"
python mvp/justia.py -i                    # modo interactivo ('salir' para terminar)
```

La versión 2.0.0 ya está en `gobernanza/control_versiones_modelos.csv` con el hash del modelo original. Un modelo reentrenado tiene otro hash y el MVP se niega a usarlo hasta que se registre como versión nueva: es la regla de la Unidad 3. Para registrarlo, cambie `VERSION` en `mvp/justia.py` y la versión en `gobernanza/registrar_version.sh`, y ejecute `bash gobernanza/registrar_version.sh`.
