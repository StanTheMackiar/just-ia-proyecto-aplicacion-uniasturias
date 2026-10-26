# Control de versiones de modelos JustIA

| id_modelo | version | fecha_entrenamiento | objetivo | corpus_fuente | riesgo_potencial | validacion_previa | estado |
|---|---|---|---|---|---|---|---|
| justia-ner | 1.0.0 | 2026-10-09 | extraccion_entidades | textos_ner.json: 10 sentencias y resoluciones simuladas | falsos_negativos, privacidad | Comparacion contra anotacion manual; revision de las visualizaciones de displacy | aprobado_piloto |
| justia-qa | 0.9.0 | 2026-10-09 | preguntas_respuestas | base_conocimiento: 8 resumenes de normas colombianas (30 pasajes) | alucinacion, sesgo | 10 preguntas de prueba con valoracion manual | rechazado |
| justia-qa | 1.0.0 | 2026-10-10 | preguntas_respuestas | base_conocimiento: 8 resumenes de normas colombianas (30 pasajes) | alucinacion, falsos_positivos | Pendiente umbral para el lector y revision de las respuestas por un docente de Derecho | en_revision |
| justia-clf-beto | 1.0.0 | 2026-10-11 | clasificacion | fragmentos_juridicos.csv: 246 fragmentos simulados, 8 areas (72 consultas coloquiales) | sesgo, falsos_positivos | Reporte por clase con 50 fragmentos de prueba, revision de los 10 errores y 5 frases nuevas. Aprobado solo como sugerencia: lo que quede bajo 70 % va a un practicante | aprobado_piloto |
| justia-clf-beto | 1.1.0 | 2026-10-18 | clasificacion | 246 fragmentos + 60 consultas penales y de genero anonimizadas (simulado, aun sin archivo) | sesgo, opacidad | Pendiente: prueba de regresion contra 1.0.0 y 30 casos revisados por docentes | en_revision |
| justia-clf-beto | 2.0.0 | 2026-10-26 | clasificacion | 196 fragmentos de entrenamiento (split U2, semilla 42) + 60 consultas coloquiales | sesgo, falsos_positivos | Prueba U2 (50) + prueba ciega (40) + revision de errores; solo sugiere area, todo pasa por practicante | aprobado_piloto |
