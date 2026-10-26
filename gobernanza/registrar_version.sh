#!/usr/bin/env bash
# Registra BETO 2.0.0 en el control de versiones de la Unidad 3. Ejecutar desde la raiz tras mvp/entrenar.py.
set -e
R=gobernanza/registro_modelos.py
python $R registrar --id justia-clf-beto --version 2.0.0 --fecha 2026-10-26 \
  --objetivo clasificacion \
  --tecnica "Fine-tuning de BETO (dccuchile/bert-base-spanish-wwm-cased), 5 epocas, lr 3e-5, lote 8; umbral 0,70 + reglas + oclusion" \
  --corpus datos \
  --corpus-desc "196 fragmentos de entrenamiento (split U2, semilla 42) + 60 consultas coloquiales" \
  --artefacto modelos/beto_justia_v2.0.0 \
  --metricas "Prueba U2 (50): accuracy 0,88, F1 macro 0,873; prueba ciega (40): accuracy 0,75, F1 macro 0,733" \
  --riesgo "sesgo|falsos_positivos" \
  --detalle-riesgo "Administrativo coloquial falla (F1 0,33); en la prueba ciega 8 de 10 errores superan el umbral 0,70" \
  --validacion "Prueba U2 (50) + prueba ciega (40) + revision de errores; solo sugiere area, todo pasa por practicante" \
  --revisado-por "Docente revisor (simulado)" --estado aprobado_piloto
python $R listar
python $R verificar --id justia-clf-beto --version 2.0.0 --artefacto modelos/beto_justia_v2.0.0
python $R exportar-md
