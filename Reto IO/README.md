# Reto IO — Planificador de horarios (OGATHON 5.0)

Genera el horario semanal de un instituto colocando **eventos** (clases),
**reuniones** y **guardias** en la rejilla de slots de mañana (L–V, slots 1–6 +
recreo `R`), cumpliendo todas las restricciones duras y **minimizando el total de
huecos del profesorado**.

## Enfoque

Modelo de **programación con restricciones** (CP-SAT de Google OR-Tools).

- **Unidad** = un bloque (eventos en paralelo: desdobles/optativas) o un evento
  suelto. Las unidades colocan cada una de sus sesiones en `(día, slot inicial)`.
  Tratar el bloque como una unidad resuelve de forma natural el caso "un profesor
  imparte a varios grupos a la vez = una sola clase".
- **Sesiones** de 1, 2 o 3 slots contiguos (60/120/180 min) que no cruzan el recreo.
- **Objetivo**: huecos linealizados. Una posición vacía cuenta como hueco si tiene
  actividad antes y después del profesor ese día (recreo = 0,5 h).

### Restricciones duras modeladas
- Cobertura total de eventos (nº de sesiones según `Distribución Semanal`), reuniones y guardias.
- Máx. 1 sesión/día de una misma actividad.
- No solape de **docente** y **grupo** por slot (excepto paralelos del mismo bloque).
- Demanda de guardias por slot **y** cupo semanal por docente (exactos).
- Reuniones compatibles con todos los participantes; nunca en recreo.
- Indisponibilidades **DURA** (docente y evento) y `No_Coincidir`.
- Bloques en paralelo.
- Mín/máx horas de clase al día por docente (semicontinuo: 0 ó dentro de `[min,max]`).

> **Aulas**: el enunciado indica que las aulas están pre-asignadas y no son un dato
> clave (el algoritmo no las asigna). Los datos reutilizan aulas comodín entre grupos,
> por lo que la capacidad de aula **no** se trata como restricción dura.

## Uso

```bash
py run.py dataset_smartiming/2122_small_4_grupos_simplified.json --time 60   # una instancia
py run.py --all --time 300                                                   # todas + entrega.zip
py run.py --eval soluciones/<inst>.json dataset_smartiming/<inst>.json       # validar una salida
```

La salida se escribe en `soluciones/` con **el mismo nombre que la entrada** y se
empaqueta en `entrega.zip`.

## Estructura
- `planner/instance.py` — carga y modelo de datos derivado.
- `planner/solver.py` — modelo CP-SAT y extracción de la solución.
- `planner/evaluator.py` — validación de restricciones duras y cálculo de huecos.
- `run.py` — CLI / generación de la entrega.
