# Reto IO — Planificador de horarios (OGATHON 5.0)

Genera el horario semanal de un instituto colocando **eventos** (clases),
**reuniones** y **guardias** en la rejilla de slots de mañana (L–V, slots 1–6 +
recreo `R`), cumpliendo todas las restricciones duras y **minimizando el total de
huecos del profesorado**.

## Enfoque

Modelo de **programación con restricciones** (CP-SAT de Google OR-Tools) con una
estrategia de resolución multifase.

- **Unidad** = un bloque (eventos en paralelo: desdobles/optativas) o un evento
  suelto. Las unidades colocan cada una de sus sesiones en `(día, slot inicial)`.
  Tratar el bloque como una unidad resuelve de forma natural el caso "un profesor
  imparte a varios grupos a la vez = una sola clase".
- **Sesiones** de 1, 2 o 3 slots de clase consecutivos (60/120/180 min). Los slots
  1–6 forman una secuencia contigua; una sesión multi-slot puede dejar el recreo en
  medio (computa como 0,5 h de hueco, así que el optimizador lo evita salvo que el
  empaquetado lo exija).
- **Objetivo**: huecos linealizados. Una posición vacía cuenta como hueco si tiene
  actividad antes y después del profesor ese día (recreo = 0,5 h).

### Estrategia de resolución (multifase)
Los grupos de ESO tienen el horario completo (30 h = 30 slots), por lo que la rejilla
está saturada y la optimización conjunta se estanca. Se combinan:

1. **Factibilidad**: se busca una solución válida sin objetivo (rápida).
2. **Optimización conjunta** (día + slot) con esa solución como pista.
3. **Compactación intradía**: como los huecos se computan por día, se fija la
   asignación de día de cada actividad y se re-optimiza solo el slot dentro del día
   (subproblema mucho más fácil → compacta cada jornada casi a óptimo).
4. Se **alternan** los pasos 2 y 3 en varias rondas, conservando siempre la mejor
   solución factible. La alternancia mejora la asignación de días (paso 2) y luego
   la compacta (paso 3), reduciendo los huecos de forma sostenida.

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
