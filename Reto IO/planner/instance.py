"""Carga y modelo de datos de una instancia del planificador de horarios.

Convierte el JSON de entrada en estructuras derivadas listas para el solver
y el evaluador: rejilla temporal, unidades (bloques/eventos), sesiones con su
duracion, ocupacion de recursos y restricciones duras.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Orden canonico de los dias de la semana (mañana, L-V)
CANON_DAYS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"]

# Tipos de guardia tal y como deben aparecer en la salida
GUARD_TYPES = ["Pasillo", "Recreo", "Convivencia"]


@dataclass
class Session:
    """Una sesion a programar de una unidad. `length` en numero de slots de clase."""
    unit_idx: int
    sess_idx: int       # indice de sesion dentro de la unidad
    length: int         # 1, 2 o 3 slots contiguos


@dataclass
class Unit:
    """Unidad programable: un bloque (eventos en paralelo) o un evento suelto.

    Todos los eventos miembros comparten la misma posicion (dia, slot inicial)
    para cada sesion. Ocupan en conjunto un set de docentes, grupos y aulas.
    """
    idx: int
    event_ids: list[str]
    block_id: Optional[str]
    session_lengths: list[int]          # duracion de cada sesion (en slots)
    teachers: set[str]                  # docentes implicados (cuenta 1 por unidad)
    groups: set[str]                    # grupos implicados
    # ocupacion de aula a nivel de evento: lista de (aula_id) por cada evento
    rooms: list[str] = field(default_factory=list)


@dataclass
class Instance:
    name: str
    raw: dict

    days: list[str]
    # timeline: orden temporal de posiciones dentro de un dia
    # cada posicion: (slot_label, half_hours, is_recreo)
    positions: list[tuple[str, int, bool]]
    class_slots: list[str]              # slots de clase, en orden temporal: ["1".."6"]
    recreo_slot: Optional[str]          # etiqueta del slot de recreo ("R")
    pos_index: dict[str, int]           # slot_label -> indice en `positions`

    docentes: dict[str, dict]
    grupos: dict[str, dict]
    aulas: dict[str, dict]
    asignaturas: dict[str, dict]
    eventos: dict[str, dict]

    units: list[Unit]
    event_to_unit: dict[str, int]

    # guardias
    guard_demand: dict[tuple[str, str, str], int]   # (dia, slot, tipo) -> demanda
    guard_quota: dict[tuple[str, str], int]         # (docente, tipo) -> cupo semanal

    # reuniones
    reuniones: dict[str, dict]                       # id -> {dur, participantes}

    # restricciones duras
    # docente no disponible en (dia, slot) -> set
    teacher_forbidden: set[tuple[str, str, str]]
    # evento no puede ir en (dia, slot)
    event_forbidden: set[tuple[str, str, str]]
    # pares de eventos que no pueden coincidir en mismo (dia, slot)
    no_coincidir: list[tuple[str, str]]
    # preferencias blandas docente: (docente, dia, slot, nivel)
    teacher_soft: list[tuple[str, str, str, str]]
    event_soft: list[tuple[str, str, str, str]]

    def valid_starts(self, length: int) -> list[int]:
        """Indices de class_slots donde puede empezar una sesion de `length`
        slots sin cruzar el recreo (bloques mañana / tarde separados)."""
        # construir bloques contiguos de class_slots segun la timeline real
        blocks = self._contiguous_blocks()
        starts = []
        for blk in blocks:
            for i in range(len(blk) - length + 1):
                starts.append(blk[i])
        return starts

    def _contiguous_blocks(self) -> list[list[int]]:
        """Agrupa indices de class_slots en tramos sin recreo intermedio."""
        blocks = []
        cur = []
        for p_idx, (label, _hh, is_rec) in enumerate(self.positions):
            if is_rec:
                if cur:
                    blocks.append(cur)
                    cur = []
            else:
                cur.append(self.class_slots.index(label))
        if cur:
            blocks.append(cur)
        return blocks


def _parse_distribution(dist) -> list[int]:
    """'60,60,120' -> [1,1,2] (numero de slots por sesion)."""
    parts = [p.strip() for p in str(dist).split(",") if str(p).strip()]
    out = []
    for p in parts:
        minutes = int(float(p))
        out.append(max(1, round(minutes / 60)))
    return out


def load_instance(path: str) -> Instance:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    name = path.replace("\\", "/").split("/")[-1]

    # ---- Timeline ----
    slots = raw["Slots"]
    days_present = {s["Dia"] for s in slots}
    days = [d for d in CANON_DAYS if d in days_present]
    # añadir cualquier dia no canonico al final por robustez
    for d in sorted(days_present):
        if d not in days:
            days.append(d)

    # ordenar posiciones de un dia por hora de inicio (usamos el primer dia)
    sample_day = days[0]
    day_slots = [s for s in slots if s["Dia"] == sample_day]
    day_slots.sort(key=lambda s: s["Inicio"])

    positions: list[tuple[str, int, bool]] = []
    class_slots: list[str] = []
    recreo_slot = None
    for s in day_slots:
        is_rec = str(s["Tipo"]).lower().startswith("recreo")
        half_hours = _slot_half_hours(s)
        positions.append((s["Slot"], half_hours, is_rec))
        if is_rec:
            recreo_slot = s["Slot"]
        else:
            class_slots.append(s["Slot"])
    pos_index = {lbl: i for i, (lbl, _, _) in enumerate(positions)}

    docentes = {d["Docente Id"]: d for d in raw["Docentes"]}
    grupos = {g["Grupo Id"]: g for g in raw["Grupos"]}
    aulas = {a["Aula Id"]: a for a in raw["Aulas"]}
    asignaturas = {a["Asignatura Id"]: a for a in raw["Asignaturas"]}
    eventos = {e["Evento Id"]: e for e in raw["Eventos"]}

    # ---- Unidades (bloques + eventos sueltos) ----
    block_members: dict[str, list[str]] = defaultdict(list)
    for m in raw.get("Miembros_Bloque", []):
        block_members[m["Bloque Id"]].append(m["Evento Id"])

    evid_in_block = {eid for evs in block_members.values() for eid in evs}

    units: list[Unit] = []
    event_to_unit: dict[str, int] = {}

    def session_lengths_for(event_id: str) -> list[int]:
        ev = eventos[event_id]
        asg = asignaturas.get(ev["Asignatura Id"])
        if not asg:
            return [1]
        return _parse_distribution(asg["Distribucion Semanal"])

    # bloques
    for block_id, evids in block_members.items():
        evids = [e for e in evids if e in eventos]
        if not evids:
            continue
        # plantilla de sesiones: la del primer miembro (homogeneos por construccion)
        sess_lens = session_lengths_for(evids[0])
        teachers = {eventos[e]["Docente Id"] for e in evids}
        groups = {eventos[e]["Grupo Id"] for e in evids}
        rooms = [eventos[e].get("Aula Id", "") for e in evids]
        u = Unit(idx=len(units), event_ids=evids, block_id=block_id,
                 session_lengths=sess_lens, teachers=teachers,
                 groups=groups, rooms=rooms)
        for e in evids:
            event_to_unit[e] = u.idx
        units.append(u)

    # eventos sueltos
    for eid, ev in eventos.items():
        if eid in evid_in_block:
            continue
        u = Unit(idx=len(units), event_ids=[eid], block_id=None,
                 session_lengths=session_lengths_for(eid),
                 teachers={ev["Docente Id"]}, groups={ev["Grupo Id"]},
                 rooms=[ev.get("Aula Id", "")])
        event_to_unit[eid] = u.idx
        units.append(u)

    # ---- Guardias ----
    guard_demand: dict[tuple[str, str, str], int] = {}
    for s in slots:
        for tipo, key in (("Pasillo", "Demanda Guardias Pasillo"),
                          ("Recreo", "Demanda Guardias Recreo"),
                          ("Convivencia", "Demanda Guardias Convivencia")):
            v = int(s.get(key, 0) or 0)
            if v:
                guard_demand[(s["Dia"], s["Slot"], tipo)] = v

    guard_quota: dict[tuple[str, str], int] = {}
    for doc_id, doc in docentes.items():
        for tipo, key in (("Pasillo", "Guardias Pasillo Semana"),
                          ("Recreo", "Guardias Recreo Semana"),
                          ("Convivencia", "Guardias Convivencia Semana")):
            v = int(doc.get(key, 0) or 0)
            if v:
                guard_quota[(doc_id, tipo)] = v

    # ---- Reuniones ----
    participantes: dict[str, list[str]] = defaultdict(list)
    for p in raw.get("Participantes_Reunion", []):
        participantes[p["Reunion Id"]].append(p["Docente Id"])
    reuniones = {}
    for r in raw.get("Reuniones", []):
        rid = r["Reunion Id"]
        reuniones[rid] = {
            "dur": int(r.get("Duracion Slots", 1) or 1),
            "participantes": participantes.get(rid, []),
        }

    # ---- Restricciones duras / blandas ----
    teacher_forbidden: set[tuple[str, str, str]] = set()
    teacher_soft: list[tuple[str, str, str, str]] = []
    for x in raw.get("Indisponibilidad", []) + raw.get("Conciliacion", []):
        key = (x["Docente Id"], x["Dia"], str(x["Slot"]))
        if str(x.get("Tipo Restriccion", "")).upper() == "DURA":
            teacher_forbidden.add(key)
        else:
            teacher_soft.append((x["Docente Id"], x["Dia"], str(x["Slot"]),
                                 str(x.get("Tipo Restriccion", "")).upper()))

    event_forbidden: set[tuple[str, str, str]] = set()
    event_soft: list[tuple[str, str, str, str]] = []
    for x in raw.get("Indisponibilidad_Eventos", []):
        key = (x["Evento Id"], x["Dia"], str(x["Slot"]))
        if str(x.get("Tipo Restriccion", "")).upper() == "DURA":
            event_forbidden.add(key)
        else:
            event_soft.append((x["Evento Id"], x["Dia"], str(x["Slot"]),
                               str(x.get("Tipo Restriccion", "")).upper()))

    no_coincidir = []
    for x in raw.get("No_Coincidir", []):
        a, b = x.get("Evento Id_1"), x.get("Evento Id_2")
        if a and b and a != b:
            no_coincidir.append((a, b))

    return Instance(
        name=name, raw=raw, days=days, positions=positions,
        class_slots=class_slots, recreo_slot=recreo_slot, pos_index=pos_index,
        docentes=docentes, grupos=grupos, aulas=aulas,
        asignaturas=asignaturas, eventos=eventos,
        units=units, event_to_unit=event_to_unit,
        guard_demand=guard_demand, guard_quota=guard_quota,
        reuniones=reuniones,
        teacher_forbidden=teacher_forbidden, event_forbidden=event_forbidden,
        no_coincidir=no_coincidir,
        teacher_soft=teacher_soft, event_soft=event_soft,
    )


def _slot_half_hours(slot: dict) -> int:
    """Duracion del slot en medias horas (entero), a partir de Inicio/Fin."""
    try:
        h1, m1 = map(int, str(slot["Inicio"]).split(":"))
        h2, m2 = map(int, str(slot["Fin"]).split(":"))
        minutes = (h2 * 60 + m2) - (h1 * 60 + m1)
        return max(1, round(minutes / 30))
    except Exception:
        # por defecto: clase=2 (1h), recreo=1 (30min)
        return 1 if str(slot.get("Tipo", "")).lower().startswith("recreo") else 2
