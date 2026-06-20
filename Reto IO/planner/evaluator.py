"""Validacion de soluciones y calculo de la metrica de huecos.

`evaluate` recibe una Instance y un dict solucion ({Eventos, Reuniones, Guardias})
y devuelve un informe con: violaciones de restricciones duras y total de huecos.

Calculo de huecos: por cada (docente, dia) se toma el conjunto de posiciones
ocupadas (clases + reuniones + guardias) en la timeline ordenada del dia; los
huecos son la suma de duracion de las posiciones vacias ESTRICTAMENTE entre la
primera y la ultima ocupada. La duracion se mide en horas (recreo = 0.5).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .instance import Instance


@dataclass
class EvalReport:
    huecos: float = 0.0
    hard_violations: list[str] = field(default_factory=list)
    # huecos por docente para diagnostico
    huecos_por_docente: dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.hard_violations


def _slot_supports(inst: Instance, slot: str, length: int) -> bool:
    """True si una sesion de `length` slots puede empezar en `slot` sin cruzar recreo."""
    if slot not in inst.class_slots:
        return False
    return inst.class_slots.index(slot) in set(inst.valid_starts(length))


def _session_length_map(inst: Instance, eventos_sol: list[dict]) -> dict:
    """Empareja cada entrada de evento con la duracion de una de sus sesiones.

    La salida no codifica la duracion de cada entrada. El criterio canonico
    (consistente con el solver, que emite las sesiones en el orden de la
    `Distribucion Semanal`) es: la k-esima entrada de un evento corresponde a la
    k-esima duracion de su distribucion. Si ese emparejamiento por orden resultara
    en alguna colocacion invalida, se intenta un emparejamiento por slots como
    respaldo. Devuelve {indice_entrada: longitud_en_slots}.
    """
    by_event: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(eventos_sol):
        by_event[e["Evento Id"]].append(i)

    length_of_entry: dict[int, int] = {}
    for eid, idxs in by_event.items():
        u_idx = inst.event_to_unit.get(eid)
        lens = list(inst.units[u_idx].session_lengths) if u_idx is not None else [1] * len(idxs)
        while len(lens) < len(idxs):
            lens.append(1)
        slots = [str(eventos_sol[i]["Slot"]) for i in idxs]

        # emparejamiento canonico por orden de aparicion == orden de distribucion
        ordered = lens[:len(idxs)]
        if all(_slot_supports(inst, sl, L) for sl, L in zip(slots, ordered)):
            assignment = ordered
        else:
            assignment = _match_lengths(inst, slots, sorted(lens, reverse=True)) or ordered
        for entry_idx, L in zip(idxs, assignment):
            length_of_entry[entry_idx] = L
    return length_of_entry


def _match_lengths(inst: Instance, slots: list[str], lens: list[int]):
    """Respaldo: asigna longitudes (alineadas con `slots`) usando el multiset
    `lens` tal que cada slot admita su longitud. None si no existe."""
    n = len(slots)
    result = [None] * n
    used = [False] * len(lens)

    def bt(pos: int) -> bool:
        if pos == n:
            return True
        tried = set()
        for k, L in enumerate(lens):
            if used[k] or L in tried:
                continue
            tried.add(L)
            if _slot_supports(inst, slots[pos], L):
                used[k] = True
                result[pos] = L
                if bt(pos + 1):
                    return True
                used[k] = False
                result[pos] = None
        return False

    return result if bt(0) else None


def _covered_class_positions(inst: Instance, slot: str, length: int) -> list[int] | None:
    """Posiciones de timeline cubiertas por una sesion de `length` slots que
    empieza en `slot` (slot de clase). None si no cabe o cruza recreo."""
    if slot not in inst.class_slots:
        return None
    start_cs = inst.class_slots.index(slot)
    valid = set(inst.valid_starts(length))
    if start_cs not in valid:
        return None
    positions = []
    for k in range(length):
        cs_label = inst.class_slots[start_cs + k]
        positions.append(inst.pos_index[cs_label])
    return positions


def evaluate(inst: Instance, solution: dict) -> EvalReport:
    rep = EvalReport()
    V = rep.hard_violations

    eventos_sol = solution.get("Eventos", [])
    reuniones_sol = solution.get("Reuniones", [])
    guardias_sol = solution.get("Guardias", [])

    length_of_entry = _session_length_map(inst, eventos_sol)

    # ocupacion[(docente, dia)] -> set(pos_index)   (clases+reuniones+guardias)
    busy: dict[tuple[str, str], set[int]] = defaultdict(set)
    # claves de actividad por (grupo, dia, pos): dedupe por bloque
    group_occ: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    # claves de actividad de docente por (dia, pos): >1 distinta = conflicto.
    # Eventos del mismo bloque comparten clave (1 clase aunque sean varios grupos).
    teacher_act: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    # posiciones de clase (solo eventos) por (docente, dia) -> horas de clase
    teacher_event_pos: dict[tuple[str, str], set[int]] = defaultdict(set)
    # sesiones por (evento, dia) para chequear 1/dia
    ev_day_count: dict[tuple[str, str], int] = defaultdict(int)
    ev_total_count: dict[str, int] = defaultdict(int)
    # posicion de cada evento por entrada (para No_Coincidir y bloques)
    event_placements: dict[str, list[tuple[str, str]]] = defaultdict(list)  # eid -> [(dia,slot)]

    # ---------- EVENTOS ----------
    for i, e in enumerate(eventos_sol):
        eid = e["Evento Id"]
        dia = e["Dia semana"]
        slot = str(e["Slot"])
        if eid not in inst.eventos:
            V.append(f"Evento desconocido {eid}")
            continue
        if dia not in inst.days:
            V.append(f"{eid}: dia invalido {dia}")
            continue
        length = length_of_entry.get(i, 1)
        positions = _covered_class_positions(inst, slot, length)
        if positions is None:
            V.append(f"{eid}: slot {slot} invalido para sesion de {length} slot(s) "
                     f"(recreo o fuera de rejilla)")
            continue

        u = inst.units[inst.event_to_unit[eid]]
        ev = inst.eventos[eid]
        ev_day_count[(eid, dia)] += 1
        ev_total_count[eid] += 1
        event_placements[eid].append((dia, slot))

        # restriccion evento DURA
        if (eid, dia, slot) in inst.event_forbidden:
            V.append(f"{eid}: indisponibilidad DURA de evento en {dia} {slot}")

        bkey = u.block_id or f"ev_{eid}"
        t = ev["Docente Id"]
        for p in positions:
            busy[(t, dia)].add(p)
            teacher_act[(t, dia, p)].add(bkey)
            teacher_event_pos[(t, dia)].add(p)
            # DURA docente
            plabel = inst.positions[p][0]
            if (t, dia, plabel) in inst.teacher_forbidden:
                V.append(f"{eid}: docente {t} indisponible DURA en {dia} {plabel}")
            # grupo (dedupe por bloque: desdobles del mismo bloque comparten clave)
            group_occ[(ev["Grupo Id"], dia, p)].add(bkey)

    # ---------- REUNIONES ----------
    seen_reu = set()
    for r in reuniones_sol:
        rid = r["Reunion Id"]
        dia = r["Dia semana"]
        slot = str(r["Slot"])
        seen_reu.add(rid)
        if rid not in inst.reuniones:
            V.append(f"Reunion desconocida {rid}")
            continue
        if slot == inst.recreo_slot:
            V.append(f"{rid}: reunion en recreo (no permitido)")
            continue
        if slot not in inst.class_slots:
            V.append(f"{rid}: slot invalido {slot}")
            continue
        dur = inst.reuniones[rid]["dur"]
        positions = _covered_class_positions(inst, slot, dur)
        if positions is None:
            V.append(f"{rid}: reunion de {dur} slots no cabe en {slot}")
            continue
        for t in inst.reuniones[rid]["participantes"]:
            for p in positions:
                plabel = inst.positions[p][0]
                if (t, dia, plabel) in inst.teacher_forbidden:
                    V.append(f"{rid}: participante {t} indisponible DURA en {dia} {plabel}")
                busy[(t, dia)].add(p)
                teacher_act[(t, dia, p)].add(f"REU_{rid}")

    for rid in inst.reuniones:
        if rid not in seen_reu:
            V.append(f"Reunion {rid} no programada")

    # ---------- GUARDIAS ----------
    guard_count: dict[tuple[str, str], int] = defaultdict(int)   # (doc,tipo)->n
    demand_count: dict[tuple[str, str, str], int] = defaultdict(int)
    for g in guardias_sol:
        tipo = g["Tipo guardia"]
        t = g["Profesor Id"]
        dia = g["Dia semana"]
        slot = str(g["Slot"])
        if t not in inst.docentes:
            V.append(f"Guardia: docente desconocido {t}")
            continue
        if tipo == "Recreo":
            if slot != inst.recreo_slot:
                V.append(f"Guardia recreo de {t} fuera del slot de recreo ({slot})")
                continue
            p = inst.pos_index[slot]
        else:
            if slot not in inst.class_slots:
                V.append(f"Guardia {tipo} de {t} en slot invalido {slot}")
                continue
            p = inst.pos_index[slot]
        plabel = inst.positions[p][0]
        if (t, dia, plabel) in inst.teacher_forbidden:
            V.append(f"Guardia {tipo}: {t} indisponible DURA en {dia} {plabel}")
        busy[(t, dia)].add(p)
        teacher_act[(t, dia, p)].add(f"G_{tipo}")
        guard_count[(t, tipo)] += 1
        demand_count[(dia, slot, tipo)] += 1

    # cobertura de demanda de guardias por slot
    for (dia, slot, tipo), need in inst.guard_demand.items():
        got = demand_count.get((dia, slot, tipo), 0)
        if got != need:
            V.append(f"Demanda guardia {tipo} {dia} {slot}: requeridas {need}, asignadas {got}")
    # cupo semanal por docente: es un maximo (la demanda por slot es la obligatoria)
    for (doc, tipo), cap in inst.guard_quota.items():
        got = guard_count.get((doc, tipo), 0)
        if got > cap:
            V.append(f"Cupo guardia {tipo} de {doc}: maximo {cap}, asignadas {got}")

    # ---------- COBERTURA EVENTOS ----------
    for eid, u_idx in inst.event_to_unit.items():
        need = len(inst.units[u_idx].session_lengths)
        got = ev_total_count.get(eid, 0)
        if got != need:
            V.append(f"{eid}: {need} sesiones requeridas, {got} programadas")
    for (eid, dia), c in ev_day_count.items():
        if c > 1:
            V.append(f"{eid}: {c} sesiones el {dia} (max 1/dia)")

    # ---------- CONFLICTOS DOCENTE ----------
    for (t, dia, p), keys in teacher_act.items():
        if len(keys) > 1:
            plabel = inst.positions[p][0]
            V.append(f"Docente {t} con {len(keys)} actividades en {dia} {plabel}: {sorted(keys)}")

    # ---------- CONFLICTOS GRUPO (salvo mismo bloque) ----------
    for (grp, dia, p), keys in group_occ.items():
        # permitido solo si todas las actividades pertenecen al mismo bloque
        if len(keys) > 1:
            plabel = inst.positions[p][0]
            V.append(f"Grupo {grp} solapado en {dia} {plabel}: {sorted(keys)}")

    # ---------- CAPACIDAD AULA ----------
    # Las aulas estan pre-asignadas y no son un dato clave (el algoritmo no
    # asigna aulas); el enunciado lo indica y los datos reutilizan aulas comodin.
    # No se valida como restriccion dura.

    # ---------- MIN/MAX CLASES DIA ----------
    # horas de clase = nº de posiciones de clase ocupadas por eventos (1/hora);
    # los desdobles del mismo bloque ya estan deduplicados en teacher_event_pos.
    for doc_id, doc in inst.docentes.items():
        mn = int(doc.get("Minimo Clases Dia", 0) or 0)
        mx = int(doc.get("Maximo Clases Dia", 99) or 99)
        for dia in inst.days:
            ch = sum(1 for p in teacher_event_pos.get((doc_id, dia), ())
                     if not inst.positions[p][2])
            if ch == 0:
                continue
            if ch > mx:
                V.append(f"Docente {doc_id} {dia}: {ch}h clase > max {mx}")
            if ch < mn:
                V.append(f"Docente {doc_id} {dia}: {ch}h clase < min {mn}")

    # ---------- NO COINCIDIR ----------
    for a, b in inst.no_coincidir:
        pa = set(event_placements.get(a, []))
        pb = set(event_placements.get(b, []))
        inter = pa & pb
        if inter:
            V.append(f"No_Coincidir {a}/{b} coinciden en {sorted(inter)}")

    # ---------- BLOQUES EN PARALELO ----------
    for u in inst.units:
        if u.block_id is None or len(u.event_ids) < 2:
            continue
        placements = [sorted(event_placements.get(e, [])) for e in u.event_ids]
        ref = placements[0]
        for e, pl in zip(u.event_ids, placements):
            if pl != ref:
                V.append(f"Bloque {u.block_id}: evento {e} no paralelo al resto")
                break

    # ---------- HUECOS ----------
    total = 0  # en medias horas
    for (t, dia), positions in busy.items():
        if len(positions) < 2:
            continue
        lo, hi = min(positions), max(positions)
        gap_hh = 0
        for p in range(lo + 1, hi):
            if p not in positions:
                gap_hh += inst.positions[p][1]
        if gap_hh:
            total += gap_hh
            rep.huecos_por_docente[t] = rep.huecos_por_docente.get(t, 0.0) + gap_hh / 2.0
    rep.huecos = total / 2.0
    return rep
