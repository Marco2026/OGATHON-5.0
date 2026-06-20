"""Solver CP-SAT del planificador de horarios.

Modelo unidad-sesion:
 - cada unidad (bloque o evento suelto) coloca sus sesiones en (dia, slot inicial)
 - restricciones duras: cobertura, 1 sesion/dia por unidad, no solape de docente/
   grupo/aula, indisponibilidades DURA, No_Coincidir, bloques en paralelo,
   demanda y cupo de guardias, reuniones compatibles, min/max horas clase/dia
 - objetivo: minimizar el total de huecos del profesorado (clases+reuniones+guardias)
"""
from __future__ import annotations

from collections import defaultdict

from ortools.sat.python import cp_model

from .instance import Instance, GUARD_TYPES


def solve(inst: Instance, time_limit: float = 60.0, workers: int = 8,
          log: bool = False, enable: dict | None = None,
          optimize: bool = True, hint: dict | None = None,
          fix_solution: dict | None = None,
          free_teachers: set | None = None) -> tuple[dict, dict]:
    """Resuelve la instancia. Devuelve (solucion_dict, info).

    `enable` permite desactivar grupos de restricciones para diagnostico:
    {guardias, reuniones, minmax, nocoincidir, aula, grupo} (por defecto todo True).
    """
    # Nota: la restriccion de aula se desactiva por defecto. El enunciado indica
    # que las aulas estan pre-asignadas y no son un dato clave (el algoritmo no
    # asigna aulas); ademas los datos contienen aulas comodin reutilizadas por
    # varios grupos, lo que haria el modelo infactible.
    EN = {"guardias": True, "reuniones": True, "minmax": True,
          "nocoincidir": True, "aula": False, "grupo": True, "dura": True}
    if enable:
        EN.update(enable)
    m = cp_model.CpModel()
    P = len(inst.positions)
    days = inst.days
    half = [inst.positions[p][1] for p in range(P)]   # duracion en medias horas
    rec_pos = inst.pos_index.get(inst.recreo_slot) if inst.recreo_slot else None

    # ---------- placement vars de eventos: x[u,s][(d,cs)] ----------
    # cs = indice en class_slots del slot inicial
    x: dict[tuple[int, int], dict[tuple[str, int], cp_model.IntVar]] = {}
    # unitocc[u][(d,p)] = 1 si la unidad u ocupa la posicion p el dia d
    unit_cover: dict[tuple[int, str, int], list] = defaultdict(list)

    def teacher_forbidden_at(t, d, p):
        return (t, d, inst.positions[p][0]) in inst.teacher_forbidden

    for u in inst.units:
        for s, L in enumerate(u.session_lengths):
            key = (u.idx, s)
            x[key] = {}
            for d in days:
                for cs in inst.valid_starts(L):
                    # posiciones cubiertas
                    covered = [inst.pos_index[inst.class_slots[cs + k]] for k in range(L)]
                    # prohibir por DURA de docente o de evento
                    ok = True
                    if EN["dura"]:
                        for p in covered:
                            for t in u.teachers:
                                if teacher_forbidden_at(t, d, p):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            start_label = inst.class_slots[cs]
                            for eid in u.event_ids:
                                if (eid, d, start_label) in inst.event_forbidden:
                                    ok = False
                                    break
                    if not ok:
                        continue
                    var = m.NewBoolVar(f"x_u{u.idx}_s{s}_{d}_{cs}")
                    x[key][(d, cs)] = var
                    for p in covered:
                        unit_cover[(u.idx, d, p)].append(var)

            # A. cada sesion exactamente una vez
            if x[key]:
                m.Add(sum(x[key].values()) == 1)
            else:
                # sin placement factible -> instancia infactible para esta sesion
                raise RuntimeError(f"Sesion {s} de unidad {u.idx} sin colocacion factible")

        # B. <=1 sesion/dia por unidad
        for d in days:
            vars_d = []
            for s in range(len(u.session_lengths)):
                for (dd, cs), v in x[(u.idx, s)].items():
                    if dd == d:
                        vars_d.append(v)
            if len(vars_d) > 1:
                m.Add(sum(vars_d) <= 1)

    # unitocc como bool (0/1): suma de coberturas (<=1 por B + 1 sesion)
    unitocc: dict[tuple[int, str, int], cp_model.IntVar] = {}
    for (uidx, d, p), vlist in unit_cover.items():
        b = m.NewBoolVar(f"uocc_{uidx}_{d}_{p}")
        m.Add(b == sum(vlist))
        unitocc[(uidx, d, p)] = b

    def uocc(uidx, d, p):
        return unitocc.get((uidx, d, p))

    # ---------- reuniones: y[rid][(d,cs)] ----------
    y: dict[str, dict[tuple[str, int], cp_model.IntVar]] = {}
    reu_cover_teacher: dict[tuple[str, str, int], list] = defaultdict(list)
    for rid, info in (inst.reuniones.items() if EN["reuniones"] else []):
        dur = info["dur"]
        parts = info["participantes"]
        y[rid] = {}
        for d in days:
            for cs in inst.valid_starts(dur):
                covered = [inst.pos_index[inst.class_slots[cs + k]] for k in range(dur)]
                ok = all(not teacher_forbidden_at(t, d, p)
                         for t in parts for p in covered)
                if not ok:
                    continue
                var = m.NewBoolVar(f"y_{rid}_{d}_{cs}")
                y[rid][(d, cs)] = var
                for t in parts:
                    for p in covered:
                        reu_cover_teacher[(t, d, p)].append(var)
        if y[rid]:
            m.Add(sum(y[rid].values()) == 1)
        else:
            raise RuntimeError(f"Reunion {rid} sin colocacion factible")

    # ---------- guardias: g[(t,d,slot,tipo)] ----------
    g: dict[tuple[str, str, str, str], cp_model.IntVar] = {}
    guard_cover_teacher: dict[tuple[str, str, int], list] = defaultdict(list)
    # por tipo, slots candidatos
    for (doc, tipo), quota in (inst.guard_quota.items() if EN["guardias"] else []):
        for d in days:
            for (slabel, hh, is_rec) in inst.positions:
                if tipo == "Recreo" and not is_rec:
                    continue
                if tipo != "Recreo" and is_rec:
                    continue
                # debe existir demanda en ese slot/tipo
                if (d, slabel, tipo) not in inst.guard_demand:
                    continue
                p = inst.pos_index[slabel]
                if teacher_forbidden_at(doc, d, p):
                    continue
                var = m.NewBoolVar(f"g_{doc}_{d}_{slabel}_{tipo}")
                g[(doc, d, slabel, tipo)] = var
                guard_cover_teacher[(doc, d, p)].append(var)

    # demanda por slot exacta
    for (d, slabel, tipo), need in (inst.guard_demand.items() if EN["guardias"] else []):
        vs = [g[(doc, d, slabel, tipo)] for (doc, t2) in inst.guard_quota
              if t2 == tipo and (doc, d, slabel, tipo) in g]
        if vs:
            m.Add(sum(vs) == need)
        elif need > 0:
            raise RuntimeError(f"Demanda guardia {tipo} {d} {slabel} sin candidatos")
    # cupo semanal por docente como MAXIMO (la demanda por slot es la obligatoria).
    # En los datos la demanda total puede ser < cupo total (p.ej. Pasillo 111 vs 112),
    # por lo que forzar igualdad en ambos seria infactible. Cuando demanda==cupo
    # (Convivencia, Recreo) el <= se vuelve igualdad de forma automatica.
    for (doc, tipo), quota in (inst.guard_quota.items() if EN["guardias"] else []):
        vs = [g[k] for k in g if k[0] == doc and k[3] == tipo]
        if vs:
            m.Add(sum(vs) <= quota)

    # ---------- ocupacion por docente / grupo / aula ----------
    # mapear unidades por docente y grupo
    units_of_teacher: dict[str, list[int]] = defaultdict(list)
    units_of_group: dict[str, list[int]] = defaultdict(list)
    # aulas: (room) -> list of (uidx) one per evento usando room
    room_units: dict[str, list[int]] = defaultdict(list)
    for u in inst.units:
        for t in u.teachers:
            units_of_teacher[t].append(u.idx)
        for grp in u.groups:
            units_of_group[grp].append(u.idx)
        for r in u.rooms:
            if r:
                room_units[r].append(u.idx)

    # busy[t,d,p] y restriccion de ocupacion unica del docente
    busy: dict[tuple[str, str, int], cp_model.IntVar] = {}
    for t in inst.docentes:
        for d in days:
            for p in range(P):
                terms = []
                for uidx in units_of_teacher.get(t, []):
                    b = uocc(uidx, d, p)
                    if b is not None:
                        terms.append(b)
                terms += reu_cover_teacher.get((t, d, p), [])
                terms += guard_cover_teacher.get((t, d, p), [])
                if not terms:
                    continue
                bv = m.NewBoolVar(f"busy_{t}_{d}_{p}")
                m.Add(bv == sum(terms))   # fuerza <=1 (ocupacion unica)
                busy[(t, d, p)] = bv

    # grupo: <=1 unidad por (grupo,d,p) (bloques ya unificados en una unidad)
    for grp, uidxs in (units_of_group.items() if EN["grupo"] else []):
        for d in days:
            for p in range(P):
                terms = [uocc(u, d, p) for u in set(uidxs) if uocc(u, d, p) is not None]
                if len(terms) > 1:
                    m.Add(sum(terms) <= 1)

    # aula: capacidad por (room,d,p)
    for room, uidxs in (room_units.items() if EN["aula"] else []):
        aula = inst.aulas.get(room)
        if aula and aula.get("Compartible"):
            cap = int(aula.get("Capacidad", 1) or 1)
        else:
            cap = 1
        for d in days:
            for p in range(P):
                # cada evento que usa room aporta 1 si su unidad ocupa (d,p)
                terms = []
                for u in inst.units:
                    cnt = sum(1 for r in u.rooms if r == room)
                    if cnt and uocc(u.idx, d, p) is not None:
                        terms.append((cnt, uocc(u.idx, d, p)))
                if terms:
                    m.Add(sum(c * v for c, v in terms) <= cap)

    # ---------- min/max horas clase por dia ----------
    for doc_id, doc in (inst.docentes.items() if EN["minmax"] else []):
        mn = int(doc.get("Minimo Clases Dia", 0) or 0)
        mx = int(doc.get("Maximo Clases Dia", 99) or 99)
        mx = min(mx, len(inst.class_slots))
        for d in days:
            terms = []
            for uidx in units_of_teacher.get(doc_id, []):
                for p in range(P):
                    if inst.positions[p][2]:   # recreo no es clase
                        continue
                    b = uocc(uidx, d, p)
                    if b is not None:
                        terms.append(b)
            if not terms:
                continue
            ch = sum(terms)
            if mn <= 0:
                m.Add(ch <= mx)
            else:
                works = m.NewBoolVar(f"works_{doc_id}_{d}")
                m.Add(ch <= mx * works)
                m.Add(ch >= mn * works)

    # ---------- No_Coincidir ----------
    for a, b in (inst.no_coincidir if EN["nocoincidir"] else []):
        ua = inst.event_to_unit.get(a)
        ub = inst.event_to_unit.get(b)
        if ua is None or ub is None:
            continue
        for d in days:
            for p in range(P):
                va = uocc(ua, d, p)
                vb = uocc(ub, d, p)
                if va is not None and vb is not None:
                    m.Add(va + vb <= 1)

    # ---------- objetivo: huecos ----------
    # Para cada docente-dia se recorre TODA la timeline. Una posicion es hueco si
    # esta vacia y tiene actividad antes y despues ese dia. Las posiciones donde el
    # docente no puede tener actividad se tratan como vacias constantes (busy=0).
    obj_terms = []
    for t in (inst.docentes if optimize else []):
        for d in days:
            # expresion busy por posicion (var o constante 0)
            bexpr = [busy.get((t, d, p), 0) for p in range(P)]
            # si el docente no tiene ninguna variable este dia, no hay huecos
            if not any(not isinstance(b, int) for b in bexpr):
                continue
            for p in range(1, P - 1):
                before = [bexpr[q] for q in range(p) if not isinstance(bexpr[q], int)]
                after = [bexpr[q] for q in range(p + 1, P) if not isinstance(bexpr[q], int)]
                if not before or not after:
                    continue
                hb = m.NewBoolVar(f"hb_{t}_{d}_{p}")
                ha = m.NewBoolVar(f"ha_{t}_{d}_{p}")
                for bb in before:
                    m.Add(hb >= bb)
                for aa in after:
                    m.Add(ha >= aa)
                gap = m.NewBoolVar(f"gap_{t}_{d}_{p}")
                m.Add(gap >= hb + ha - bexpr[p] - 1)
                obj_terms.append(half[p] * gap)

    if optimize:
        m.Minimize(sum(obj_terms))

    # ---------- pista (warm start) desde una solucion previa ----------
    if hint:
        _apply_hint(m, inst, hint, x, y, g)

    # ---------- LNS: fijar el complemento del vecindario ----------
    # Fija a su valor actual toda actividad que NO involucre a `free_teachers`,
    # dejando libres solo los horarios de esos docentes para re-optimizarlos.
    if fix_solution is not None and free_teachers == "DAYS":
        _fix_days(m, inst, fix_solution, x, y, g)
    elif fix_solution is not None and free_teachers is not None:
        _fix_complement(m, inst, fix_solution, free_teachers, x, y, g)

    # ---------- resolver ----------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.log_search_progress = bool(log)
    status = solver.Solve(m)

    info = {
        "status": solver.StatusName(status),
        "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "best_bound": solver.BestObjectiveBound() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "wall_time": solver.WallTime(),
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"Eventos": [], "Reuniones": [], "Guardias": []}, info

    # ---------- extraer solucion ----------
    sol = {"Eventos": [], "Reuniones": [], "Guardias": []}
    for u in inst.units:
        for s, L in enumerate(u.session_lengths):
            for (d, cs), v in x[(u.idx, s)].items():
                if solver.Value(v):
                    slabel = inst.class_slots[cs]
                    for eid in u.event_ids:
                        sol["Eventos"].append({
                            "Evento Id": eid, "Dia semana": d, "Slot": slabel})
    for rid, placements in y.items():
        for (d, cs), v in placements.items():
            if solver.Value(v):
                sol["Reuniones"].append({
                    "Reunion Id": rid, "Dia semana": d,
                    "Slot": inst.class_slots[cs]})
    for (doc, d, slabel, tipo), v in g.items():
        if solver.Value(v):
            sol["Guardias"].append({
                "Tipo guardia": tipo, "Profesor Id": doc,
                "Dia semana": d, "Slot": slabel})
    return sol, info


def lns_improve(inst: Instance, sol: dict, total_time: float, workers: int = 12,
                k: int = 8, iter_time: float = 20.0, seed: int = 0,
                verbose: bool = True) -> tuple[dict, float]:
    """Mejora una solucion factible mediante LNS dirigida: en cada iteracion
    libera los horarios de un vecindario de docentes (un docente con muchos huecos
    y los que comparten grupo con el) y deja que CP-SAT los re-optimice fijando el
    resto. Conserva la mejor solucion factible encontrada."""
    import time
    from .evaluator import evaluate

    rep = evaluate(inst, sol)
    best, best_h = sol, rep.huecos
    if not rep.feasible:
        return best, best_h

    # adyacencia docente-docente por grupos compartidos
    group_teachers: dict[str, set] = {}
    for u in inst.units:
        for grp in u.groups:
            group_teachers.setdefault(grp, set()).update(u.teachers)
    adj: dict[str, set] = {t: set() for t in inst.docentes}
    for grp, ts in group_teachers.items():
        for t in ts:
            adj[t].update(ts)

    rng_state = seed
    def rnd(n):
        nonlocal rng_state
        rng_state = (rng_state * 1103515245 + 12345) & 0x7fffffff
        return rng_state % n

    t0 = time.time()
    it = 0
    while time.time() - t0 < total_time:
        it += 1
        hp = evaluate(inst, best).huecos_por_docente
        ranked = sorted(inst.docentes, key=lambda t: -hp.get(t, 0.0))
        # semilla entre los de mas huecos (con algo de aleatoriedad)
        seed_t = ranked[rnd(min(10, len(ranked)))]
        nbrs = list(adj[seed_t])
        # vecindario: semilla + vecinos con mas huecos, hasta k
        nbrs.sort(key=lambda t: -hp.get(t, 0.0))
        free = set([seed_t] + nbrs[:max(0, k - 1)])
        remaining = total_time - (time.time() - t0)
        it_t = min(iter_time, remaining)
        if it_t < 3:
            break
        sol2, info = solve(inst, time_limit=it_t, workers=workers, optimize=True,
                           hint=best, fix_solution=best, free_teachers=free)
        r2 = evaluate(inst, sol2)
        if r2.feasible and r2.huecos < best_h - 1e-9:
            if verbose:
                print(f"    LNS it{it}: {best_h} -> {r2.huecos} (semilla {seed_t}, |N|={len(free)})")
            best, best_h = sol2, r2.huecos
    return best, best_h


def _fix_days(m, inst: Instance, sol: dict, x, y, g) -> None:
    """Fija la asignacion de DIA de cada actividad (sesiones, reuniones, guardias)
    a la de `sol`, dejando libre solo el SLOT dentro de ese dia. Como los huecos se
    computan por dia, esto re-optimiza la distribucion intradia (subproblema mucho
    mas facil que la asignacion conjunta dia+slot)."""
    from collections import defaultdict
    # dias de cada sesion por unidad (por orden de distribucion)
    unit_slots: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for e in sol.get("Eventos", []):
        u_idx = inst.event_to_unit.get(e["Evento Id"])
        if u_idx is None:
            continue
        if e["Evento Id"] != inst.units[u_idx].event_ids[0]:
            continue
        unit_slots[u_idx].append((e["Dia semana"], str(e["Slot"])))

    for u in inst.units:
        lens = list(u.session_lengths)
        used = [False] * len(lens)
        for (d, slot) in unit_slots.get(u.idx, []):
            if slot not in inst.class_slots:
                continue
            cs = inst.class_slots.index(slot)
            for k, L in enumerate(lens):
                if used[k]:
                    continue
                if cs in set(inst.valid_starts(L)):
                    # mantener sesion k en el dia d, slot libre
                    same_day = [v for (dd, cc), v in x[(u.idx, k)].items() if dd == d]
                    if same_day:
                        m.Add(sum(same_day) == 1)
                        used[k] = True
                    break

    # reuniones: mantener dia, slot libre
    cur_reu = defaultdict(list)
    for r in sol.get("Reuniones", []):
        cur_reu[r["Reunion Id"]].append(r["Dia semana"])
    for rid, dias in cur_reu.items():
        for d in dias:
            same_day = [v for (dd, cc), v in y.get(rid, {}).items() if dd == d]
            if same_day:
                m.Add(sum(same_day) == 1)

    # guardias: mantener (docente, dia, tipo) y su numero ese dia; slot libre
    cnt = defaultdict(int)   # (doc, dia, tipo) -> nº ese dia
    for gd in sol.get("Guardias", []):
        cnt[(gd["Profesor Id"], gd["Dia semana"], gd["Tipo guardia"])] += 1
    by_ddt = defaultdict(list)
    for key, var in g.items():
        doc, d, slot, tipo = key
        by_ddt[(doc, d, tipo)].append(var)
    for (doc, d, tipo), vars_ in by_ddt.items():
        m.Add(sum(vars_) == cnt.get((doc, d, tipo), 0))


def _fix_complement(m, inst: Instance, sol: dict, free_teachers: set, x, y, g) -> None:
    """Fija a su valor actual toda actividad que no involucre a `free_teachers`."""
    from collections import defaultdict
    # placements por unidad desde la solucion (miembros comparten colocacion)
    unit_slots: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for e in sol.get("Eventos", []):
        u_idx = inst.event_to_unit.get(e["Evento Id"])
        if u_idx is None:
            continue
        if e["Evento Id"] != inst.units[u_idx].event_ids[0]:
            continue
        unit_slots[u_idx].append((e["Dia semana"], str(e["Slot"])))

    for u in inst.units:
        if u.teachers & free_teachers:
            continue  # vecindario: libre
        lens = list(u.session_lengths)
        used = [False] * len(lens)
        for (d, slot) in unit_slots.get(u.idx, []):
            if slot not in inst.class_slots:
                continue
            cs = inst.class_slots.index(slot)
            for k, L in enumerate(lens):
                if used[k]:
                    continue
                if cs in set(inst.valid_starts(L)) and (d, cs) in x.get((u.idx, k), {}):
                    m.Add(x[(u.idx, k)][(d, cs)] == 1)
                    used[k] = True
                    break

    # reuniones: libres si algun participante esta en el vecindario
    cur_reu = {(r["Reunion Id"], r["Dia semana"], str(r["Slot"])) for r in sol.get("Reuniones", [])}
    for rid, info in inst.reuniones.items():
        if set(info["participantes"]) & free_teachers:
            continue
        for (d, slot) in [(rd, rs) for (ri, rd, rs) in cur_reu if ri == rid]:
            if slot in inst.class_slots:
                cs = inst.class_slots.index(slot)
                if (d, cs) in y.get(rid, {}):
                    m.Add(y[rid][(d, cs)] == 1)

    # guardias: fijar todas las de docentes fuera del vecindario
    cur_g = {(gd["Profesor Id"], gd["Dia semana"], str(gd["Slot"]), gd["Tipo guardia"])
             for gd in sol.get("Guardias", [])}
    for key, var in g.items():
        doc = key[0]
        if doc in free_teachers:
            continue
        m.Add(var == (1 if key in cur_g else 0))


def _apply_hint(m, inst: Instance, hint: dict, x, y, g) -> None:
    """Aplica una solucion previa como warm-start (best-effort)."""
    from collections import defaultdict
    # eventos: agrupar slots por unidad (los miembros comparten colocacion)
    unit_slots: dict[int, list[tuple[str, str]]] = defaultdict(list)
    seen_unit = set()
    for e in hint.get("Eventos", []):
        eid = e["Evento Id"]
        u_idx = inst.event_to_unit.get(eid)
        if u_idx is None:
            continue
        # usar solo un evento representativo por unidad
        rep = inst.units[u_idx].event_ids[0]
        if eid != rep:
            continue
        unit_slots[u_idx].append((e["Dia semana"], str(e["Slot"])))

    for u_idx, placements in unit_slots.items():
        u = inst.units[u_idx]
        lens = list(u.session_lengths)
        # emparejar cada (dia,slot) con una longitud valida y su variable x
        used = [False] * len(lens)
        for (d, slot) in placements:
            if slot not in inst.class_slots:
                continue
            cs = inst.class_slots.index(slot)
            for k, L in enumerate(lens):
                if used[k]:
                    continue
                if cs in set(inst.valid_starts(L)) and (d, cs) in x.get((u_idx, k), {}):
                    m.AddHint(x[(u_idx, k)][(d, cs)], 1)
                    used[k] = True
                    break

    for r in hint.get("Reuniones", []):
        rid = r["Reunion Id"]
        slot = str(r["Slot"])
        if slot in inst.class_slots and rid in y:
            cs = inst.class_slots.index(slot)
            if (r["Dia semana"], cs) in y[rid]:
                m.AddHint(y[rid][(r["Dia semana"], cs)], 1)

    for gd in hint.get("Guardias", []):
        key = (gd["Profesor Id"], gd["Dia semana"], str(gd["Slot"]), gd["Tipo guardia"])
        if key in g:
            m.AddHint(g[key], 1)
