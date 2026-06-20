"""Ejecuta el planificador sobre una o varias instancias y genera la entrega.

Uso:
  py run.py <ruta.json> [--time 60]          resuelve una instancia
  py run.py --all [--time 120]               resuelve todas y crea soluciones/ + entrega.zip
  py run.py --eval <salida.json> <entrada.json>   solo valida una salida
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from glob import glob

from planner.instance import load_instance
from planner.solver import solve
from planner.evaluator import evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset_smartiming")
OUT_DIR = os.path.join(HERE, "soluciones")


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def run_one(path: str, time_limit: float, workers: int, log: bool = False) -> dict:
    inst = load_instance(path)
    print(f"\n=== {inst.name} ===")
    print(f"  unidades={len(inst.units)} docentes={len(inst.docentes)} "
          f"reuniones={len(inst.reuniones)} guardias_demanda={sum(inst.guard_demand.values())}")

    # Fase 1: factibilidad pura (rapida, sin objetivo de huecos)
    t_feas = min(max(20.0, time_limit * 0.25), 90.0)
    sol_f, info_f = solve(inst, time_limit=t_feas, workers=workers, optimize=False)
    rep_f = evaluate(inst, sol_f)
    print(f"  [factib] status={info_f['status']} t={info_f['wall_time']:.1f}s "
          f"HUECOS={rep_f.huecos} factible={rep_f.feasible} viol={len(rep_f.hard_violations)}")

    # Fase 2: optimizar huecos usando la solucion factible como pista
    sol_o, info_o = solve(inst, time_limit=time_limit, workers=workers, log=log,
                          optimize=True, hint=sol_f if rep_f.feasible else None)
    rep_o = evaluate(inst, sol_o)
    print(f"  [optim ] status={info_o['status']} objetivo={info_o['objective']} "
          f"cota={info_o['best_bound']} t={info_o['wall_time']:.1f}s "
          f"HUECOS={rep_o.huecos} factible={rep_o.feasible} viol={len(rep_o.hard_violations)}")

    # elegir la mejor solucion factible (menos huecos); si ninguna es factible,
    # quedarse con la de optimizacion
    cands = [(rep_o, sol_o, info_o), (rep_f, sol_f, info_f)]
    feas = [(r, s, i) for r, s, i in cands if r.feasible]
    if feas:
        rep, sol, info = min(feas, key=lambda c: c[0].huecos)
    else:
        rep, sol, info = cands[0]

    print(f"  >> ELEGIDA: HUECOS={rep.huecos} factible={rep.feasible} "
          f"violaciones={len(rep.hard_violations)}")
    for v in rep.hard_violations[:15]:
        print("    !", v)
    if len(rep.hard_violations) > 15:
        print(f"    ... (+{len(rep.hard_violations) - 15} mas)")
    return {"name": inst.name, "sol": sol, "info": info, "report": rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="ruta a instancia .json")
    ap.add_argument("--all", action="store_true", help="resolver todas las instancias")
    ap.add_argument("--time", type=float, default=60.0, help="limite de tiempo por instancia (s)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log", action="store_true", help="log del solver")
    ap.add_argument("--eval", nargs=2, metavar=("SALIDA", "ENTRADA"),
                    help="validar una salida contra su entrada")
    args = ap.parse_args()

    if args.eval:
        sal, ent = args.eval
        inst = load_instance(ent)
        with open(sal, encoding="utf-8") as fh:
            sol = json.load(fh)
        rep = evaluate(inst, sol)
        print(f"HUECOS={rep.huecos} factible={rep.feasible} violaciones={len(rep.hard_violations)}")
        for v in rep.hard_violations:
            print(" !", v)
        return

    if args.all:
        os.makedirs(OUT_DIR, exist_ok=True)
        results = []
        for path in sorted(glob(os.path.join(DATASET_DIR, "*.json"))):
            res = run_one(path, args.time, args.workers, args.log)
            out_path = os.path.join(OUT_DIR, res["name"])  # mismo nombre que la entrada
            _write_json(out_path, res["sol"])
            results.append(res)
        # crear zip de entrega
        zip_path = os.path.join(HERE, "entrega.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for res in results:
                zf.write(os.path.join(OUT_DIR, res["name"]), arcname=res["name"])
        print("\n================ RESUMEN ================")
        for res in results:
            r = res["report"]
            print(f"  {res['name']:<45} huecos={r.huecos:<8} "
                  f"factible={r.feasible} viol={len(r.hard_violations)}")
        print(f"\nEntrega: {zip_path}")
        return

    if not args.path:
        ap.print_help()
        sys.exit(1)
    res = run_one(args.path, args.time, args.workers, args.log)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, res["name"])
    _write_json(out_path, res["sol"])
    print(f"  salida -> {out_path}")


if __name__ == "__main__":
    main()
