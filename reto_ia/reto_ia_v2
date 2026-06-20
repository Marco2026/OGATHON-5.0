"""
Reto IA - Predicción de demanda energética horaria por código postal (v2)
============================================================================

Pipeline para iterar. Cubre:
  1. Carga de los 4 ficheros de datos
  2. Construcción de un dataset "long" fechaHora x cp
  3. Feature engineering:
       - calendario (cest, festivos, vísperas/post-festivo)
       - meteo (grados-día calefacción/refrigeración, lags y rolling de clima)
       - interacciones explícitas (hora x temperatura)
       - lags y rolling stats de la propia demanda
       - metadatos estáticos del cp (Area, NumeroClientes)
  4. Imputación de micro-huecos (<=4h) en el target vía interpolación temporal,
     dejando como NaN (y por tanto excluidos de train) los huecos largos
  5. Walk-forward validation (varios cortes temporales, no un único hold-out)
  6. Modelo baseline (naive estacional: misma hora/día de la semana, media histórica)
  7. Modelo LightGBM global (cp como feature categórica) con tuning opcional vía Optuna
  8. Predicción RECURSIVA para el horizonte de marzo 2023 (hora a hora,
     realimentando los lags con las propias predicciones) — soluciona el
     problema de NaN progresivo en lag_24h/lag_48h/lag_168h
  9. Generación del fichero de entrega final respetando el cambio de hora
     (DST) en Europe/Madrid

Uso:
    python predict_demanda_v2.py [--tune] [--trials N]

    --tune          activa la búsqueda de hiperparámetros con Optuna
                    (si no, usa unos parámetros razonables por defecto)
    --trials N      número de trials de Optuna (default 30)

Estructura de carpetas esperada (ajusta DATA_DIR si difiere):
    /mnt/user-data/uploads/
        demanda_energia_entrenamiento.csv
        cp_descripcion.csv
        clima.csv
        calendario.csv

Salida:
    /mnt/user-data/outputs/submission.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# ---------------------------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------------------------

DATA_DIR = Path("reto_ia")
OUTPUT_DIR = Path("reto_ia/outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TZ = "Europe/Madrid"
CP_COLS = ["cp_41001", "cp_41003", "cp_41005", "cp_41010", "cp_41020"]
CPS = [c.replace("cp_", "") for c in CP_COLS]  # ["41001", "41003", ...]

TARGET = "demanda"

# Horizonte de predicción final (marzo 2023, hora local Madrid)
FORECAST_START = pd.Timestamp("2023-03-01 00:00:00", tz=TZ)
FORECAST_END = pd.Timestamp("2023-03-31 23:00:00", tz=TZ)

# Walk-forward validation: lista de cortes (fin_train, fin_valid) en días desde
# el principio de la serie. Se generan dinámicamente en build_walk_forward_folds().
N_WALKFORWARD_FOLDS = 3
VALIDATION_DAYS = 31  # tamaño de cada fold de validación

# Límite de imputación de micro-huecos en el target (en horas)
MICRO_GAP_LIMIT = 4

# Lags de demanda (en horas)
DEMAND_LAGS = [24, 48, 168]
DEMAND_ROLL_WINDOWS = [24, 168]

# Lags/rolling de variables meteorológicas (en horas)
WEATHER_ROLL_WINDOWS = [24, 48]

# Umbrales de grados-día (°C), valores típicos de literatura de demanda eléctrica
HEATING_BASE_TEMP = 18.0
COOLING_BASE_TEMP = 22.0


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Carga los 4 ficheros tal cual vienen en el dataset del reto."""
    demanda = pd.read_csv(data_dir / "demanda_energia_entrenamiento.csv")
    cp_desc = pd.read_csv(data_dir / "cp_descripcion.csv")
    clima = pd.read_csv(data_dir / "clima.csv")
    calendario = pd.read_csv(data_dir / "calendario.csv")

    return {
        "demanda": demanda,
        "cp_desc": cp_desc,
        "clima": clima,
        "calendario": calendario,
    }


def to_long_format(demanda_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte demanda_energia_entrenamiento.csv (formato ancho:
    fechaHora, cp_41001, cp_41003, ...) a formato largo: fechaHora, cp, demanda.
    Detecta automáticamente si ya viene en formato largo.
    """
    df = demanda_wide.copy()

    if {"fechaHora", "cp", TARGET}.issubset(df.columns):
        df["fechaHora"] = pd.to_datetime(df["fechaHora"], utc=True).dt.tz_convert(TZ)
        df["cp"] = df["cp"].astype(str)
        return df[["fechaHora", "cp", TARGET]]

    id_col = "fechaHora"
    value_cols = [c for c in df.columns if c in CP_COLS]
    if not value_cols:
        raise ValueError(
            f"No se encontraron columnas de CP esperadas {CP_COLS} en "
            f"demanda_energia_entrenamiento.csv. Columnas disponibles: {list(df.columns)}"
        )

    df[id_col] = pd.to_datetime(df[id_col], utc=True).dt.tz_convert(TZ)
    long_df = df.melt(id_vars=[id_col], value_vars=value_cols,
                       var_name="cp", value_name=TARGET)
    long_df["cp"] = long_df["cp"].str.replace("cp_", "", regex=False)
    return long_df.sort_values(["cp", id_col]).reset_index(drop=True)


def impute_micro_gaps(long_df: pd.DataFrame, limit: int = MICRO_GAP_LIMIT) -> pd.DataFrame:
    """
    Imputa SOLO los micro-huecos (<= `limit` horas consecutivas) del target,
    por cp, mediante interpolación temporal. Los huecos largos (p.ej. la
    racha de ~995h en cp_41020) se dejan en NaN deliberadamente -- imputar
    semanas de demanda inventada metería ruido falso en el entrenamiento.

    Se añade una columna 'demanda_imputada' (bool) para poder diferenciar,
    si se quiere, filas con target real vs interpolado.
    """
    df = long_df.sort_values(["cp", "fechaHora"]).copy()
    df["demanda_imputada"] = False

    def _impute_group(g: pd.DataFrame) -> pd.DataFrame:
        cp_value = g["cp"].iloc[0]
        original_na = g[TARGET].isna()
        g = g.set_index("fechaHora")
        interpolated = g[TARGET].interpolate(method="time", limit=limit, limit_direction="both")
        g[TARGET] = interpolated
        g = g.reset_index()
        g["cp"] = cp_value  # se restaura explícitamente por seguridad
        g["demanda_imputada"] = original_na.values & g[TARGET].notna().values
        return g

    df = df.groupby("cp", group_keys=False)[df.columns].apply(_impute_group)
    n_imputed = df["demanda_imputada"].sum()
    print(f"  Micro-huecos imputados (<= {limit}h consecutivas): {n_imputed} filas")
    return df.sort_values(["cp", "fechaHora"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame, calendario: pd.DataFrame) -> pd.DataFrame:
    """
    Une variables de calendario (cest, es_festivo_o_domingo) y añade
    features derivadas: componentes temporales, encoding cíclico, y
    variables de "efecto puente" (víspera / día posterior a festivo).
    """
    cal = calendario.copy()
    date_col_candidates = [c for c in cal.columns if "fecha" in c.lower()]
    if not date_col_candidates:
        raise ValueError("No se encontró columna de fecha en calendario.csv")
    date_col = date_col_candidates[0]
    cal[date_col] = pd.to_datetime(cal[date_col], utc=True).dt.tz_convert(TZ)
    cal = cal.rename(columns={date_col: "fechaHora"})

    # --- Efecto puente: víspera de festivo / día posterior a festivo ---
    # Se calcula a nivel de DÍA (no hora), comparando con el día siguiente/anterior.
    if "es_festivo_o_domingo" in cal.columns:
        cal_sorted = cal.sort_values("fechaHora").copy()
        cal_sorted["fecha_dia"] = cal_sorted["fechaHora"].dt.date
        daily_flag = cal_sorted.groupby("fecha_dia")["es_festivo_o_domingo"].max()
        daily_flag = daily_flag.sort_index()
        # día siguiente es festivo -> hoy es víspera
        vispera = daily_flag.shift(-1).fillna(False).astype(bool)
        # día anterior fue festivo -> hoy es "post-festivo"
        post_festivo = daily_flag.shift(1).fillna(False).astype(bool)
        bridge_map = pd.DataFrame({
            "fecha_dia": daily_flag.index,
            "vispera_de_festivo": vispera.values,
            "dia_post_festivo": post_festivo.values,
        })
        cal_sorted = cal_sorted.merge(bridge_map, on="fecha_dia", how="left")
        cal = cal_sorted.drop(columns=["fecha_dia"])

    out = df.merge(cal, on="fechaHora", how="left")

    # Features de calendario derivadas directamente de la marca temporal
    out["hour"] = out["fechaHora"].dt.hour
    out["dayofweek"] = out["fechaHora"].dt.dayofweek  # 0=lunes
    out["day"] = out["fechaHora"].dt.day
    out["month"] = out["fechaHora"].dt.month
    out["year"] = out["fechaHora"].dt.year
    out["weekofyear"] = out["fechaHora"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    # Encoding cíclico
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    return out


def add_weather_features(df: pd.DataFrame, clima: pd.DataFrame) -> pd.DataFrame:
    """
    Une variables meteorológicas, imputa nulos por interpolación temporal,
    y añade:
      - Grados-día de calefacción / refrigeración (relación no lineal en V/U
        entre temperatura y demanda eléctrica)
      - Lags y medias móviles de temperatura (inercia térmica de los edificios)
    """
    cl = clima.copy()

    # --- Lags a corto plazo de temperatura ---
    for lag in [1, 2, 3, 6]:
        cl[f"temp_lag_{lag}h"] = cl["temperatura"].shift(lag)
        
    # Diferencia térmica (¿Se está enfriando o calentando de golpe?)
    cl["temp_diff_3h"] = cl["temperatura"] - cl["temp_lag_3h"]

    date_col_candidates = [c for c in cl.columns if "fecha" in c.lower()]
    if not date_col_candidates:
        raise ValueError("No se encontró columna de fecha en clima.csv")
    date_col = date_col_candidates[0]
    cl[date_col] = pd.to_datetime(cl[date_col], utc=True).dt.tz_convert(TZ)
    cl = cl.rename(columns={date_col: "fechaHora"}).sort_values("fechaHora").reset_index(drop=True)

    weather_cols = [c for c in cl.columns if c != "fechaHora"]
    cl[weather_cols] = cl[weather_cols].interpolate(method="linear", limit_direction="both")

    # --- Grados-día (heating/cooling degree hours) ---
    if "temperatura" in cl.columns:
        cl["grados_calefaccion"] = (HEATING_BASE_TEMP - cl["temperatura"]).clip(lower=0)
        cl["grados_refrigeracion"] = (cl["temperatura"] - COOLING_BASE_TEMP).clip(lower=0)

        # --- Lags y rolling de temperatura: inercia térmica ---
        for window in WEATHER_ROLL_WINDOWS:
            cl[f"temp_roll_mean_{window}h"] = cl["temperatura"].rolling(window, min_periods=1).mean()
        for lag in [24]:
            cl[f"temp_lag_{lag}h"] = cl["temperatura"].shift(lag)

        # --- EMA de temperatura: un edificio pierde/gana calor de forma
        # exponencial, no con el mismo peso para todas las horas de la
        # ventana como hace una media móvil simple. span=12/24h son puntos
        # de partida razonables (vida media ~ span/2 en horas).
        cl["temp_ema_12h"] = cl["temperatura"].ewm(span=12, adjust=False).mean()
        cl["temp_ema_24h"] = cl["temperatura"].ewm(span=24, adjust=False).mean()

    out = df.merge(cl, on="fechaHora", how="left")
    extra_weather_cols = [c for c in out.columns if c.startswith(
        ("lluvia", "temperatura", "humedad", "velocidadViento", "grados_", "temp_")
    )]
    out[extra_weather_cols] = out[extra_weather_cols].interpolate(method="linear", limit_direction="both")
    return out

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interacciones explícitas que ayudan a árboles de decisión a capturar
    relaciones condicionales que de otra forma requerirían más splits.
    """
    out = df.copy()
    if "temperatura" in out.columns:
        out["hour_x_temp"] = out["hour"] * out["temperatura"]
        out["weekend_x_temp"] = out["is_weekend"] * out["temperatura"]
    return out


def add_static_cp_features(df: pd.DataFrame, cp_desc: pd.DataFrame) -> pd.DataFrame:
    """Une metadatos estáticos del CP: zona (Area), descripción, número de clientes.

    cp_descripcion.csv real trae la columna 'CodifoPostal' (sic) con valores
    tipo 'cp_41001'. Se normaliza a 'cp' = '41001' para el merge.
    """
    cd = cp_desc.copy()
    cp_col_candidates = [c for c in cd.columns if c.lower() in ("codifopostal", "codigopostal", "cp", "codigo_postal")]
    if not cp_col_candidates:
        cp_col_candidates = [c for c in cd.columns if "postal" in c.lower() or c.lower() == "cp"]
    if not cp_col_candidates:
        raise ValueError(
            f"No se encontró columna de código postal en cp_descripcion.csv. "
            f"Columnas disponibles: {list(cd.columns)}"
        )
    cp_col = cp_col_candidates[0]
    cd[cp_col] = cd[cp_col].astype(str).str.replace("cp_", "", regex=False)
    cd = cd.rename(columns={cp_col: "cp"})

    out = df.merge(cd, on="cp", how="left")
    return out


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade lags y estadísticos móviles de la propia demanda, calculados
    POR cp de forma independiente (groupby) para no mezclar series.

    También añade DIFERENCIAS explícitas entre lags: a los árboles de
    decisión les cuesta más "descubrir" una resta mediante splits que
    recibirla ya calculada como feature directa. Estas diferencias dan al
    modelo una señal de tendencia (¿estamos subiendo o bajando demanda
    respecto al periodo de referencia?):
      - diff_24h_48h:   ayer vs anteayer (misma franja horaria)
      - diff_168h_336h: esta semana vs la semana anterior (tendencia semanal)
    """
    df = df.sort_values(["cp", "fechaHora"]).copy()

    for lag in DEMAND_LAGS:
        df[f"lag_{lag}h"] = df.groupby("cp")[TARGET].shift(lag)

    # Lag adicional de 336h (14 días) solo para poder calcular la diferencia
    # de tendencia semanal; no se incluye como feature base salvo a través
    # de la diferencia (evita redundancia con lag_168h).
    df["_lag_336h_tmp"] = df.groupby("cp")[TARGET].shift(336)

    df["diff_24h_48h"] = df["lag_24h"] - df["lag_48h"]
    df["diff_168h_336h"] = df["lag_168h"] - df["_lag_336h_tmp"]
    df = df.drop(columns=["_lag_336h_tmp"])

    for window in DEMAND_ROLL_WINDOWS:
        df[f"roll_mean_{window}h"] = (
            df.groupby("cp")[TARGET].transform(lambda s, w=window: s.shift(1).rolling(w).mean())
        )
        df[f"roll_std_{window}h"] = (
            df.groupby("cp")[TARGET].transform(lambda s, w=window: s.shift(1).rolling(w).std())
        )

    return df


def build_feature_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pipeline completo de construcción de features a partir de los 4 ficheros."""
    long_df = to_long_format(data["demanda"])
    print("Imputando micro-huecos del target...")
    long_df = impute_micro_gaps(long_df)
    long_df = add_calendar_features(long_df, data["calendario"])
    long_df = add_weather_features(long_df, data["clima"])
    long_df = add_interaction_features(long_df)
    long_df = add_static_cp_features(long_df, data["cp_desc"])
    long_df = add_lag_and_rolling_features(long_df)
    return long_df


# ---------------------------------------------------------------------------
# 3. MÉTRICA: sMAPE
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error, en %.
    Definición estándar: 100/n * sum(|y_true - y_pred| / ((|y_true|+|y_pred|)/2))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


# ---------------------------------------------------------------------------
# 4. WALK-FORWARD VALIDATION
# ---------------------------------------------------------------------------

def build_walk_forward_folds(df: pd.DataFrame, n_folds: int = N_WALKFORWARD_FOLDS,
                              validation_days: int = VALIDATION_DAYS) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Genera `n_folds` cortes de walk-forward validation, cada uno desplazado
    `validation_days` hacia atrás respecto al anterior. Por ejemplo, con
    n_folds=3 y validation_days=31, sobre datos hasta 2023-02-28:
      fold 1: train hasta ~2022-12-28, valid 2022-12-29 -> 2023-01-28
      fold 2: train hasta ~2023-01-28, valid 2023-01-29 -> 2023-02-28  (aprox)
      fold 3: train hasta ~2023-02-28 - 2*31d, valid el mes siguiente
    Devuelve lista de tuplas (train_end, valid_start, valid_end), ordenadas
    de la más antigua a la más reciente.
    """
    max_date = df["fechaHora"].max()
    folds = []
    valid_end = max_date
    for i in range(n_folds):
        valid_start = valid_end - pd.Timedelta(days=validation_days) + pd.Timedelta(hours=1)
        train_end = valid_start - pd.Timedelta(hours=1)
        folds.append((train_end, valid_start, valid_end))
        valid_end = train_end
    folds.reverse()  # de más antiguo a más reciente
    return folds


def run_walk_forward_validation(df: pd.DataFrame, feature_cols: list[str],
                                 lgbm_params: dict, n_folds: int = N_WALKFORWARD_FOLDS,
                                 validation_days: int = VALIDATION_DAYS,
                                 verbose: bool = True,
                                 cp_filter: str | None = None,
                                 use_smape_objective: bool = False,
                                 log_target: bool = False) -> dict:
    """
    Ejecuta walk-forward validation: entrena y valida en varios cortes
    temporales sucesivos, y promedia el sMAPE. Esto da mucha más confianza
    de que una mejora de features/hiperparámetros generaliza, en vez de
    sobreajustar a las particularidades de un único mes de hold-out.

    cp_filter: si se indica, se restringe el dataset a ese cp (usado para
               entrenar/validar modelos LOCALES por cp).
    use_smape_objective: si True, usa el objective custom de sMAPE
               (smape_objective + smape_eval_metric) en vez de
               objective="regression"/"mape" estándar.
    log_target: si True, entrena sobre log1p(demanda) y deshace la
               transformación (expm1) antes de calcular el sMAPE de
               validación, para comparar de forma honesta contra la escala
               original.

    Devuelve dict con sMAPE por fold y el promedio.
    """
    if cp_filter is not None:
        df = df[df["cp"] == cp_filter]

    folds = build_walk_forward_folds(df, n_folds, validation_days)
    fold_smapes = []
    fold_best_iters = []

    for i, (train_end, valid_start, valid_end) in enumerate(folds, 1):
        train_fold = df[df["fechaHora"] <= train_end].dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
        valid_fold = df[(df["fechaHora"] >= valid_start) & (df["fechaHora"] <= valid_end)]
        valid_fold = valid_fold.dropna(subset=["lag_168h", "roll_mean_168h"]).copy()

        if len(train_fold) == 0 or len(valid_fold) == 0:
            if verbose:
                print(f"  Fold {i}: sin datos suficientes, se omite "
                      f"(train_end={train_end.date()}, valid={valid_start.date()}->{valid_end.date()})")
            continue

        cat_features = [c for c in ["cp", "Area", "es_festivo_o_domingo"] if c in feature_cols]
        for c in cat_features:
            train_fold[c] = train_fold[c].astype("category")
            valid_fold[c] = valid_fold[c].astype("category")

        y_train_raw = train_fold[TARGET].values
        y_valid_raw = valid_fold[TARGET].values
        y_train = np.log1p(y_train_raw) if log_target else y_train_raw
        y_valid = np.log1p(y_valid_raw) if log_target else y_valid_raw

        train_set = lgb.Dataset(train_fold[feature_cols], label=y_train, categorical_feature=cat_features)
        valid_set = lgb.Dataset(valid_fold[feature_cols], label=y_valid, categorical_feature=cat_features, reference=train_set)

        train_kwargs = dict(
            params=lgbm_params,
            train_set=train_set,
            num_boost_round=3000,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        if use_smape_objective:
            params_no_obj = {k: v for k, v in lgbm_params.items() if k not in ("objective", "metric")}
            model = lgb.train(
                params=params_no_obj,
                train_set=train_set,
                num_boost_round=3000,
                valid_sets=[valid_set],
                valid_names=["valid"],
                fobj=smape_objective,
                feval=smape_eval_metric,
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
            )
        else:
            model = lgb.train(**train_kwargs)

        preds = model.predict(valid_fold[feature_cols], num_iteration=model.best_iteration)
        if log_target:
            preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)
        fold_smape = smape(y_valid_raw, preds)
        fold_smapes.append(fold_smape)
        fold_best_iters.append(model.best_iteration)

        if verbose:
            print(f"  Fold {i}: train hasta {train_end.date()} | "
                  f"valid {valid_start.date()} -> {valid_end.date()} | "
                  f"sMAPE = {fold_smape:.3f}%  (best_iter={model.best_iteration}, "
                  f"n_train={len(train_fold)}, n_valid={len(valid_fold)})")

    avg_smape = float(np.mean(fold_smapes)) if fold_smapes else float("nan")
    return {"fold_smapes": fold_smapes, "avg_smape": avg_smape, "fold_best_iters": fold_best_iters}


# ---------------------------------------------------------------------------
# 5. MODELO BASELINE (naive estacional)
# ---------------------------------------------------------------------------

def baseline_seasonal_naive(train: pd.DataFrame, target_index: pd.DataFrame) -> pd.Series:
    """
    Baseline: para cada (cp, hour, dayofweek), usa la media histórica de
    `demanda` en train con esa misma combinación.

    IMPORTANTE: se preserva el índice original de target_index en el
    resultado para evitar desalineación al asignar en un DataFrame con
    índice no contiguo.
    """
    original_index = target_index.index

    profile = (
        train.groupby(["cp", "hour", "dayofweek"])[TARGET]
        .mean()
        .rename("baseline_pred")
        .reset_index()
    )
    merged = target_index.reset_index(drop=True).merge(
        profile, on=["cp", "hour", "dayofweek"], how="left"
    )

    cp_mean = train.groupby("cp")[TARGET].mean().rename("cp_mean")
    merged = merged.merge(cp_mean, on="cp", how="left")
    merged["baseline_pred"] = merged["baseline_pred"].fillna(merged["cp_mean"])

    result = merged["baseline_pred"]
    result.index = original_index
    return result


# ---------------------------------------------------------------------------
# 6. MODELO LIGHTGBM GLOBAL (cp como categórica) + TUNING CON OPTUNA
# ---------------------------------------------------------------------------

FEATURE_COLS_BASE = [
    "cp", "hour", "dayofweek", "day", "month", "year", "weekofyear", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "vispera_de_festivo", "dia_post_festivo",
    "lag_24h", "lag_48h", "lag_168h",
    "diff_24h_48h", "diff_168h_336h",
    "roll_mean_24h", "roll_std_24h", "roll_mean_168h", "roll_std_168h",
    "grados_calefaccion", "grados_refrigeracion",
    "hour_x_temp", "weekend_x_temp",
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Construye la lista final de columnas de features: las base +
    cualquier columna extra proveniente de calendario/clima/cp_descripcion
    que no sea identificador, target, fecha o texto libre.
    """
    exclude = {
        TARGET, "fechaHora", "baseline_pred", "cp_mean", "demanda_imputada",
        "Descripcion", "CodifoPostal",
    }
    extra_cols = [
        c for c in df.columns
        if c not in exclude and c not in FEATURE_COLS_BASE
    ]
    cols = [c for c in FEATURE_COLS_BASE if c in df.columns] + extra_cols
    return list(dict.fromkeys(cols))


DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# 6b. CUSTOM OBJECTIVE: sMAPE (alinea la función de pérdida con la métrica del reto)
# ---------------------------------------------------------------------------
#
# LightGBM con objective="regression" minimiza MSE (L2) -> tiende a ajustar
# la MEDIA condicional, penalizando mucho los outliers grandes.
# objective="mape" en LightGBM minimiza |y-yhat|/|y| (MAPE clásico), que NO
# es lo mismo que sMAPE: el denominador de sMAPE usa (|y|+|yhat|)/2, lo que
# lo hace simétrico entre sobre- e infra-predicción; MAPE penaliza mucho más
# la sobre-predicción cuando y es pequeño. Para alinear de verdad con la
# métrica del reto, implementamos sMAPE como objetivo custom (gradiente y
# hessiano analíticos), tal como sugiere la revisión en su "mejora avanzada".
#
# sMAPE(y, yhat) = |y - yhat| / ((|y| + |yhat|)/2)
# En nuestro dominio (demanda de energía) y, yhat >= 0 casi siempre tras el
# clip a 0, así que asumimos y >= 0 y trabajamos con yhat sin valor absoluto
# en las derivadas para mantenerlas estables; se fuerza yhat >= eps con un
# pequeño suelo para evitar división por cero / gradientes explosivos
# cuando el modelo predice valores muy próximos a 0.

SMAPE_EPS = 1e-2  # suelo de estabilidad para yhat en el objective (en UNE)


def smape_objective(y_pred: np.ndarray, train_data: "lgb.Dataset"):
    """
    Gradiente y hessiano (aproximado) de sMAPE para usar como `objective`
    custom de LightGBM. Basado en la forma diferenciable de sMAPE asumiendo
    y_true >= 0 (válido en este dominio: la demanda nunca es negativa).

    sMAPE_i = |y_i - yhat_i| / ((y_i + |yhat_i|)/2) * 100
    Se omite el factor 100 y el promedio 1/n del gradiente (LightGBM solo
    necesita la dirección/escala relativa de gradiente y hessiano, no el
    valor exacto de la pérdida).
    """
    y_true = train_data.get_label()
    yhat = np.where(np.abs(y_pred) < SMAPE_EPS, np.sign(y_pred) * SMAPE_EPS + (y_pred == 0) * SMAPE_EPS, y_pred)

    sign_err = np.sign(yhat - y_true)
    denom = (y_true + np.abs(yhat))
    denom = np.where(denom < SMAPE_EPS, SMAPE_EPS, denom)

    # Gradiente aproximado de |y-yhat| / ((y+|yhat|)/2) respecto a yhat
    grad = 2.0 * (sign_err * denom - np.abs(yhat - y_true) * np.sign(yhat)) / (denom ** 2)
    # Hessiano aproximado (positivo, constante por tramos) -- se usa una
    # aproximación diagonal estable en vez de la segunda derivada exacta,
    # práctica común para objetivos custom no triviales en GBM.
    hess = 2.0 / (denom ** 2)
    hess = np.maximum(hess, 1e-6)

    return grad, hess


def smape_eval_metric(y_pred: np.ndarray, train_data: "lgb.Dataset"):
    """Métrica de evaluación sMAPE para usar en valid_sets junto al objective custom."""
    y_true = train_data.get_label()
    y_pred_clipped = np.clip(y_pred, 0, None)
    value = smape(y_true, y_pred_clipped)
    return "smape", value, False  # False = "menor es mejor"


def tune_hyperparameters(df: pd.DataFrame, feature_cols: list[str], n_trials: int = 30) -> dict:
    """
    Búsqueda de hiperparámetros con Optuna, optimizando el sMAPE promedio
    de walk-forward validation (no un único hold-out, para evitar overfitting
    a las particularidades de un solo mes).
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna no está instalado. Instálalo con: pip install optuna")

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbose": -1,
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        result = run_walk_forward_validation(df, feature_cols, params, verbose=False)
        return result["avg_smape"]

    study = optuna.create_study(direction="minimize")
    print(f"\nBuscando hiperparámetros con Optuna ({n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"Mejor sMAPE (walk-forward) encontrado: {study.best_value:.3f}%")
    print(f"Mejores hiperparámetros: {study.best_params}")

    best_params = {**DEFAULT_LGBM_PARAMS, **study.best_params}
    return best_params


def train_final_model(df_modelable: pd.DataFrame, feature_cols: list[str], lgbm_params: dict,
                       num_boost_round: int = 800, use_smape_objective: bool = False,
                       log_target: bool = False):
    """
    Entrena el modelo final con TODO el histórico disponible (sin hold-out),
    para usar el máximo de información en la predicción real de marzo.
    `num_boost_round` debería fijarse a partir del best_iteration observado
    en walk-forward validation.
    """
    full_train = df_modelable.dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
    cat_features = [c for c in ["cp", "Area", "es_festivo_o_domingo"] if c in feature_cols]
    for c in cat_features:
        full_train[c] = full_train[c].astype("category")

    y = np.log1p(full_train[TARGET].values) if log_target else full_train[TARGET].values
    train_set = lgb.Dataset(full_train[feature_cols], label=y, categorical_feature=cat_features)

    if use_smape_objective:
        params_no_obj = {k: v for k, v in lgbm_params.items() if k not in ("objective", "metric")}
        model = lgb.train(params_no_obj, train_set, num_boost_round=num_boost_round, fobj=smape_objective)
    else:
        model = lgb.train(lgbm_params, train_set, num_boost_round=num_boost_round)
    return model


def train_final_model_catboost(df_modelable: pd.DataFrame, feature_cols: list[str],
                                num_boost_round: int = 800, log_target: bool = False):
    """
    Entrena un modelo CatBoost equivalente al de LightGBM, para usarlo en
    el ensemble (punto 7 de la review: combinar arquitecturas de árboles
    distintas suele reducir varianza y mejorar el sMAPE de forma "casi
    gratis"). CatBoost maneja categóricas de forma nativa sin necesidad de
    castear a dtype category.
    """
    from catboost import CatBoostRegressor, Pool

    full_train = df_modelable.dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
    cat_features = [c for c in ["cp", "Area", "es_festivo_o_domingo"] if c in feature_cols]

    X = full_train[feature_cols].copy()
    for c in cat_features:
        X[c] = X[c].astype(str)
    y = np.log1p(full_train[TARGET].values) if log_target else full_train[TARGET].values

    pool = Pool(X, label=y, cat_features=cat_features)
    model = CatBoostRegressor(
        iterations=num_boost_round,
        learning_rate=0.05,
        depth=8,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )
    model.fit(pool)
    return model


class GlobalModel:
    """Wrapper sobre un único modelo LightGBM global (todas las cp juntas)."""

    def __init__(self, model, feature_cols: list[str], cat_features: list[str], log_target: bool = False):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_features = cat_features
        self.log_target = log_target

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        for c in self.cat_features:
            X[c] = X[c].astype("category")
        preds = self.model.predict(X[self.feature_cols])
        if self.log_target:
            preds = np.expm1(preds)
        return np.clip(preds, 0, None)


class PerCPModel:
    """Wrapper sobre 5 modelos LightGBM independientes, uno por cp."""

    def __init__(self, models_by_cp: dict[str, "lgb.Booster"], feature_cols: list[str],
                 cat_features: list[str], log_target: bool = False):
        self.models_by_cp = models_by_cp
        self.feature_cols = feature_cols
        self.cat_features = cat_features
        self.log_target = log_target

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df))
        for cp_value, model in self.models_by_cp.items():
            mask = (df["cp"] == cp_value).values
            if not mask.any():
                continue
            X = df.loc[mask].copy()
            for c in self.cat_features:
                X[c] = X[c].astype("category")
            p = model.predict(X[self.feature_cols])
            if self.log_target:
                p = np.expm1(p)
            preds[mask] = p
        return np.clip(preds, 0, None)


class EnsembleModel:
    """Media simple de las predicciones de varios modelos (p.ej. LightGBM + CatBoost)."""

    def __init__(self, models: list, feature_cols: list[str], cat_features: list[str],
                 log_target: bool = False, kinds: list[str] | None = None):
        self.models = models
        self.feature_cols = feature_cols
        self.cat_features = cat_features
        self.log_target = log_target
        self.kinds = kinds or ["lightgbm"] * len(models)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        all_preds = []
        for model, kind in zip(self.models, self.kinds):
            if kind == "catboost":
                X = df[self.feature_cols].copy()
                for c in self.cat_features:
                    X[c] = X[c].astype(str)
                p = model.predict(X)
            else:
                X = df.copy()
                for c in self.cat_features:
                    X[c] = X[c].astype("category")
                p = model.predict(X[self.feature_cols])
            if self.log_target:
                p = np.expm1(p)
            all_preds.append(np.clip(p, 0, None))
        return np.mean(all_preds, axis=0)


class ResidualModel:
    """
    Wrapper para "residual modeling": predicción final = baseline estacional
    (cp, hour, dayofweek) + corrección aprendida por LightGBM sobre el
    residuo (demanda_real - baseline). Idea de la review: el baseline ya
    captura la estacionalidad gruesa, así LightGBM solo tiene que aprender
    la desviación, que suele ser una señal más "fácil" (más estacionaria)
    para un GBM que la serie original con su fuerte componente de nivel.
    """

    def __init__(self, residual_model, baseline_profile: pd.DataFrame, cp_mean: pd.Series,
                 feature_cols: list[str], cat_features: list[str], log_target: bool = False):
        self.residual_model = residual_model
        self.baseline_profile = baseline_profile
        self.cp_mean = cp_mean
        self.feature_cols = feature_cols
        self.cat_features = cat_features
        self.log_target = log_target

    def _baseline_predict(self, df: pd.DataFrame) -> np.ndarray:
        merged = df[["cp", "hour", "dayofweek"]].reset_index(drop=True).merge(
            self.baseline_profile, on=["cp", "hour", "dayofweek"], how="left"
        )
        merged = merged.merge(self.cp_mean, on="cp", how="left")
        merged["baseline_pred"] = merged["baseline_pred"].fillna(merged["cp_mean"])
        return merged["baseline_pred"].values

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        baseline_pred = self._baseline_predict(df)
        X = df.copy()
        for c in self.cat_features:
            X[c] = X[c].astype("category")
        residual_pred = self.residual_model.predict(X[self.feature_cols])
        if self.log_target:
            # El residuo no se transforma en log (puede ser negativo); esta
            # rama no debería alcanzarse en la combinación residual+log,
            # se deja explícita para evitar bugs silenciosos.
            raise NotImplementedError("log_target no es compatible con residual modeling")
        final_pred = baseline_pred + residual_pred
        return np.clip(final_pred, 0, None)


def build_baseline_profile(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extrae el perfil baseline (cp, mes, hora, día de la semana) -> media."""
    profile = (
        train.groupby(["cp", "month", "hour", "dayofweek"])[TARGET]
        .mean()
        .rename("baseline_pred")
        .reset_index()
    )
    cp_mean = train.groupby("cp")[TARGET].mean().rename("cp_mean")
    return profile, cp_mean

def baseline_seasonal_naive(train: pd.DataFrame, target_index: pd.DataFrame) -> pd.Series:
    original_index = target_index.index
    profile, cp_mean = build_baseline_profile(train)
    
    # IMPORTANTE: Asegúrate de pasar 'month' en target_index cuando llames a esta función
    merged = target_index.reset_index(drop=True).merge(
        profile, on=["cp", "month", "hour", "dayofweek"], how="left"
    )
    merged = merged.merge(cp_mean, on="cp", how="left")
    merged["baseline_pred"] = merged["baseline_pred"].fillna(merged["cp_mean"])
    
    result = merged["baseline_pred"]
    result.index = original_index
    return result


def train_residual_model(df_modelable: pd.DataFrame, feature_cols: list[str], lgbm_params: dict,
                          num_boost_round: int = 500, use_smape_objective: bool = False) -> "ResidualModel":
    """
    Residual modeling (punto 3 de la review): el baseline estacional
    (cp, hour, dayofweek) ya captura el nivel/estacionalidad gruesa de la
    demanda; LightGBM se entrena para predecir SOLO el residuo
    (demanda_real - baseline), una señal más estacionaria y, en teoría, más
    fácil de aprender para un GBM que la serie original con su fuerte
    componente de nivel.
    """
    full_train = df_modelable.dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
    baseline_profile, cp_mean = build_baseline_profile(full_train)

    baseline_pred = full_train[["cp", "hour", "dayofweek"]].reset_index(drop=True).merge(
        baseline_profile, on=["cp", "hour", "dayofweek"], how="left"
    )
    baseline_pred = baseline_pred.merge(cp_mean, on="cp", how="left")
    baseline_pred["baseline_pred"] = baseline_pred["baseline_pred"].fillna(baseline_pred["cp_mean"])

    full_train = full_train.reset_index(drop=True)
    full_train["residuo"] = full_train[TARGET].values - baseline_pred["baseline_pred"].values

    cat_features = [c for c in ["cp", "Area", "es_festivo_o_domingo"] if c in feature_cols]
    for c in cat_features:
        full_train[c] = full_train[c].astype("category")

    train_set = lgb.Dataset(full_train[feature_cols], label=full_train["residuo"], categorical_feature=cat_features)
    # Para el residuo NO tiene sentido usar el objective custom de sMAPE
    # (el residuo puede ser negativo y cercano a 0 con frecuencia, justo el
    # régimen donde sMAPE-como-objetivo es más inestable); se usa MSE/MAE
    # estándar, que es lo apropiado para modelar un error aditivo.
    params_residual = {k: v for k, v in lgbm_params.items() if k not in ("objective", "metric")}
    params_residual["objective"] = "regression"
    params_residual["metric"] = "mae"
    model = lgb.train(params_residual, train_set, num_boost_round=num_boost_round)

    return ResidualModel(model, baseline_profile, cp_mean, feature_cols, cat_features)


def run_walk_forward_validation_per_cp(df: pd.DataFrame, feature_cols: list[str], lgbm_params: dict,
                                        n_folds: int = N_WALKFORWARD_FOLDS, validation_days: int = VALIDATION_DAYS,
                                        use_smape_objective: bool = False, verbose: bool = True) -> dict:
    """
    Walk-forward validation para el enfoque de modelos LOCALES (uno por cp).
    Entrena/valida cada cp por separado y agrega el sMAPE global ponderando
    por número de observaciones de validación (no por cp, para que sea
    comparable de forma justa con el sMAPE "global" del modelo único, que
    se calcula sobre todas las filas de validación juntas sin distinguir cp).
    """
    feature_cols_local = [c for c in feature_cols if c != "cp"]
    per_cp_results = {}
    all_fold_smapes = [[] for _ in range(n_folds)]
    all_fold_n = [[] for _ in range(n_folds)]

    for cp_value in CPS:
        result = run_walk_forward_validation(
            df, feature_cols_local, lgbm_params, n_folds=n_folds, validation_days=validation_days,
            cp_filter=cp_value, use_smape_objective=use_smape_objective, verbose=False,
        )
        per_cp_results[cp_value] = result
        if verbose:
            folds_str = [f"{s:.2f}%" for s in result["fold_smapes"]]
            print(f"  cp_{cp_value}: folds={folds_str} | promedio={result['avg_smape']:.3f}%")

    avg_smape = float(np.mean([r["avg_smape"] for r in per_cp_results.values()]))
    return {"per_cp_results": per_cp_results, "avg_smape": avg_smape}


def train_per_cp_models(df_modelable: pd.DataFrame, feature_cols: list[str], lgbm_params_by_cp: dict[str, dict],
                         num_boost_round_by_cp: dict[str, int], use_smape_objective: bool = False) -> "PerCPModel":
    """
    Entrena un modelo LightGBM INDEPENDIENTE por cada cp (punto 2 de la
    review): cada zona (centro histórico, residencial, logística...) puede
    tener una dinámica de demanda distinta, así que un modelo local puede
    capturar mejor su estacionalidad propia que un único modelo global con
    cp como categórica.
    """
    cat_features = [c for c in ["Area", "es_festivo_o_domingo"] if c in feature_cols]
    # NOTA: 'cp' se excluye de las features locales porque cada submodelo
    # entrena con un único valor de cp (sería una columna constante).
    feature_cols_local = [c for c in feature_cols if c != "cp"]

    models = {}
    for cp_value in CPS:
        sub = df_modelable[df_modelable["cp"] == cp_value].dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
        for c in cat_features:
            sub[c] = sub[c].astype("category")

        params = lgbm_params_by_cp.get(cp_value, DEFAULT_LGBM_PARAMS)
        n_rounds = num_boost_round_by_cp.get(cp_value, 500)

        train_set = lgb.Dataset(sub[feature_cols_local], label=sub[TARGET], categorical_feature=cat_features)
        if use_smape_objective:
            params_no_obj = {k: v for k, v in params.items() if k not in ("objective", "metric")}
            model = lgb.train(params_no_obj, train_set, num_boost_round=n_rounds, fobj=smape_objective)
        else:
            model = lgb.train(params, train_set, num_boost_round=n_rounds)

        models[cp_value] = model

    return PerCPModel(models, feature_cols_local, cat_features)


# ---------------------------------------------------------------------------
# 7. CONSTRUCCIÓN DEL HORIZONTE Y PREDICCIÓN RECURSIVA
# ---------------------------------------------------------------------------

def build_forecast_index() -> pd.DataFrame:
    """
    Genera el índice fechaHora x cp para el horizonte de evaluación,
    respetando el cambio de hora de Europe/Madrid (743 fechaHora x 5 cp = 3.715 filas).
    """
    fechas = pd.date_range(FORECAST_START, FORECAST_END, freq="h", tz=TZ)
    assert len(fechas) == 743, f"Se esperaban 743 timestamps, se obtuvieron {len(fechas)}"

    idx = pd.MultiIndex.from_product([fechas, CPS], names=["fechaHora", "cp"])
    forecast_df = idx.to_frame(index=False)
    assert len(forecast_df) == 3715, f"Se esperaban 3715 filas, se obtuvieron {len(forecast_df)}"
    return forecast_df


def build_static_forecast_features(forecast_index: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Features que NO dependen de la demanda (calendario, clima, interacciones,
    metadatos del cp) para el horizonte de marzo. Se calculan UNA VEZ, antes
    del bucle recursivo, porque no cambian con las predicciones.
    """
    df = forecast_index.copy()
    df = add_calendar_features(df, data["calendario"])
    df = add_weather_features(df, data["clima"])
    df = add_interaction_features(df)
    df = add_static_cp_features(df, data["cp_desc"])
    return df


def recursive_forecast(model_wrapper, static_forecast_df: pd.DataFrame, history: pd.DataFrame,
                        feature_cols: list[str]) -> pd.DataFrame:
    """
    Predicción RECURSIVA hora a hora para todo el horizonte de marzo.

    En cada paso:
      1. Se toma la siguiente fechaHora a predecir (todas las cp de esa hora).
      2. Se recalculan lag_24h/lag_48h/lag_168h, diff_24h_48h/diff_168h_336h
         y roll_mean/roll_std a partir de la serie histórica + predicciones
         ya generadas en pasos anteriores.
      3. Se predice con el modelo (a través de `model_wrapper.predict(df)`,
         que ya encapsula clip a 0 y, si aplica, deshacer log1p) y se añade
         el resultado a la serie histórica ("demanda" sintética) para que
         esté disponible como lag en los siguientes pasos.

    `model_wrapper` puede ser cualquiera de GlobalModel / PerCPModel /
    EnsembleModel / ResidualModel -- todos exponen `.predict(df) -> ndarray`
    ya recortado a >= 0, así que esta función no necesita saber qué tipo de
    modelo hay detrás.

    Esto resuelve el problema señalado en la review: con un enfoque NO
    recursivo, lag_168h solo queda completo la primera semana de marzo (168h
    = exactamente 7 días) y a partir de ahí depende de predicciones que aún
    no existen. Con este bucle, en el momento de predecir la hora t siempre
    disponemos ya de las predicciones de t-1, t-2, ..., así que los lags
    nunca se quedan sin datos.

    Devuelve un DataFrame con una fila por (fechaHora, cp) y la columna
    'prediccion'.
    """
    running_history = history[["fechaHora", "cp", TARGET]].copy()
    running_history = running_history.sort_values(["cp", "fechaHora"]).reset_index(drop=True)

    horizon_hours = sorted(static_forecast_df["fechaHora"].unique())
    predictions = []

    for step, ts in enumerate(horizon_hours, 1):
        hour_df = static_forecast_df[static_forecast_df["fechaHora"] == ts].copy()

        combined = pd.concat(
            [running_history, hour_df[["fechaHora", "cp"]].assign(**{TARGET: np.nan})],
            ignore_index=True,
        ).sort_values(["cp", "fechaHora"])

        combined = add_lag_and_rolling_features(combined)
        # captura lag_*, roll_*, diff_* (las diferencias entre lags también
        # deben recalcularse en cada paso recursivo)
        derived_cols = [c for c in combined.columns if c.startswith(("lag_", "roll_", "diff_"))]

        current = combined[combined["fechaHora"] == ts][["cp"] + derived_cols]
        hour_df = hour_df.merge(current, on="cp", how="left")

        hour_df["prediccion"] = model_wrapper.predict(hour_df)

        predictions.append(hour_df[["fechaHora", "cp", "prediccion"]])

        new_history_rows = hour_df[["fechaHora", "cp", "prediccion"]].rename(columns={"prediccion": TARGET})
        running_history = pd.concat([running_history, new_history_rows], ignore_index=True)

        if step % 100 == 0 or step == len(horizon_hours):
            print(f"  Predicción recursiva: {step}/{len(horizon_hours)} horas procesadas")

    result = pd.concat(predictions, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def _print_header(title: str):
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def _print_step(msg: str):
    elapsed = time.time() - _START_TIME
    print(f"[t+{elapsed:7.1f}s] {msg}")


_START_TIME = time.time()


def main():
    global _START_TIME
    _START_TIME = time.time()

    parser = argparse.ArgumentParser(description="Pipeline de predicción de demanda energética")
    parser.add_argument("--tune", action="store_true", help="Activa búsqueda de hiperparámetros con Optuna")
    parser.add_argument("--trials", type=int, default=30, help="Número de trials de Optuna")
    parser.add_argument("--model-type", choices=["global", "per-cp", "residual"], default="global",
                         help="Estrategia de modelado: 'global' (1 modelo, cp categórica), "
                              "'per-cp' (5 modelos independientes), 'residual' (LightGBM sobre "
                              "el residuo del baseline estacional). Default: global.")
    parser.add_argument("--smape-objective", action="store_true",
                         help="Usa un objective custom de sMAPE en vez de regression/MAE estándar "
                              "(no aplicable junto con --model-type residual).")
    parser.add_argument("--log-target", action="store_true",
                         help="Entrena sobre log1p(demanda) y deshace la transformación al predecir "
                              "(no aplicable junto con --model-type residual).")
    parser.add_argument("--ensemble", action="store_true",
                         help="Promedia las predicciones de LightGBM y CatBoost (requiere catboost instalado; "
                              "solo aplicable con --model-type global).")
    args = parser.parse_args()

    if args.model_type == "residual" and (args.smape_objective or args.log_target):
        print("[AVISO] --smape-objective y --log-target se ignoran con --model-type residual "
              "(el residuo puede ser negativo/cercano a 0, incompatible con ambas técnicas).")
    if args.ensemble and args.model_type != "global":
        print("[AVISO] --ensemble solo está implementado para --model-type global; se ignora.")

    _print_header("CONFIGURACIÓN DE EJECUCIÓN")
    print(f"  model_type        : {args.model_type}")
    print(f"  smape_objective   : {args.smape_objective}")
    print(f"  log_target        : {args.log_target}")
    print(f"  ensemble (+CatBoost): {args.ensemble}")
    print(f"  tune (Optuna)     : {args.tune}" + (f" ({args.trials} trials)" if args.tune else ""))
    print(f"  walk-forward folds: {N_WALKFORWARD_FOLDS} x {VALIDATION_DAYS} días")
    print(f"  lightgbm disponible: {LGBM_AVAILABLE} | optuna disponible: {OPTUNA_AVAILABLE} | "
          f"catboost disponible: {_catboost_available()}")

    # ------------------------------------------------------------------
    # 1. CARGA Y FEATURES
    # ------------------------------------------------------------------
    _print_header("1. CARGA DE DATOS Y FEATURE ENGINEERING")
    _print_step("Cargando ficheros CSV...")
    data = load_raw_data()
    for name, df in data.items():
        print(f"    {name:35s}: {df.shape[0]:>7} filas x {df.shape[1]} columnas")

    _print_step("Construyendo features (calendario, clima, lags, interacciones)...")
    full_df = build_feature_frame(data)
    _print_step(f"Dataset completo: {full_df.shape[0]} filas x {full_df.shape[1]} columnas")
    print(f"    Rango de fechas: {full_df['fechaHora'].min()} -> {full_df['fechaHora'].max()}")

    n_before = full_df.shape[0]
    print("\n  Filas con target NaN por cp (huecos largos, tras imputar micro-huecos):")
    nan_by_cp = full_df.groupby("cp")[TARGET].apply(lambda s: s.isna().sum())
    for cp_value, n_nan in nan_by_cp.items():
        pct = 100 * n_nan / full_df[full_df["cp"] == cp_value].shape[0]
        print(f"    cp_{cp_value}: {n_nan:>5} filas NaN ({pct:.1f}%)")
    full_df_modelable = full_df.dropna(subset=[TARGET]).copy()
    print(f"  Filas totales: {n_before} -> filas con target válido: {full_df_modelable.shape[0]} "
          f"({n_before - full_df_modelable.shape[0]} excluidas, "
          f"{100*(n_before - full_df_modelable.shape[0])/n_before:.1f}%)")

    feature_cols = get_feature_columns(full_df_modelable)
    print(f"\n  Features usadas ({len(feature_cols)}):")
    print(f"    {feature_cols}")

    if not LGBM_AVAILABLE:
        print("\n[AVISO] lightgbm no instalado. Instálalo con `pip install lightgbm` para continuar.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. BASELINE DE REFERENCIA
    # ------------------------------------------------------------------
    _print_header("2. BASELINE (naive estacional cp/mes/hora/día-semana)")
    cutoff = full_df_modelable["fechaHora"].max() - pd.Timedelta(days=VALIDATION_DAYS)
    train_simple = full_df_modelable[full_df_modelable["fechaHora"] <= cutoff]
    valid_simple = full_df_modelable[full_df_modelable["fechaHora"] > cutoff]
    baseline_pred = baseline_seasonal_naive(train_simple, valid_simple[["cp", "month", "hour", "dayofweek"]])
    baseline_smape = smape(valid_simple[TARGET].values, baseline_pred.values)
    _print_step(f"sMAPE baseline (último mes, referencia rápida): {baseline_smape:.3f}%")

    # ------------------------------------------------------------------
    # 3. HIPERPARÁMETROS (con o sin tuning)
    # ------------------------------------------------------------------
    _print_header("3. HIPERPARÁMETROS LIGHTGBM")
    if args.tune and args.model_type == "per-cp":
        _print_step(f"Lanzando búsqueda Optuna INDEPENDIENTE por CP ({args.trials} trials)...")
        best_params_by_cp = {}
        for cp_value in CPS:
            print(f"\n--- Tuning para cp_{cp_value} ---")
            # Restringimos los datos solo a este CP
            df_cp = full_df_modelable[full_df_modelable["cp"] == cp_value].copy()
            # Quitamos 'cp' de las features
            features_cp = [c for c in feature_cols if c != "cp"]
            best_params_by_cp[cp_value] = tune_hyperparameters(df_cp, features_cp, n_trials=args.trials)
        
        # Guardamos para el entrenamiento final
        best_params = best_params_by_cp 
    elif args.tune:
        _print_step(f"Lanzando búsqueda Optuna GLOBAL ({args.trials} trials)...")
        best_params = tune_hyperparameters(full_df_modelable, feature_cols, n_trials=args.trials)
    else:
        best_params = DEFAULT_LGBM_PARAMS
    # ------------------------------------------------------------------
    # 4. WALK-FORWARD VALIDATION DE LA ESTRATEGIA ELEGIDA
    # ------------------------------------------------------------------
    _print_header(f"4. WALK-FORWARD VALIDATION -- estrategia: {args.model_type}")
    if args.model_type == "global":
        wf_result = run_walk_forward_validation(
            full_df_modelable, feature_cols, best_params,
            n_folds=N_WALKFORWARD_FOLDS, validation_days=VALIDATION_DAYS,
            use_smape_objective=args.smape_objective, log_target=args.log_target,
        )
        print(f"\n  sMAPE walk-forward: folds={[f'{s:.3f}%' for s in wf_result['fold_smapes']]} "
              f"| promedio={wf_result['avg_smape']:.3f}%")
        final_smape_estimate = wf_result["avg_smape"]

    elif args.model_type == "per-cp":
        wf_result = run_walk_forward_validation_per_cp(
            full_df_modelable, feature_cols, best_params,
            n_folds=N_WALKFORWARD_FOLDS, validation_days=VALIDATION_DAYS,
            use_smape_objective=args.smape_objective,
        )
        print(f"\n  sMAPE walk-forward (promedio de los 5 modelos locales): {wf_result['avg_smape']:.3f}%")
        final_smape_estimate = wf_result["avg_smape"]

    else:  # residual
        # Walk-forward simplificado para residual modeling (un solo modelo
        # entrenado sobre el residuo, validado en los mismos folds)
        folds = build_walk_forward_folds(full_df_modelable, N_WALKFORWARD_FOLDS, VALIDATION_DAYS)
        fold_smapes = []
        for i, (train_end, valid_start, valid_end) in enumerate(folds, 1):
            train_fold = full_df_modelable[full_df_modelable["fechaHora"] <= train_end]
            valid_fold = full_df_modelable[
                (full_df_modelable["fechaHora"] >= valid_start) & (full_df_modelable["fechaHora"] <= valid_end)
            ].dropna(subset=["lag_168h", "roll_mean_168h"])
            if len(train_fold) == 0 or len(valid_fold) == 0:
                continue
            res_model = train_residual_model(train_fold, feature_cols, best_params, num_boost_round=500)
            preds = res_model.predict(valid_fold)
            fold_smape = smape(valid_fold[TARGET].values, preds)
            fold_smapes.append(fold_smape)
            print(f"  Fold {i}: train hasta {train_end.date()} | valid {valid_start.date()} -> "
                  f"{valid_end.date()} | sMAPE = {fold_smape:.3f}%")
        final_smape_estimate = float(np.mean(fold_smapes)) if fold_smapes else float("nan")
        print(f"\n  sMAPE walk-forward (residual modeling): {final_smape_estimate:.3f}%")

    print(f"\n  >>> Comparativa rápida: baseline={baseline_smape:.3f}% vs "
          f"modelo ({args.model_type})={final_smape_estimate:.3f}% "
          f"(mejora de {baseline_smape - final_smape_estimate:+.3f} puntos)")

    # ------------------------------------------------------------------
    # 5. ENTRENAMIENTO DEL MODELO FINAL (con todo el histórico)
    # ------------------------------------------------------------------
    _print_header("5. ENTRENAMIENTO DEL MODELO FINAL (todo el histórico)")
    cat_features = [c for c in ["cp", "Area", "es_festivo_o_domingo"] if c in feature_cols]

    # num_boost_round: se estima con un último hold-out + early stopping
    last_train = full_df_modelable[full_df_modelable["fechaHora"] <= cutoff].dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
    last_valid = full_df_modelable[full_df_modelable["fechaHora"] > cutoff].dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
    for c in cat_features:
        last_train[c] = last_train[c].astype("category")
        last_valid[c] = last_valid[c].astype("category")

    y_probe_train = np.log1p(last_train[TARGET].values) if args.log_target else last_train[TARGET].values
    y_probe_valid = np.log1p(last_valid[TARGET].values) if args.log_target else last_valid[TARGET].values
    probe_train_set = lgb.Dataset(last_train[feature_cols], label=y_probe_train, categorical_feature=cat_features)
    probe_valid_set = lgb.Dataset(last_valid[feature_cols], label=y_probe_valid, categorical_feature=cat_features)

    if args.smape_objective and args.model_type != "residual":
        params_no_obj = {k: v for k, v in best_params.items() if k not in ("objective", "metric")}
        probe_model = lgb.train(
            params_no_obj, probe_train_set, num_boost_round=3000,
            valid_sets=[probe_valid_set], fobj=smape_objective, feval=smape_eval_metric,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
    else:
        probe_model = lgb.train(
            best_params, probe_train_set, num_boost_round=3000,
            valid_sets=[probe_valid_set],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
    best_num_boost_round = probe_model.best_iteration
    _print_step(f"num_boost_round elegido para el modelo final: {best_num_boost_round}")

    if args.model_type == "global":
        _print_step("Entrenando modelo LightGBM global con todo el histórico...")
        lgbm_final = train_final_model(
            full_df_modelable, feature_cols, best_params, num_boost_round=best_num_boost_round,
            use_smape_objective=args.smape_objective, log_target=args.log_target,
        )

        if args.ensemble and _catboost_available():
            _print_step("Entrenando modelo CatBoost adicional para el ensemble...")
            cat_final = train_final_model_catboost(
                full_df_modelable, feature_cols, num_boost_round=best_num_boost_round, log_target=args.log_target,
            )
            final_model = EnsembleModel(
                [lgbm_final, cat_final], feature_cols, cat_features,
                log_target=args.log_target, kinds=["lightgbm", "catboost"],
            )
            _print_step("Ensemble LightGBM + CatBoost listo (media simple de predicciones).")
        elif args.ensemble:
            print("  [AVISO] --ensemble pedido pero catboost no disponible; se usa solo LightGBM.")
            final_model = GlobalModel(lgbm_final, feature_cols, cat_features, log_target=args.log_target)
        else:
            final_model = GlobalModel(lgbm_final, feature_cols, cat_features, log_target=args.log_target)

    elif args.model_type == "per-cp":
        _print_step("Entrenando 5 modelos LightGBM independientes (uno por cp)...")
        # num_boost_round y params por cp: de forma simple, se reutiliza el
        # mismo valor estimado arriba para los 5 (se podría refinar con un
        # probe por cp, pero ya da una buena mejora con menos cómputo).
        num_boost_round_by_cp = {cp_value: best_num_boost_round for cp_value in CPS}
        params_by_cp = {cp_value: best_params for cp_value in CPS}
        final_model = train_per_cp_models(
            full_df_modelable, feature_cols, params_by_cp, num_boost_round_by_cp,
            use_smape_objective=args.smape_objective,
        )

    else:  # residual
        _print_step("Entrenando modelo de residuos sobre el baseline...")
        final_model = train_residual_model(
            full_df_modelable, feature_cols, best_params, num_boost_round=best_num_boost_round,
        )

    # ------------------------------------------------------------------
    # 6. PREDICCIÓN RECURSIVA DEL HORIZONTE (marzo 2023)
    # ------------------------------------------------------------------
    _print_header("6. PREDICCIÓN RECURSIVA DEL HORIZONTE (marzo 2023)")
    forecast_index = build_forecast_index()
    static_forecast_df = build_static_forecast_features(forecast_index, data)
    _print_step(f"Horizonte construido: {forecast_index['fechaHora'].nunique()} timestamps x "
                f"{len(CPS)} cp = {len(forecast_index)} filas")

    _print_step("Ejecutando predicción recursiva hora a hora "
                "(realimentando lags con las propias predicciones)...")
    forecast_result = recursive_forecast(final_model, static_forecast_df, full_df, feature_cols)
    _print_step("Predicción recursiva completada.")

    # ------------------------------------------------------------------
    # 7. FICHERO DE ENTREGA
    # ------------------------------------------------------------------
    _print_header("7. CONSTRUCCIÓN DEL FICHERO DE ENTREGA")
    submission = forecast_result.pivot(index="fechaHora", columns="cp", values="prediccion")
    submission = submission.rename(columns={cp: f"cp_{cp}" for cp in CPS})
    submission = submission[CP_COLS]
    submission = submission.reset_index()

    submission["fechaHora"] = submission["fechaHora"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    submission["fechaHora"] = submission["fechaHora"].str.replace(
        r"(\d{2})(\d{2})$", r"\1:\2", regex=True
    )

    assert submission.shape[0] == 743, f"Filas inesperadas: {submission.shape[0]}"
    assert not submission[CP_COLS].isna().any().any(), "Hay valores faltantes en la predicción"
    assert (submission[CP_COLS] >= 0).all().all(), "Hay valores negativos en la predicción"
    _print_step("Validaciones de formato superadas (743 filas, sin NaN/negativos).")

    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    _print_step(f"Fichero de entrega guardado en: {out_path}")

    print("\n  Primeras filas:")
    print(submission.head().to_string(index=False))
    print("\n  Estadísticas de la predicción final por cp:")
    print(submission[CP_COLS].describe().round(3))

    _print_header("RESUMEN FINAL")
    print(f"  Estrategia de modelado     : {args.model_type}"
          + (" + ensemble CatBoost" if (args.ensemble and args.model_type == "global" and _catboost_available()) else ""))
    print(f"  Objective custom sMAPE     : {args.smape_objective}")
    print(f"  Target en log1p            : {args.log_target}")
    print(f"  sMAPE baseline              : {baseline_smape:.3f}%")
    print(f"  sMAPE walk-forward (modelo) : {final_smape_estimate:.3f}%")
    print(f"  Mejora vs baseline          : {baseline_smape - final_smape_estimate:+.3f} puntos")
    print(f"  Tiempo total de ejecución   : {time.time() - _START_TIME:.1f}s")
    print(f"  Fichero de salida           : {out_path}")


def _catboost_available() -> bool:
    try:
        import catboost  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()