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
    """
    df = df.sort_values(["cp", "fechaHora"]).copy()

    for lag in DEMAND_LAGS:
        df[f"lag_{lag}h"] = df.groupby("cp")[TARGET].shift(lag)

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
                                 verbose: bool = True) -> dict:
    """
    Ejecuta walk-forward validation: entrena y valida en varios cortes
    temporales sucesivos, y promedia el sMAPE. Esto da mucha más confianza
    de que una mejora de features/hiperparámetros generaliza, en vez de
    sobreajustar a las particularidades de un único mes de hold-out.

    Devuelve dict con sMAPE por fold y el promedio.
    """
    folds = build_walk_forward_folds(df, n_folds, validation_days)
    fold_smapes = []

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

        train_set = lgb.Dataset(train_fold[feature_cols], label=train_fold[TARGET], categorical_feature=cat_features)
        valid_set = lgb.Dataset(valid_fold[feature_cols], label=valid_fold[TARGET], categorical_feature=cat_features, reference=train_set)

        model = lgb.train(
            lgbm_params,
            train_set,
            num_boost_round=3000,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        preds = model.predict(valid_fold[feature_cols], num_iteration=model.best_iteration)
        preds = np.clip(preds, 0, None)
        fold_smape = smape(valid_fold[TARGET].values, preds)
        fold_smapes.append(fold_smape)

        if verbose:
            print(f"  Fold {i}: train hasta {train_end.date()} | "
                  f"valid {valid_start.date()} -> {valid_end.date()} | "
                  f"sMAPE = {fold_smape:.3f}%  (best_iter={model.best_iteration})")

    avg_smape = float(np.mean(fold_smapes)) if fold_smapes else float("nan")
    return {"fold_smapes": fold_smapes, "avg_smape": avg_smape}


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
                       num_boost_round: int = 800):
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

    model = lgb.train(
        lgbm_params,
        lgb.Dataset(full_train[feature_cols], label=full_train[TARGET], categorical_feature=cat_features),
        num_boost_round=num_boost_round,
    )
    return model


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


def recursive_forecast(model, static_forecast_df: pd.DataFrame, history: pd.DataFrame,
                        feature_cols: list[str], cat_features: list[str]) -> pd.DataFrame:
    """
    Predicción RECURSIVA hora a hora para todo el horizonte de marzo.

    En cada paso:
      1. Se toma la siguiente fechaHora a predecir (todas las cp de esa hora).
      2. Se recalculan lag_24h/lag_48h/lag_168h y roll_mean/roll_std a partir
         de la serie histórica + predicciones ya generadas en pasos anteriores.
      3. Se predice con el modelo y se añade el resultado a la serie histórica
         ("demanda" sintética) para que esté disponible como lag en los
         siguientes pasos.

    Esto resuelve el problema señalado en la review: con un enfoque NO
    recursivo, lag_168h solo queda completo la primera semana de marzo (168h
    = exactamente 7 días) y a partir de ahí depende de predicciones que aún
    no existen. Con este bucle, en el momento de predecir la hora t siempre
    disponemos ya de las predicciones de t-1, t-2, ..., así que los lags
    nunca se quedan sin datos.

    Devuelve un DataFrame con una fila por (fechaHora, cp) y la columna
    'prediccion'.
    """
    # Serie histórica mutable: fechaHora, cp, demanda (real + sintética)
    running_history = history[["fechaHora", "cp", TARGET]].copy()
    running_history = running_history.sort_values(["cp", "fechaHora"]).reset_index(drop=True)

    horizon_hours = sorted(static_forecast_df["fechaHora"].unique())
    predictions = []

    for step, ts in enumerate(horizon_hours, 1):
        hour_df = static_forecast_df[static_forecast_df["fechaHora"] == ts].copy()

        # Concatenar histórico (real+sintético) con la hora actual (target NaN)
        # SOLO para calcular lags/rolling de esta hora concreta -- es más
        # barato que recalcular todo el dataframe en cada paso.
        combined = pd.concat(
            [running_history, hour_df[["fechaHora", "cp"]].assign(**{TARGET: np.nan})],
            ignore_index=True,
        ).sort_values(["cp", "fechaHora"])

        combined = add_lag_and_rolling_features(combined)
        lag_roll_cols = [c for c in combined.columns if c.startswith(("lag_", "roll_"))]

        current = combined[combined["fechaHora"] == ts][["cp"] + lag_roll_cols]
        hour_df = hour_df.merge(current, on="cp", how="left")

        for c in cat_features:
            hour_df[c] = hour_df[c].astype("category")

        hour_df["prediccion"] = model.predict(hour_df[feature_cols])
        hour_df["prediccion"] = hour_df["prediccion"].clip(lower=0)

        predictions.append(hour_df[["fechaHora", "cp", "prediccion"]])

        # Realimentar el histórico con la predicción de esta hora para que
        # esté disponible como lag en los próximos pasos
        new_history_rows = hour_df[["fechaHora", "cp", "prediccion"]].rename(columns={"prediccion": TARGET})
        running_history = pd.concat([running_history, new_history_rows], ignore_index=True)

        if step % 100 == 0 or step == len(horizon_hours):
            print(f"  Predicción recursiva: {step}/{len(horizon_hours)} horas procesadas")

    result = pd.concat(predictions, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline de predicción de demanda energética")
    parser.add_argument("--tune", action="store_true", help="Activa búsqueda de hiperparámetros con Optuna")
    parser.add_argument("--trials", type=int, default=30, help="Número de trials de Optuna")
    args = parser.parse_args()

    print("Cargando datos...")
    data = load_raw_data()

    print("Construyendo features...")
    full_df = build_feature_frame(data)

    print(f"Dataset completo: {full_df.shape[0]} filas, {full_df.shape[1]} columnas")
    print(f"Rango de fechas: {full_df['fechaHora'].min()} -> {full_df['fechaHora'].max()}")

    # --- Filtrado de filas sin target (huecos largos en demanda histórica) ---
    # Los micro-huecos (<=4h) ya se imputaron en build_feature_frame(); lo que
    # queda en NaN aquí son huecos largos genuinos que NO se inventan.
    n_before = full_df.shape[0]
    print("\nFilas con target NaN por cp (huecos largos, tras imputar micro-huecos):")
    print(full_df.groupby("cp")[TARGET].apply(lambda s: s.isna().sum()))
    full_df_modelable = full_df.dropna(subset=[TARGET]).copy()
    print(f"Filas totales: {n_before} -> filas con target válido: {full_df_modelable.shape[0]} "
          f"({n_before - full_df_modelable.shape[0]} excluidas)")

    feature_cols = get_feature_columns(full_df_modelable)
    print(f"\nFeatures usadas en LightGBM ({len(feature_cols)}): {feature_cols}")

    if not LGBM_AVAILABLE:
        print("\n[AVISO] lightgbm no instalado. Instálalo con `pip install lightgbm` para continuar.")
        sys.exit(1)

    # --- Baseline (referencia rápida, sobre el último mes) ---
    train_simple = full_df_modelable[full_df_modelable["fechaHora"] <= full_df_modelable["fechaHora"].max() - pd.Timedelta(days=VALIDATION_DAYS)]
    valid_simple = full_df_modelable[full_df_modelable["fechaHora"] > full_df_modelable["fechaHora"].max() - pd.Timedelta(days=VALIDATION_DAYS)]
    print("\n=== Baseline (naive estacional cp/hora/día-semana, último mes) ===")
    baseline_pred = baseline_seasonal_naive(train_simple, valid_simple[["cp", "hour", "dayofweek"]])
    baseline_smape = smape(valid_simple[TARGET].values, baseline_pred.values)
    print(f"sMAPE baseline (último mes): {baseline_smape:.3f}%")

    # --- Walk-forward validation con LightGBM ---
    print(f"\n=== Walk-forward validation ({N_WALKFORWARD_FOLDS} folds x {VALIDATION_DAYS} días) ===")
    wf_result = run_walk_forward_validation(
        full_df_modelable, feature_cols, DEFAULT_LGBM_PARAMS,
        n_folds=N_WALKFORWARD_FOLDS, validation_days=VALIDATION_DAYS,
    )
    print(f"sMAPE walk-forward (parámetros por defecto): "
          f"folds={[f'{s:.3f}%' for s in wf_result['fold_smapes']]} "
          f"| promedio={wf_result['avg_smape']:.3f}%")

    # --- Tuning opcional con Optuna ---
    if args.tune:
        if not OPTUNA_AVAILABLE:
            print("\n[AVISO] optuna no instalado. Instálalo con `pip install optuna` para usar --tune. "
                  "Se continúa con los parámetros por defecto.")
            best_params = DEFAULT_LGBM_PARAMS
        else:
            best_params = tune_hyperparameters(full_df_modelable, feature_cols, n_trials=args.trials)
            wf_result_tuned = run_walk_forward_validation(
                full_df_modelable, feature_cols, best_params,
                n_folds=N_WALKFORWARD_FOLDS, validation_days=VALIDATION_DAYS,
            )
            print(f"sMAPE walk-forward (parámetros tuneados): "
                  f"folds={[f'{s:.3f}%' for s in wf_result_tuned['fold_smapes']]} "
                  f"| promedio={wf_result_tuned['avg_smape']:.3f}%")
    else:
        best_params = DEFAULT_LGBM_PARAMS

    # --- Determinar num_boost_round a partir de un último fold con early stopping ---
    last_train = full_df_modelable[full_df_modelable["fechaHora"] <= full_df_modelable["fechaHora"].max() - pd.Timedelta(days=VALIDATION_DAYS)]
    last_train = last_train.dropna(subset=["lag_168h", "roll_mean_168h"]).copy()
    last_valid = full_df_modelable[full_df_modelable["fechaHora"] > full_df_modelable["fechaHora"].max() - pd.Timedelta(days=VALIDATION_DAYS)]
    last_valid = last_valid.dropna(subset=["lag_168h", "roll_mean_168h"]).copy()

    cat_features = [c for c in ["cp", "Area", "es_festivo_o_domingo"] if c in feature_cols]
    for c in cat_features:
        last_train[c] = last_train[c].astype("category")
        last_valid[c] = last_valid[c].astype("category")

    probe_model = lgb.train(
        best_params,
        lgb.Dataset(last_train[feature_cols], label=last_train[TARGET], categorical_feature=cat_features),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(last_valid[feature_cols], label=last_valid[TARGET], categorical_feature=cat_features)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    best_num_boost_round = probe_model.best_iteration
    print(f"\nnum_boost_round elegido para el modelo final: {best_num_boost_round}")

    # --- Entrenamiento final con TODO el histórico ---
    print("\nReentrenando con todo el histórico para la predicción final...")
    final_model = train_final_model(full_df_modelable, feature_cols, best_params, num_boost_round=best_num_boost_round)

    # --- Construcción del horizonte y predicción RECURSIVA ---
    print("\nConstruyendo horizonte de predicción (marzo 2023)...")
    forecast_index = build_forecast_index()
    static_forecast_df = build_static_forecast_features(forecast_index, data)

    print("Ejecutando predicción recursiva hora a hora...")
    forecast_result = recursive_forecast(
        final_model, static_forecast_df, full_df, feature_cols, cat_features
    )

    # --- Pivotar a formato ancho requerido por el reto ---
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

    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nFichero de entrega guardado en: {out_path}")
    print(submission.head())
    print("\nEstadísticas de la predicción final:")
    print(submission[CP_COLS].describe())


if __name__ == "__main__":
    main()