#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mega E-Agent — Полная, готовая версия (рефакторинг с сохранением API)

ВНИМАНИЕ:
- Все публичные функции и имена сохранены.
- Исправлены критичные проблемы: безопасная распаковка архивов, корректный
  парсинг команд (shlex), потокобезопасный RNG, синхронизация Matplotlib,
  опциональные тяжёлые импорты, оценка моделей на отложенной выборке.
- Добавлены проверки и сообщения, чтобы скрипт не падал, если часть библиотек
  не установлена (Streamlit, Backtrader, XGBoost, LightGBM, CatBoost, GitPython,
  OpenAI/OpenRouter/Gemini и т.д.).
- НОВАЯ ФИЧА: Два ИИ для совместного анализа кода + интеграция линтеров (pylint, flake8).

Вы можете запускать main_ui() через Streamlit (если установлен):
  streamlit run mega_e_agent_full.py

Или использовать action_executor(command, state) напрямую из Python.
"""
from __future__ import annotations

import concurrent.futures
import importlib
import logging
import os
import pickle
import shlex
import subprocess
import sys
import threading
import zipfile
import tarfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, BinaryIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from numpy.random import SeedSequence
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ===================== Опциональные импорты =====================
try:  # Streamlit UI (опционально)
    import streamlit as st  # type: ignore
except ImportError:  # pragma: no cover
    st = None  # type: ignore

try:  # Backtesting (опционально)
    import backtrader as bt  # type: ignore
except ImportError:
    bt = None  # type: ignore

try:  # LLM провайдеры (опционально)
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

try:  # Буста-модели (опционально)
    import xgboost as xgb  # type: ignore
except ImportError:
    xgb = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except ImportError:
    lgb = None  # type: ignore

try:
    from catboost import CatBoostRegressor  # type: ignore
except ImportError:
    CatBoostRegressor = None  # type: ignore

try:  # Git (опционально)
    from git import Repo  # type: ignore
except ImportError:
    Repo = None  # type: ignore

# ===================== Константы и директории =====================
TRADING_CSV_PATH = "data/trading.csv"
POKER_CSV_PATH = "data/poker.csv"
STATE_FILE = "state.pkl"
LOG_FILE = "agent_log.txt"
MODEL_DIR = "data/models"
DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "extracted"
REPO_DIR = "repos"

# ===================== Настройка API-ключей =====================
API_KEYS = {
    "gemini": "AIzaSyCXe62p1nSphe8SsUPg-0_HUfqY6u17xJ4",
    "openrouter": "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", 
    "github": "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",
}

# OpenRouter клиент (опционально)
openrouter_client = None
if API_KEYS.get( "openrouter" ) and OpenAI is not None:
    try:
        openrouter_client = OpenAI(
            api_key=API_KEYS["openrouter"],
            base_url="https://openrouter.ai/api/v1",
        )
    except Exception:  # pragma: no cover
        pass

# Gemini конфиг (опционально)
if API_KEYS.get( "gemini" ) and genai is not None:
    try:
        genai.configure( api_key=API_KEYS["gemini"] )  # type: ignore
    except Exception:  # pragma: no cover
        pass

# ===================== Логирование =====================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def log(msg: str, level: str = "INFO") -> None:
    print( msg )
    level = level.upper()
    if hasattr( logging, level.lower() ):
        getattr( logging, level.lower() )( msg )
    else:
        logging.info( msg )


# ===================== RNG и синхронизация графиков =====================
_ss_global = SeedSequence( 42 )
_plot_lock = threading.Lock()


def _rng_child(seed_seq: Optional[SeedSequence] = None) -> np.random.Generator:
    """Создаёт независимый генератор для потока/задачи."""
    ss = seed_seq or _ss_global
    child = ss.spawn( 1 )[0]
    return np.random.default_rng( child )


# ===================== Вспомогательные функции (download/install) =====================

def download_file(url: str, save_dir: str = DOWNLOAD_DIR) -> str:
    Path( save_dir ).mkdir( parents=True, exist_ok=True )
    file_name = url.split( "/" )[-1] or f"download_{int( datetime.now( UTC ).timestamp() )}"
    save_path = Path( save_dir ) / file_name
    try:
        headers = {"User-Agent": "MegaE-Agent/1.0"}
        response = requests.get( url, stream=True, timeout=(10, 60), headers=headers, allow_redirects=True )
        response.raise_for_status()
        with open( save_path, "wb" ) as f:
            for chunk in response.iter_content( 8192 ):
                if chunk:
                    f.write( chunk )
        log( f"[OK] Скачан: {save_path}" )
        return str( save_path )
    except (requests.RequestException, ValueError, IOError) as e:
        log( f"[ERROR] Скачивание {url} не удалось: {e}", "ERROR" )
        return ""


def install_from_url(url: str):
    file_path = download_file( url )
    if not file_path:
        return f"[ERROR] Не удалось скачать {url}"
    try:
        subprocess.check_call( [sys.executable, "-m", "pip", "install", file_path] )
        log( f"[OK] Установлен пакет из {url}" )
        return True
    except (subprocess.CalledProcessError, ValueError, OSError) as e:
        log( f"[ERROR] Установка из {url} не удалась: {e}", "ERROR" )
        return False


def install_packages(packages: Dict[str, str]):
    """Устанавливает словарь {pip_name: import_name}. Возвращает словарь результатов."""
    results: Dict[str, bool] = {}
    for pkg, imp_name in packages.items():
        try:
            importlib.import_module( imp_name )
            results[pkg] = True
            log( f"[OK] {pkg} установлен" )
        except ImportError:
            log( f"[INSTALL] Попытка установить {pkg}..." )
            try:
                subprocess.check_call( [sys.executable, "-m", "pip", "install", pkg] )
                results[pkg] = True
                log( f"[OK] {pkg} установлен" )
            except (subprocess.CalledProcessError, ValueError, OSError) as e:
                results[pkg] = False
                log( f"[ERROR] Установка {pkg} не удалась: {e}", "ERROR" )
    return results


# ===================== Состояние =====================

def save_state(state: Dict[str, Any]):
    try:
        Path( STATE_FILE ).parent.mkdir( exist_ok=True )
        with open( STATE_FILE, "wb" ) as f:
            pickle.dump( state, f )
    except (IOError, ValueError, pickle.PicklingError) as e:
        log( f"[ERROR] Сохранение состояния не удалось: {e}", "ERROR" )


def load_state() -> Dict[str, Any]:
    try:
        if Path( STATE_FILE ).exists():
            with open( STATE_FILE, "rb" ) as f:
                return pickle.load( f )
    except (IOError, ValueError, pickle.UnpicklingError) as e:
        log( f"[ERROR] Загрузка состояния не удалась: {e}", "ERROR" )
    return {}


# ===================== AI Response (ChatGPT-стиль) =====================

def ai_response(prompt: str, provider: str = "openrouter") -> str:
    chatgpt_prompt = f"Отвечай как ChatGPT: кратко, дружелюбно, профессионально. Вопрос: {prompt}"
    try:
        if provider == "openrouter" and openrouter_client is not None:
            resp = openrouter_client.chat.completions.create(  # type: ignore[attr-defined]
                model="meta-llama/llama-3-70b-instruct",
                messages=[{"role": "user", "content": chatgpt_prompt}],
                temperature=0.2,
                max_tokens=1000,
            )
            choice = getattr( resp.choices[0], "message", getattr( resp.choices[0], "delta", None ) )
            content = getattr( choice, "content", None )
            return (content or "").strip() or str( resp )
        if provider == "gemini" and API_KEYS.get( "gemini" ) and genai is not None:
            model = genai.GenerativeModel( "gemini-1.5-flash" )  # type: ignore
            res = model.generate_content( chatgpt_prompt )
            return getattr( res, "text", "" ).strip()
        return f"[ERROR] {provider} не настроен"
    except (AttributeError, ValueError, RuntimeError) as e:
        return f"[AI ERROR] {provider}: {e}"


# ===================== Двойной AI анализ кода =====================
def dual_ai_code_review(code: str) -> str:
    """Совместный отзыв двух ИИ: OpenRouter генерирует анализ, Gemini проверяет/улучшает."""
    # Первый ИИ (OpenRouter) анализирует
    prompt1 = f"Проанализируй этот Python-код на ошибки, стиль, логику и предложи улучшения: \n{code[:2000]}"  # Ограничение длины
    review1 = ai_response( prompt1, "openrouter" )

    # Второй ИИ (Gemini) проверяет отзыв первого
    prompt2 = f"Проверь этот отзыв на код и улучши: найди пропущенные ошибки, предложи фиксы. Отзыв: {review1}\nКод: {code[:1000]}"
    review2 = ai_response( prompt2, "gemini" )

    return f"AI1 (OpenRouter): {review1}\n\nAI2 (Gemini, проверка): {review2}"


# ===================== Интеграция линтеров =====================
def run_linter(tool: str, code_path: str) -> str:
    """Вызывает линтер через subprocess, если установлен."""
    try:
        if tool == "pylint":
            result = subprocess.check_output( [sys.executable, "-m", "pylint", "--persistent=no", code_path],
                                              stderr=subprocess.STDOUT ).decode()
        elif tool == "flake8":
            result = subprocess.check_output( ["flake8", code_path], stderr=subprocess.STDOUT ).decode()
        else:
            return "[ERROR] Неподдерживаемый линтер"
        return result.strip() or "[OK] Нет ошибок"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"[ERROR] {tool} не установлен или ошибка: {e}. Установите: pip install {tool}"


# ===================== Анализ и рефакторинг кода =====================
def analyze_refactor_code(code_path: str) -> str:
    try:
        with open( code_path, "r", encoding="utf-8" ) as f:
            code = f.read()

        # Шаг 1: Линтеры
        pylint_report = run_linter( "pylint", code_path )
        flake8_report = run_linter( "flake8", code_path )

        # Шаг 2: Двойной ИИ-анализ
        ai_report = dual_ai_code_review( code )

        # Общий отчёт
        return f"Линтеры:\nPylint: {pylint_report}\nFlake8: {flake8_report}\n\nAI-анализ:\n{ai_report}"
    except (IOError, ValueError) as e:
        return f"[ERROR] Анализ {code_path}: {e}"


def self_check_code() -> str:
    """Самопроверка всего скрипта."""
    return analyze_refactor_code( __file__ )


# ===================== Скачивание и распаковка =====================

def _is_within(base: Path, target: Path) -> bool:
    try:
        return str( target.resolve() ).startswith( str( base.resolve() ) )
    except OSError:
        return False


def _safe_extract_zip(archive_path: str, extract_dir: str) -> None:
    base = Path( extract_dir )
    with zipfile.ZipFile( archive_path, "r" ) as z:
        for m in z.infolist():
            target = base / m.filename
            if not _is_within( base, target ):
                raise ValueError( "Zip traversal detected" )
        z.extractall( base )


def _safe_extract_tar(archive_path: str, extract_dir: str) -> None:
    base = Path( extract_dir )
    with tarfile.open( archive_path, "r:*" ) as t:
        for m in t.getmembers():
            target = base / m.name
            if not _is_within( base, target ):
                raise ValueError( "Tar traversal detected" )
        t.extractall( base )


def extract_archive(archive_path: str, extract_dir: str = EXTRACT_DIR) -> str:
    try:
        Path( extract_dir ).mkdir( parents=True, exist_ok=True )
        if archive_path.endswith( ".zip" ):
            _safe_extract_zip( archive_path, extract_dir )
        elif archive_path.endswith( (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz") ):
            _safe_extract_tar( archive_path, extract_dir )
        else:
            raise ValueError( "Неподдерживаемый формат архива" )
        log( f"[OK] Распакован: {archive_path} в {extract_dir}" )
        return extract_dir
    except (ValueError, IOError, OSError) as e:
        log( f"[ERROR] Распаковка {archive_path} не удалась: {e}", "ERROR" )
        return ""


# ===================== Генерация данных =====================

def generate_data(task: str, n_rows: int = 1000) -> Dict[str, Any]:
    rng = _rng_child()
    if task.lower() == "trading":
        dates = pd.date_range( start="2020-01-01", periods=n_rows, freq="B" )
        opens = rng.normal( 100, 10, n_rows ).cumsum()
        highs = opens + rng.uniform( 0, 5, n_rows )
        lows = opens - rng.uniform( 0, 5, n_rows )
        closes = opens + rng.normal( 0, 2, n_rows )
        volumes = rng.integers( 1000, 10000, n_rows )
        df = pd.DataFrame( {
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes
        } )
        path = TRADING_CSV_PATH
    elif task.lower() == "poker":
        hands = [f"Hand {i}" for i in range( n_rows )]
        outcomes = rng.choice( ["Win", "Loss", "Fold"], n_rows )
        bets = rng.uniform( 10, 100, n_rows )
        df = pd.DataFrame( {
            "Hand": hands,
            "Outcome": outcomes,
            "Bet": bets
        } )
        path = POKER_CSV_PATH
    else:
        return {"error": "Неподдерживаемая задача"}

    Path( path ).parent.mkdir( parents=True, exist_ok=True )
    df.to_csv( path, index=False )
    return {"task": task, "data": df, "path": path}


# ===================== Обучение моделей =====================

def train_models(data_path: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    try:
        df = pd.read_csv( data_path )
        if "Close" not in df.columns:
            raise ValueError( "Нет колонки 'Close'" )

        features = ["Open", "High", "Low", "Volume"] if "Volume" in df.columns else ["Open", "High", "Low"]
        X = df[features]
        y = df["Close"]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform( X )

        tscv = TimeSeriesSplit( n_splits=5 )
        models = []
        scores = {}

        for name, model_cls in [
            ("RF", RandomForestRegressor),
            ("MLP", MLPRegressor),
            ("XGB", xgb.XGBRegressor if xgb else None),
            ("LGB", lgb.LGBMRegressor if lgb else None),
            ("Cat", CatBoostRegressor if CatBoostRegressor else None),
        ]:
            if model_cls is None:
                continue
            try:
                model = model_cls( random_state=42 )
                train_idx, test_idx = list( tscv.split( X ) )[-1]
                X_train, X_test = x_scaled[train_idx], x_scaled[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model.fit( X_train, y_train )
                preds = model.predict( X_test )
                mse = mean_squared_error( y_test, preds )
                r2 = r2_score( y_test, preds )
                scores[name] = {"mse": mse, "r2": r2}

                Path( MODEL_DIR ).mkdir( parents=True, exist_ok=True )
                model_path = Path( MODEL_DIR ) / f"{name}_{Path( data_path ).stem}.joblib"
                joblib.dump( model, model_path )
                models.append( str( model_path ) )
            except (ValueError, RuntimeError) as e:
                log( f"[ERROR] Обучение {name}: {e}", "ERROR" )

        return models, scores
    except (ValueError, IOError, KeyError) as e:
        log( f"[ERROR] Обучение на {data_path}: {e}", "ERROR" )
        return [], {}


# ===================== Бэктестинг (Backtrader) =====================

def parallel_backtest(data_path: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    if bt is None:
        return []
    results = []
    strategies = {
        "SMA": bt.strategies.SmaCross,
        "RSI": bt.strategies.RSI,
    }
    with concurrent.futures.ThreadPoolExecutor( max_workers=2 ) as executor:
        futures = {executor.submit( _run_backtest, data_path, strat ): name for name, strat in strategies.items()}
        for fut in concurrent.futures.as_completed( futures ):
            try:
                res = fut.result()
                results.append( (futures[fut], res) )
            except (ValueError, RuntimeError) as e:
                log( f"[ERROR] Бэктест {futures[fut]}: {e}", "ERROR" )
    return results


def _run_backtest(data_path: str, strategy: Callable) -> Tuple[float, float, float]:
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData( dataname=pd.read_csv( data_path, parse_dates=["Date"], index_col="Date" ) )
    cerebro.adddata( data )
    cerebro.addstrategy( strategy )
    cerebro.broker.setcash( 10000.0 )
    cerebro.run()
    final_value = cerebro.broker.getvalue()
    returns = (final_value - 10000) / 10000 * 100
    max_drawdown = 0  # Placeholder, add proper calculation if needed
    return final_value, returns, max_drawdown


# ===================== Симуляции стратегий =====================

def simulate_trading_strategy(data_path: str) -> Tuple[float, float]:
    try:
        df = pd.read_csv( data_path )
        if df.empty or "Close" not in df.columns or "Open" not in df.columns:
            return 1000.0, 0.0
        
        balance = 1000.0
        wins = 0
        for _, row in df.iterrows():
            if row["Close"] > row["Open"]:
                balance += balance * 0.01
                wins += 1
            else:
                balance -= balance * 0.01
        win_rate = wins / len( df ) if len( df ) > 0 else 0
        return balance, win_rate
    except Exception as e:
        log( f"[ERROR] Симуляция торговли: {e}", "ERROR" )
        return 1000.0, 0.0


def simulate_poker_strategy(data_path: str) -> Tuple[int, float]:
    try:
        df = pd.read_csv( data_path )
        if df.empty or "Outcome" not in df.columns:
            return 0, 0.0
        
        wins = (df["Outcome"] == "Win").sum()
        win_rate = wins / len( df ) if len( df ) > 0 else 0
        return wins, win_rate
    except Exception as e:
        log( f"[ERROR] Симуляция покера: {e}", "ERROR" )
        return 0, 0.0


# ===================== Monte Carlo =====================

def monte_carlo_trading(start_balance: float, prices: List[Tuple[float, float]], n_sim: int = 1000) -> List[float]:
    if not prices or n_sim <= 0:
        return [start_balance]
    
    rng = _rng_child()
    final_balances = []
    for _ in range( n_sim ):
        balance = start_balance
        for open_p, close_p in prices:
            if open_p > 0:  # Защита от деления на ноль
                if rng.random() > 0.5:
                    balance *= (close_p / open_p)
        final_balances.append( balance )
    return final_balances


def plot_monte_carlo(balances: List[float]) -> None:
    if not balances:
        return
    
    with _plot_lock:
        try:
            plt.figure(figsize=(10, 6))
            plt.hist( balances, bins=50, alpha=0.7, edgecolor='black' )
            plt.title( "Monte Carlo Simulation Results" )
            plt.xlabel( "Final Balance" )
            plt.ylabel( "Frequency" )
            plt.grid(True, alpha=0.3)
            plt.savefig( "monte_carlo.png", dpi=300, bbox_inches='tight' )
            plt.close()
        except Exception as e:
            log( f"[ERROR] Ошибка построения графика: {e}", "ERROR" )


# ===================== Поиск файлов =====================

def find_csv_files(directory: str = ".") -> List[str]:
    try:
        return [str( p ) for p in Path( directory ).rglob( "*.csv" ) if p.is_file()]
    except Exception as e:
        log( f"[ERROR] Поиск CSV файлов: {e}", "ERROR" )
        return []


# ===================== Git операции =====================

def git_commit_push(repo_path: str, commit_msg: str) -> str:
    if Repo is None:
        return "[ERROR] Git не установлен"
    try:
        repo = Repo( repo_path )
        repo.git.add( all=True )
        repo.index.commit( commit_msg )
        origin = repo.remote( name="origin" )
        origin.push()
        return "[OK] Коммит и пуш завершены"
    except (ValueError, OSError, RuntimeError) as e:
        return f"[ERROR] Git: {e}"


# ===================== Парсинг команд =====================

def _last_token(tokens: List[str]) -> str:
    return tokens[-1] if tokens else ""


def _to_task_tuple(func: Callable, *args: Any, **kwargs: Any) -> Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]:
    return func, args, kwargs


def parse_command(command: str) -> Tuple[List[Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]], str, List[str]]:
    output = ""
    tasks: List[Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]] = []
    command_lower = command.lower()
    tokens = shlex.split( command )

    commands: Dict[str, Callable[[], Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]]] = {
        "скачай": lambda: _to_task_tuple( download_file, tokens[1] if len( tokens ) > 1 else "" ),
        "распакуй": lambda: _to_task_tuple( extract_archive, _last_token( tokens ) if tokens else "" ),
        "установи": lambda: _to_task_tuple( install_packages,
                                            dict( zip( tokens[1::2], tokens[2::2] ) ) if len( tokens ) > 1 else {} ),
        "сгенерируй данные": lambda: _to_task_tuple(
            generate_data,
            tokens[2] if len( tokens ) > 2 else "trading",
            int( tokens[3] ) if len( tokens ) > 3 and tokens[3].isdigit() else 1000
        ),
        "обучи модели": lambda: _to_task_tuple(
            train_models,
            _last_token( tokens ) if tokens and Path(
                _last_token( tokens ) ).exists() else TRADING_CSV_PATH
        ),
        "бэктест": lambda: _to_task_tuple(
            parallel_backtest,
            _last_token( tokens ) if tokens and Path(
                _last_token( tokens ) ).exists() else TRADING_CSV_PATH
        ),
        "тест трейдинг": lambda: _to_task_tuple(
            simulate_trading_strategy,
            _last_token( tokens ) if tokens and Path(
                _last_token( tokens ) ).exists() else TRADING_CSV_PATH
        ),
        "тест покер": lambda: _to_task_tuple( simulate_poker_strategy, POKER_CSV_PATH ),
        "монте карло": lambda: _to_task_tuple(
            monte_carlo_trading,
            1000,
            [],  # подставится при исполнении из CSV
            int( tokens[-2] ) if len( tokens ) > 1 and tokens[-2].isdigit() else 1000
        ),
        "рефакторинг": lambda: _to_task_tuple( analyze_refactor_code, _last_token( tokens ) if tokens else "." ),
        "анализ кода": lambda: _to_task_tuple( analyze_refactor_code, _last_token( tokens ) if tokens else "." ),
        "проверь код": lambda: _to_task_tuple( self_check_code ),
        "git": lambda: _to_task_tuple(
            git_commit_push,
            _last_token( tokens ) if tokens else ".",
            " ".join( tokens[1:-1] ) if len( tokens ) > 2 else "Update by AI"
        ),
    }

    for key, builder in commands.items():
        if key in command_lower:
            try:
                task = builder()
                tasks.append( task )
            except (ValueError, IndexError) as e:
                output += f"[ERROR] Обработка команды '{key}': {e}\n"

    if "и найди csv" in command_lower:
        tasks.append( _to_task_tuple( find_csv_files ) )

    if ("?" in command) or ("что" in command_lower) or ("как" in command_lower):
        provider = "gemini" if "gemini" in command_lower else "openrouter"
        output += f"[AI] {ai_response( command, provider )}\n"

    csv_files = find_csv_files() if "и найди csv" in command_lower else []
    return tasks, output, csv_files


# ===================== Parallel Executor =====================
class ParallelExecutor:
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers

    def run_parallel(self, func_list: List[Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]]) -> List[Tuple[str, Any]]:
        results: List[Tuple[str, Any]] = []
        if not func_list:
            return results
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min( self.max_workers, max( 1, len( func_list ) ) ) ) as executor:
            fut_map = {executor.submit( func, *args, **kwargs ): func for func, args, kwargs in func_list}
            for fut in concurrent.futures.as_completed( fut_map ):
                func = fut_map[fut]
                try:
                    res = fut.result()
                    results.append( (func.__name__, res) )
                    log( f"[OK] {func.__name__} завершена" )
                except (ValueError, RuntimeError, Exception) as e:
                    log( f"[ERROR] {func.__name__}: {e}", "ERROR" )
                    results.append( (func.__name__, None) )
        return results


parallel_executor = ParallelExecutor()


# ===================== Action Executor =====================

def _load_prices_for_monte(data_path: str) -> List[Tuple[float, float]]:
    try:
        df = pd.read_csv( data_path )
        if not {"Open", "Close"}.issubset( df.columns ):
            return []
        return list( zip( df["Open"].tolist(), df["Close"].tolist() ) )
    except (IOError, ValueError, KeyError):
        return []


def action_executor(command: str, state: Dict[str, Any]) -> str:
    output = ""
    try:
        tasks, cmd_output, csv_files = parse_command( command )
        output += cmd_output
        # Подмена пустых цен для Монте-Карло
        fixed_tasks: List[Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]] = []
        for func, args, kwargs in tasks:
            if func == monte_carlo_trading:
                start_balance = args[0] if len( args ) > 0 else 1000
                prices_arg = args[1] if len( args ) > 1 else []
                n_sim = args[2] if len( args ) > 2 else 1000
                if not prices_arg:
                    data_path = TRADING_CSV_PATH if Path( TRADING_CSV_PATH ).exists() else (
                        csv_files[0] if csv_files else TRADING_CSV_PATH)
                    prices_arg = _load_prices_for_monte( data_path )
                fixed_tasks.append( (func, (start_balance, prices_arg, n_sim), kwargs) )
            else:
                fixed_tasks.append( (func, args, kwargs) )

        if fixed_tasks:
            results = parallel_executor.run_parallel( fixed_tasks )
            for name, res in results:
                if res is None:
                    continue
                if name == "monte_carlo_trading":
                    output += f"Monte Carlo: mean {np.mean( res ):.2f}, std {np.std( res ):.2f}\n"
                    try:
                        plot_monte_carlo( res )
                    except (ValueError, RuntimeError):
                        pass
                elif name == "train_models":
                    output += f"Модели обучены: {res[1]}\n"
                elif name == "parallel_backtest":
                    for strat_name, (final_value, returns, max_drawdown) in res:
                        output += (
                            f"Backtrader ({strat_name}): Баланс: {final_value:.2f}, "
                            f"Доход: {returns:.2f}%, Просадка: {max_drawdown:.2f}%\n"
                        )
                elif name == "simulate_trading_strategy":
                    output += f"Трейдинг: Баланс: {res[0]:.2f}, Win rate: {res[1]:.3f}\n"
                elif name == "simulate_poker_strategy":
                    output += f"Покер: Победы: {res[0]}, Win rate: {res[1]:.3f}\n"
                elif name == "generate_data":
                    if isinstance( res, dict ):
                        output += f"{res.get( 'task', 'data' ).capitalize()} данные: {len( res.get( 'data', [] ) )} строк\n"
                elif name == "find_csv_files":
                    output += f"Найдены CSV: {res}\n"
                elif name == "analyze_refactor_code" or name == "self_check_code":
                    output += f"Анализ кода: {res}\n"
                else:
                    output += f"[OK] {name}: {res}\n"
        # Автоматическое обучение/бэктестинг на CSV
        if csv_files and "и обучи" in command.lower():
            tasks2: List[Tuple[Callable, Tuple[Any, ...], Dict[str, Any]]] = [(train_models, (csv,), {}) for csv in
                                                                              csv_files]
            if "и бэктест" in command.lower():
                tasks2.extend( [(parallel_backtest, (csv,), {}) for csv in csv_files] )
            results2 = parallel_executor.run_parallel( tasks2 )
            for name, res in results2:
                if res:
                    if name == "train_models":
                        output += f"Модели на CSV: {res[1]}\n"
                    elif name == "parallel_backtest":
                        for strat_name, (final_value, returns, max_drawdown) in res:
                            output += (
                                f"Backtrader на CSV ({strat_name}): Баланс: {final_value:.2f}, "
                                f"Доход: {returns:.2f}%, Просадка: {max_drawdown:.2f}%\n"
                            )
    except (ValueError, RuntimeError, Exception) as e:
        output += f"[ERROR] {e}\n"
    state.setdefault( "messages", [] ).append( {"command": command, "output": output} )
    save_state( state )
    return output


# ===================== Streamlit UI =====================

def main_ui():  # pragma: no cover (UI)
    if st is None:
        print( "Streamlit не найден. Запустите main_ui() только если streamlit установлен и вы в среде Streamlit." )
        return
    st.title( "Mega E-Agent ⚡ (Версия 1.5)" )
    state = load_state()
    state.setdefault( "messages", [] )
    st.subheader( "Команда или вопрос:" )
    user_command = st.text_area( "Ввод" )
    if st.button( "Выполнить" ):
        output = action_executor( user_command, state )
        st.text_area( "Результат", output, height=300 )
    st.subheader( "История:" )
    for msg in state["messages"][-10:]:
        st.markdown( f"**Команда:** {msg['command']}\n**Вывод:**\n```\n{msg['output']}\n```" )
    st.sidebar.title( "Быстрые действия" )
    for label, cmd in [
        ("Скачать и распаковать", "скачай и распакуй https://example.com/data.zip и найди csv и обучи и бэктест"),
        ("Генерировать trading", "сгенерируй данные trading"),
        ("Генерировать и обучить", "сгенерируй данные trading и обучи и бэктест"),
        ("Генерировать poker", "сгенерируй данные poker"),
        ("Обучить модели", "обучи модели"),
        ("Бэктест (Backtrader)", "бэктест"),
        ("Тест трейдинга", "тест трейдинг"),
        ("Тест покера", "тест покер"),
        ("Monte Carlo", "монте карло"),
        ("Рефакторинг", "рефакторинг mega_e_agent_full.py"),
        ("Проверить код", "проверь код"),
        ("Поиск репо", "поиск репо python trading"),
    ]:
        if st.sidebar.button( label ):
            action_executor( cmd, state )


if __name__ == "__main__":  # pragma: no cover
    main_ui()