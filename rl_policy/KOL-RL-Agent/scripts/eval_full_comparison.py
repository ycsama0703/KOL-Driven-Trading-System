"""One-click evaluation for all KOLs: export decisions, compute metrics, plot four curves (Baseline, Trained, Free, SPY).

Baseline = KOL 基线（严格按文本信号）；Trained = 残差/衰减策略；Free = 取消基线约束，仅用 Actor 输出；SPY = 市场基准。
需要已训练好的最新 checkpoint 存在 outputs/<KOL>_residual/<KOL>_*/checkpoints/policy.pt。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def run_cmd(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def parse_portfolio(text: str) -> Dict[str, float]:
    res: Dict[str, float] = {}
    if not isinstance(text, str) or not text:
        return res
    for item in text.split(";"):
        if not item:
            continue
        try:
            t, w = item.split(":")
            res[t] = float(w)
        except ValueError:
            continue
    return res


def turnover(before: str, after: str) -> float:
    b = parse_portfolio(before)
    a = parse_portfolio(after)
    tickers = set(b) | set(a)
    delta = sum(abs(a.get(t, 0.0) - b.get(t, 0.0)) for t in tickers)
    return delta / 2.0


def metrics_from_equity(eq: pd.Series) -> Dict[str, float]:
    ret = eq.pct_change().dropna()
    cumulative = float(eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    sharpe = float(ret.mean() / (ret.std() + 1e-8) * np.sqrt(252)) if len(ret) > 1 else 0.0
    vol = float(ret.std()) if len(ret) > 0 else 0.0
    peak = eq.cummax()
    mdd = float(((peak - eq) / (peak + 1e-8)).max()) if len(eq) > 0 else 0.0
    return {"cumulative_return": cumulative, "sharpe": sharpe, "max_drawdown": mdd, "volatility": vol}


def compute_turnover(df: pd.DataFrame, before_col: str, after_col: str) -> float:
    if before_col not in df or after_col not in df:
        return 0.0
    tps = df.apply(lambda r: turnover(r[before_col], r[after_col]), axis=1)
    return float(tps.mean()) if len(tps) else 0.0


def fidelity_deviation(df: pd.DataFrame, action_col: str) -> float:
    """Fraction of signals where actions differ from baseline_actions."""
    if "baseline_actions" not in df or action_col not in df:
        return 0.0
    diff = (df["baseline_actions"] != df[action_col]).mean()
    return float(diff)


def ensure_decisions(run_path: Path, kol: str, residual_scale: float, decay_scale: float, max_weight: float, hold_decay: float) -> Tuple[Path, Path]:
    ckpt = run_path / "checkpoints" / "policy.pt"
    reward_csv = Path("data/processed/reward") / kol / "test.csv"
    vocab = Path("models/embedding/ticker_vocab.json")
    emb = Path("models/embedding/ticker_embedding.pt")
    sig_res = run_path / "signal_decisions_test.csv"
    sig_free = run_path / "signal_decisions_test_free.csv"
    if not sig_res.exists():
        run_cmd([
            sys.executable,
            "experiments/residual_sweep/export_decisions_residual.py",
            "--checkpoint",
            str(ckpt),
            "--reward-csv",
            str(reward_csv),
            "--vocab-path",
            str(vocab),
            "--embedding-path",
            str(emb),
            "--output",
            str(sig_res),
            "--residual-scale",
            str(residual_scale),
            "--decay-scale",
            str(decay_scale),
            "--max-weight",
            str(max_weight),
            "--hold-decay",
            str(hold_decay),
            "--mode",
            "residual",
        ])
    if not sig_free.exists():
        run_cmd([
            sys.executable,
            "experiments/residual_sweep/export_decisions_residual.py",
            "--checkpoint",
            str(ckpt),
            "--reward-csv",
            str(reward_csv),
            "--vocab-path",
            str(vocab),
            "--embedding-path",
            str(emb),
            "--output",
            str(sig_free),
            "--residual-scale",
            str(residual_scale),
            "--decay-scale",
            str(decay_scale),
            "--max-weight",
            str(max_weight),
            "--hold-decay",
            str(hold_decay),
            "--mode",
            "free",
        ])
    return sig_res, sig_free


def fetch_spy_equity(start, end) -> pd.Series:
    spy = yf.download("SPY", start=start, end=end, progress=False)
    col = None
    for key in spy.columns:
        name = key[-1] if isinstance(key, tuple) else key
        if str(name).lower() == "adj close":
            col = key
            break
    if col is None:
        for key in spy.columns:
            name = key[-1] if isinstance(key, tuple) else key
            if "close" in str(name).lower():
                col = key
                break
    if col is None:
        col = spy.select_dtypes(include="number").columns[0]
    adj = spy[col]
    return (1 + adj.pct_change().fillna(0)).cumprod()


def plot_curves(kol: str, run: Path, res_df: pd.DataFrame, free_df: pd.DataFrame, spy_equity: pd.Series) -> None:
    # Rebase to 1.0 for visual comparability
    b0 = res_df["equity_baseline"].iloc[0]
    t0 = res_df["equity_trained"].iloc[0]
    f0 = free_df["equity_trained"].iloc[0]
    s0 = spy_equity.iloc[0] if len(spy_equity) else 1.0

    plt.figure(figsize=(10, 5))
    norm_baseline = res_df["equity_baseline"] / (b0 + 1e-9)
    plt.plot(res_df["date"], norm_baseline, "k--", label="baseline1_KOL")
    plt.plot(res_df["date"], res_df["equity_trained"] / (t0 + 1e-9), label="Trained (residual)")
    plt.plot(free_df["date"], free_df["equity_trained"] / (f0 + 1e-9), label="baseline2_free")
    plt.plot(spy_equity.index, spy_equity.values / (s0 + 1e-9), "g-.", label="SPY")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(f"{kol}: Baseline vs Trained vs Free vs SPY (Test)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = run / "equity_with_spy_free.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline/trained/free for all KOLs and plot.")
    parser.add_argument("--kols", nargs="*", default=["Everything_Money", "Invest_with_Henry", "MarketBeat"])
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--decay-scale", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=0.2)
    parser.add_argument("--hold-decay", type=float, default=0.99)
    args = parser.parse_args()

    for kol in args.kols:
        base = Path("outputs") / f"{kol}_residual"
        runs = sorted(base.glob(f"{kol}_*"), key=lambda p: p.name, reverse=True)
        if not runs:
            print(f"[SKIP] {kol}: no runs found")
            continue
        latest = runs[0]
        print(f"[INFO] {kol}: latest run {latest}")
        sig_res, sig_free = ensure_decisions(
            latest,
            kol,
            residual_scale=args.residual_scale,
            decay_scale=args.decay_scale,
            max_weight=args.max_weight,
            hold_decay=args.hold_decay,
        )

        res_df = pd.read_csv(sig_res, parse_dates=["date"]).sort_values("date")
        free_df = pd.read_csv(sig_free, parse_dates=["date"]).sort_values("date")
        spy_equity = fetch_spy_equity(res_df["date"].min(), res_df["date"].max())

        # Metrics
        metrics = {
            "baseline": metrics_from_equity(res_df["equity_baseline"]),
            "trained": metrics_from_equity(res_df["equity_trained"]),
            "free": metrics_from_equity(free_df["equity_trained"]),
            "spy": metrics_from_equity(pd.Series(spy_equity.values, index=spy_equity.index)),
        }
        # Turnover & fidelity
        metrics["baseline"]["turnover"] = compute_turnover(res_df, "portfolio_before_baseline", "portfolio_after_baseline")
        metrics["trained"]["turnover"] = compute_turnover(res_df, "portfolio_before_trained", "portfolio_after_trained")
        metrics["free"]["turnover"] = compute_turnover(free_df, "portfolio_before_trained", "portfolio_after_trained")
        metrics["trained"]["fidelity_deviation"] = fidelity_deviation(res_df, "trained_actions")
        metrics["free"]["fidelity_deviation"] = fidelity_deviation(free_df, "trained_actions")
        metrics["spy"]["turnover"] = 0.0
        metrics["spy"]["fidelity_deviation"] = 0.0

        with open(latest / "metrics_full.json", "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"[METRICS] Saved {latest/'metrics_full.json'}")

        plot_curves(kol, latest, res_df, free_df, spy_equity)


if __name__ == "__main__":
    main()
