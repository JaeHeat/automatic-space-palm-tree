"""Strategy 2 - Cryptocurrency ANN quantile classifier (book section 18.2).

Faithful to the book's design, adapted from 15-min bars to daily BTC bars:

  * Inputs  : normalized return R-hat, plus EMA / EMSD / RSI at several windows
              (eqs. 521-529).
  * Output  : K classes = quantile buckets of the *next-day* normalized return
              (eq. 530); network outputs class probabilities via softmax.
  * Network : MLP with ReLU hidden layers + softmax output, trained by
              cross-entropy (eqs. 532-536) -> sklearn MLPClassifier.
  * Trade   : buy iff predicted class = top quantile, sell iff bottom (eq. 537).

Rigor: strict walk-forward. The model is retrained on a rolling 365-day window and
predicts the next 30 days; quantile thresholds and feature scaling are fit on TRAIN
only. Positions decided at close t earn the t->t+1 return; costs charged on turnover.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from backtest_engine import load_closes, perf_metrics, apply_costs, fmt

warnings.filterwarnings("ignore")

COST_BPS = 10.0
TRAIN_WIN = 365          # rolling training window (days)
REFIT_EVERY = 30         # walk-forward step (days)
K = 5                    # number of return quantiles
EMA_WINS = [3, 7, 14, 30]
RSI_WINS = [7, 14]
SEED = 42


def features(close):
    """Build the section-18.2 feature set on a daily BTC close series."""
    r = close.pct_change()
    vol = r.rolling(30).std()
    rhat = (r - r.rolling(30).mean()) / vol          # normalized return (524)
    feat = {"rhat": rhat}
    for tau in EMA_WINS:
        lam = (tau - 1) / (tau + 1)
        ema = rhat.ewm(alpha=1 - lam).mean()         # EMA (526)
        emsd = rhat.ewm(alpha=1 - lam).std()         # EMSD (527)
        feat[f"ema{tau}"] = ema
        feat[f"emsd{tau}"] = emsd
    for tau in RSI_WINS:                             # RSI (528-529)
        up = rhat.clip(lower=0).rolling(tau).sum()
        dn = (-rhat.clip(upper=0)).rolling(tau).sum()
        feat[f"rsi{tau}"] = up / (up + dn)
    X = pd.DataFrame(feat)
    target_norm = rhat.shift(-1)                     # next-day normalized return
    return X, target_norm, r


def run(close):
    X, target_norm, r = features(close)
    idx = X.dropna().index
    X = X.loc[idx]
    target_norm = target_norm.loc[idx]

    positions = pd.Series(0.0, index=X.index)
    dates = X.index
    start = TRAIN_WIN
    while start < len(dates):
        tr_lo, tr_hi = start - TRAIN_WIN, start
        te_hi = min(start + REFIT_EVERY, len(dates))
        tr_idx = dates[tr_lo:tr_hi]
        te_idx = dates[tr_hi if False else start:te_hi]

        ytr_raw = target_norm.loc[tr_idx]
        # quantile thresholds from TRAIN only -> class labels 0..K-1
        qs = np.quantile(ytr_raw.dropna(), np.linspace(0, 1, K + 1)[1:-1])
        ytr = np.digitize(ytr_raw, qs)
        Xtr = X.loc[tr_idx]
        good = ytr_raw.notna().values
        Xtr, ytr = Xtr[good], ytr[good]
        if len(np.unique(ytr)) < 2:
            start += REFIT_EVERY
            continue

        scaler = StandardScaler().fit(Xtr)
        clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu",
                            solver="adam", alpha=1e-3, max_iter=400,
                            random_state=SEED)
        clf.fit(scaler.transform(Xtr), ytr)

        Xte = X.loc[te_idx]
        pred = clf.predict(scaler.transform(Xte))
        pos = np.where(pred == K - 1, 1.0, np.where(pred == 0, -1.0, 0.0))
        positions.loc[te_idx] = pos
        start += REFIT_EVERY

    pos_df = positions.to_frame("BTC")
    ret_df = r.loc[positions.index].to_frame("BTC")
    net, turnover = apply_costs(pos_df, ret_df, COST_BPS)
    return net, turnover, positions


def main():
    closes = load_closes()
    btc = closes["BTC"].dropna()
    net, turn, pos = run(btc)
    m = perf_metrics(net)
    # directional accuracy on days we actually held a position
    nxt = btc.pct_change().shift(-1).reindex(pos.index)
    active = pos != 0
    correct = (np.sign(pos[active]) == np.sign(nxt[active]))
    dir_acc = correct.mean() if active.any() else float("nan")
    print(f"ANN long/short (top/bottom quantile)  {fmt(m)}")
    print(f"  exposure: long {100*(pos>0).mean():.0f}% of days, "
          f"short {100*(pos<0).mean():.0f}%, flat {100*(pos==0).mean():.0f}%  "
          f"| avg turnover/day={turn.mean():.3f}")
    print(f"  directional accuracy on active days = {dir_acc*100:.1f}%  "
          f"(coin-flip = 50%)")

    bench = perf_metrics(btc.pct_change().loc[net.index])
    print(f"Buy&hold BTC (same span)              {fmt(bench)}")
    return m, bench


if __name__ == "__main__":
    main()
