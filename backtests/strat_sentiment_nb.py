"""Strategy 3 - Sentiment analysis, Bernoulli naive Bayes (book section 18.3).

The classifier is implemented directly from the book's equations:

    P(C_a | X) ~ P(C_a) * prod_i prod_w  P(w|C_a)^{x_iw} (1-P(w|C_a))^{1-x_iw}   (546)

with conditional word probabilities P(w|C_a) estimated from training frequencies
(Laplace-smoothed), classes C_a = {next-day BTC down, up}, and the trade rule
"predicted up -> long, predicted down -> short".

DATA CAVEAT (important / honest)
--------------------------------
A historical archive of BTC tweets 2022-2026 is NOT reachable from this environment,
so we CANNOT measure a real-world sentiment edge. Instead this is a *controlled
simulation* on the REAL BTC price series: each day we synthesize a bag of words whose
content carries a TUNABLE fraction `signal` of true information about the next day's
move. We sweep `signal` to show the pipeline:
  * recovers embedded signal proportionally (sanity check the math is correct), and
  * earns ~0 with pure-noise tweets (signal=0) -> no spurious edge.
This validates the section-18.3 machinery; it is NOT evidence the strategy makes
money on live Twitter data.
"""
import numpy as np
import pandas as pd

from backtest_engine import load_closes, perf_metrics, apply_costs, fmt

COST_BPS = 10.0
TRAIN_WIN = 365
REFIT_EVERY = 30
RNG = np.random.default_rng(7)

BULL = ["moon", "pump", "buy", "bull", "long", "breakout", "rally", "hodl"]
BEAR = ["dump", "crash", "sell", "bear", "short", "rug", "fear", "capitulate"]
NEUTRAL = [f"w{i}" for i in range(20)]          # filler / stop-words
VOCAB = BULL + BEAR + NEUTRAL
W2I = {w: i for i, w in enumerate(VOCAB)}


def synth_day_features(up, signal):
    """One day's Bernoulli word-presence vector over VOCAB.

    With prob `signal` the day leans bull/bear consistent with the true `up` label;
    otherwise words are random noise. Neutral filler is always sprinkled in.
    """
    x = np.zeros(len(VOCAB))
    for w in RNG.choice(NEUTRAL, size=5, replace=False):
        x[W2I[w]] = 1
    informed = RNG.random() < signal
    pool = (BULL if up else BEAR) if informed else (BULL + BEAR)
    for w in RNG.choice(pool, size=3, replace=False):
        x[W2I[w]] = 1
    return x


class BernoulliNB546:
    """Bernoulli naive Bayes exactly per eqs. 543-546 (log-space, Laplace smoothing)."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.logprior_, self.logp_, self.log1mp_ = {}, {}, {}
        for c in self.classes_:
            Xc = X[y == c]
            p = (Xc.sum(axis=0) + 1.0) / (len(Xc) + 2.0)   # P(w|C_a), smoothed
            self.logprior_[c] = np.log(len(Xc) / len(X))
            self.logp_[c] = np.log(p)
            self.log1mp_[c] = np.log(1.0 - p)
        return self

    def predict(self, X):
        scores = []
        for c in self.classes_:
            s = self.logprior_[c] + X @ self.logp_[c] + (1 - X) @ self.log1mp_[c]
            scores.append(s)
        return self.classes_[np.argmax(np.vstack(scores), axis=0)]


def run(btc, signal):
    r = btc.pct_change()
    up = (r.shift(-1) > 0).astype(int)              # next-day label
    dates = btc.index[1:-1]
    X = np.vstack([synth_day_features(bool(up.loc[d]), signal) for d in dates])
    y = up.loc[dates].values

    pos = pd.Series(0.0, index=dates)
    start = TRAIN_WIN
    while start < len(dates):
        te_hi = min(start + REFIT_EVERY, len(dates))
        clf = BernoulliNB546().fit(X[start - TRAIN_WIN:start], y[start - TRAIN_WIN:start])
        pred = clf.predict(X[start:te_hi])
        pos.iloc[start:te_hi] = np.where(pred == 1, 1.0, -1.0)
        start += REFIT_EVERY

    pos_df = pos.to_frame("BTC")
    ret_df = r.reindex(pos.index).to_frame("BTC")
    net, turn = apply_costs(pos_df, ret_df, COST_BPS)
    # directional accuracy on the held-out (post-train) days
    active = pos != 0
    nxt = r.shift(-1).reindex(pos.index)
    acc = (np.sign(pos[active]) == np.sign(nxt[active])).mean()
    return net, acc


def main():
    btc = load_closes()["BTC"].dropna()
    print("Controlled simulation on REAL BTC prices (synthetic tweet stream).")
    print("signal = fraction of days whose tweets actually carry true information.\n")
    for sig in [0.0, 0.15, 0.35, 0.60]:
        net, acc = run(btc, sig)
        m = perf_metrics(net)
        tag = "(pure noise)" if sig == 0 else ""
        print(f"signal={sig:.2f} {tag:12s} dir.acc={acc*100:4.1f}%  {fmt(m)}")
    print("\nReading: at signal=0 the strategy is ~flat/negative after costs (no edge);")
    print("as real information rises, accuracy and Sharpe climb -> pipeline is correct.")


if __name__ == "__main__":
    main()
