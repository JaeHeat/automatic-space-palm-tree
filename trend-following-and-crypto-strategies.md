# Deep Dive: Trend Following & Crypto Strategies

Detailed write-ups of three strategies from **_151 Trading Strategies_**
(Kakushadze & Serur, 2018): futures **trend following** (§10.4) and the two
cryptocurrency strategies — **ANN** (§18.2) and **naïve Bayes sentiment** (§18.3).

Source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3247865 ·
full text: https://arxiv.org/abs/1912.04492

> Equation numbers in parentheses refer to the source paper. Notation follows the book.

---

## 1. Trend Following / Momentum — Futures (§10.4)

**Idea.** Go long futures with positive recent returns and short those with
negative recent returns, sizing each position inversely to its volatility. This is
the classic time-series/cross-sectional momentum construction (cf. Moskowitz–Ooi–Pedersen
2012; Balta–Kosowski 2013).

### Core construction

Let $R_i$ be the return of futures contract $i = 1, \dots, N$ over a past lookback
period $T$ (measured in days, weeks, or months). The portfolio weights are:

$$w_i = \gamma\,\frac{\eta_i}{\sigma_i} \qquad (474)$$

$$\eta_i = \operatorname{sign}(R_i) \qquad (475)$$

where $\sigma_i$ is the historical volatility (over $T$ or another window), and
$\gamma > 0$ is fixed by the normalization

$$\sum_{i=1}^{N} |w_i| = 1 \qquad (476)$$

So the **sign of recent return picks the direction** ($\eta_i$), and **inverse
volatility picks the size** ($1/\sigma_i$).

> **Equivalence.** This is the optimization strategy (§3.18) with a diagonal
> covariance $C_{ij} = \sigma_i^2\,\delta_{ij}$ (correlations between futures ignored)
> and expected returns $E_i = \eta_i\,\sigma_i$.

### Practical refinements

- **Signal instability.** For small $|R_i|$ (relative to $\sigma_i$), $\operatorname{sign}(R_i)$
  flips on tiny moves, causing excessive turnover. Smooth it:
  $\eta_i = \tanh(R_i / \kappa)$, with $\kappa$ e.g. the cross-sectional std-dev of $R_i$.
  Alternatively just use $E_i = R_i$ (and a non-diagonal $C_{ij}$).

- **Dollar-neutrality.** Eq. (474) is *not* dollar-neutral. Demean to fix:

  $$w_i = \gamma\left(\frac{\eta_i}{\sigma_i} - \frac{1}{N}\sum_{j=1}^{N}\frac{\eta_j}{\sigma_j}\right) \qquad (477)$$

  Drawback: some winners ($\eta_i>0$) may end up sold and some losers bought.

- **Two-gamma alternative.** Split into winners $H_+$ and losers $H_-$ and scale
  each side separately:

  $$w_i = \gamma_+\frac{\eta_i}{\sigma_i},\ i\in H_+ \qquad w_i = \gamma_-\frac{\eta_i}{\sigma_i},\ i\in H_- \qquad (478\text{–}479)$$

  with $\gamma_\pm$ chosen to satisfy both (476) and dollar-neutrality $\sum_i w_i = 0$ (480).

- **Market bias.** If the market is strongly bullish/bearish most $\eta_i$ share a
  sign, leaving one side poorly diversified. Use **market-demeaned returns**
  $\tilde R_i = R_i - R_m$ ($R_m$ = market-index return), then $\eta_i = \operatorname{sign}(\tilde R_i)$
  is unbiased and dollar-neutrality follows.

- **Better trend estimators.** Instead of cumulative $R_i$, use exponential moving
  averages, the Hodrick–Prescott filter, a Kalman filter, or other time-series filters.

**Rebalance / holding:** once per lookback period $T$.

---

## 2. Cryptocurrency — Artificial Neural Network (§18.2)

**Idea.** Forecast short-term BTC moves with a feed-forward neural network whose
**inputs are technical indicators** and whose **output is a probability over return
quantiles** (cf. Nakano–Takahashi–Takahashi 2018). Crypto has no clear "fundamentals,"
so the book leans on data-mining/ML for trends.

### Feature engineering (input layer)

Let $P(t)$ be the BTC price at time $t$ (e.g. 15-minute bars; $t=1$ is most recent):

$$R(t) = \frac{P(t)}{P(t+1)} - 1 \qquad (521)$$

$$\overline{R}(t,T_1) = \frac{1}{T_1}\sum_{t'=t+1}^{t+T_1} R(t') \qquad (523)$$

$$\widetilde{R}(t,T_1) = R(t) - \overline{R}(t,T_1) \qquad (522)$$

$$[\sigma(t,T_1)]^2 = \frac{1}{T_1-1}\sum_{t'=t+1}^{t+T_1} [\widetilde{R}(t,T_1)]^2 \qquad (525)$$

$$\widehat{R}(t,T_1) = \frac{\widetilde{R}(t,T_1)}{\sigma(t,T_1)} \qquad (524)$$

i.e. $\widehat{R}$ is the **demeaned, volatility-normalized return**. From it, build
indicators:

$$\mathrm{EMA}(t,\lambda,\tau) = \frac{1-\lambda}{1-\lambda^{\tau}}\sum_{t'=t+1}^{t+\tau}\lambda^{t'-t-1}\widehat{R}(t') \qquad (526)$$

$$[\mathrm{EMSD}(t,\lambda,\tau)]^2 = \frac{1-\lambda}{\lambda-\lambda^{\tau}}\sum_{t'=t+1}^{t+\tau}\lambda^{t'-t-1}\,[\widehat{R}(t')-\mathrm{EMA}(t,\lambda,\tau)]^2 \qquad (527)$$

$$\mathrm{RSI}(t,\tau) = \frac{\nu_+(t,\tau)}{\nu_+(t,\tau)+\nu_-(t,\tau)},\qquad \nu_\pm(t,\tau)=\sum_{t'=t+1}^{t+\tau}\max(\pm\widehat{R}(t'),0) \qquad (528\text{–}529)$$

The **input layer** is then $\widehat{R}(t)$ plus $\mathrm{EMA}$, $\mathrm{EMSD}$,
$\mathrm{RSI}$ at several windows $\tau$ (e.g. 30 min, 1 h, 3 h, 6 h). To cut
parameters, set $\lambda = (\tau-1)/(\tau+1)$.

### Output layer (return quantiles)

Pick $K$ quantiles of the normalized return. With training-set quantile thresholds
$q_\alpha$, define one-hot supervision vectors $S_\alpha(t)$ that flag which bucket
$\widehat{R}(t)$ fell into (530). The network outputs probabilities $p_\alpha(t)$ with
$\sum_{\alpha=1}^{K} p_\alpha(t) = 1$ (531).

### Network & training

For layers $\ell = 1,\dots,L$:

$$X^{(\ell)}_{i} = h^{(\ell)}_i\!\big(Y^{(\ell)}\big),\qquad Y^{(\ell)}_{i} = \sum_{j} A^{(\ell)}_{ij}\,X^{(\ell-1)}_{j} + B^{(\ell)}_{i} \qquad (532\text{–}533)$$

with weights $A$ and biases $B$ learned in training. Activations:

$$h^{(\ell)}_i = \max\!\big(Y^{(\ell)}_i,\,0\big)\ \text{(ReLU, hidden)},\qquad h^{(L)}_i = \frac{Y^{(L)}_i}{\sum_j Y^{(L)}_j}\ \text{(softmax, output)} \qquad (534\text{–}535)$$

Train by minimizing **cross-entropy** via stochastic gradient descent (SGD):

$$E = -\sum_{t\in D_{\text{train}}}\sum_{\alpha=1}^{K} S_\alpha(t)\,\ln p_\alpha(t) \qquad (536)$$

### Trading rule

$$\text{Signal} = \begin{cases} \textbf{Buy}, & \max_\alpha p_\alpha(t) = p_K(t)\ \text{(top quantile)} \\ \textbf{Sell}, & \max_\alpha p_\alpha(t) = p_1(t)\ \text{(bottom quantile)} \end{cases} \qquad (537)$$

Variant: trade on the **top-2 / bottom-2** quantiles instead of just the extremes.

> ⚠️ **Overfitting risk.** Many free parameters ($\tau_a$, $\lambda_a$, $K$, layer
> sizes). Choose via out-of-sample backtesting and watch for overfit.

---

## 3. Cryptocurrency — Sentiment Analysis, naïve Bayes (Bernoulli) (§18.3)

**Idea.** Predict the direction of BTC from **Twitter sentiment** using a naïve
Bayes classifier (cf. Colianni–Rosales–Signorotti 2015; Georgoula et al. 2015).

### Data & features

1. Collect all tweets over a period containing ≥1 keyword from a learning
   vocabulary $V$ ($M = |V|$).
2. **Clean:** drop duplicate/bot tweets, remove stop-words, apply stemming
   (e.g. Porter's algorithm: "investing"/"invested" → "invest").
3. Assign each tweet $i = 1,\dots,N$ a feature vector $X_i$ with components $X_{ia}$:
   - **Bernoulli (this strategy):** $X_{ia} = 1$ if word $w_a$ appears in tweet $i$,
     else $0$.
   - *(Multinomial alternative:* $X_{ia} = n_{ia}$, the count.)*

### Classifier

Choose $K$ classes $C_\alpha$ — e.g. $K=2$ (BTC up/down → buy/sell), or $K$ return
quantiles as in §18.2. By Bayes' theorem:

$$P(C_\alpha \mid X_1,\dots,X_N) = \frac{P(X_1,\dots,X_N \mid C_\alpha)\,P(C_\alpha)}{P(X_1,\dots,X_N)} \qquad (539)$$

The **naïve** conditional-independence assumption $P(X_i \mid C_\alpha, X_{j\neq i}) = P(X_i \mid C_\alpha)$ (540) gives:

$$P(C_\alpha \mid X_1,\dots,X_N) = \gamma\,P(C_\alpha)\prod_{i=1}^{N} P(X_i \mid C_\alpha),\qquad \gamma = 1/P(X_1,\dots,X_N) \qquad (541\text{–}542)$$

Per-tweet likelihood from per-word conditionals $P(w_a \mid C_\alpha)$:

$$P(X_i \mid C_\alpha) = \prod_{a=1}^{M} Q_{ia\alpha},\qquad Q_{ia\alpha} = \begin{cases} P(w_a \mid C_\alpha), & X_{ia}=1 \\ 1 - P(w_a \mid C_\alpha), & X_{ia}=0 \end{cases} \qquad (543\text{–}545)$$

Both $P(w_a \mid C_\alpha)$ and the priors $P(C_\alpha)$ are estimated from **training
frequencies**. The predicted class (and hence trade signal) is:

$$C_{\text{pred}} = \underset{\alpha\in\{1,\dots,K\}}{\arg\max}\ P(C_\alpha)\prod_{i=1}^{N}\prod_{a=1}^{M} [P(w_a \mid C_\alpha)]^{X_{ia}}\,[1 - P(w_a \mid C_\alpha)]^{1-X_{ia}} \qquad (546)$$

### Trading rule

Map the predicted class to a position: with $K=2$, predicted "up" → **buy**,
predicted "down" → **sell**; with quantile classes, trade the extremes as in §18.2.

> The book notes related approaches for the same data: SVMs, logistic regression,
> and tree boosting (XGBoost-style).

---

*See `151-trading-strategies.md` for the full catalog of all 150+ strategies.*
