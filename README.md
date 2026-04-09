# Kalman Filter for Latent State Estimation, applications in Financial Markets

A from-scratch implementation of the Linear Gaussian Kalman Filter (KF) in Python, designed for extracting unobservable states — such as true returns, latent momentum, and hidden volatility — from noisy financial time series. This package uses the Expectation-Maximization (EM) algorithm for parameter estimation, the Rauch-Tung-Striebel (RTS) smoother for offline state refinement, and supports both causal (real-time) and non-causal (historical) inference.

---

## Table of Contents

1. [Why a Kalman Filter?](#1-why-a-kalman-filter)
2. [From HMMs to Kalman Filters](#2-from-hmms-to-kalman-filters)
3. [The Bayesian Foundation](#3-the-bayesian-foundation)
4. [Model Definition](#4-model-definition)
5. [Parameter Initialization](#5-parameter-initialization)
6. [The Forward Pass (Filtering)](#6-the-forward-pass-filtering)
7. [The Backward Pass (RTS Smoothing)](#7-the-backward-pass-rts-smoothing)
8. [Computing Cross-Covariance — The KF's "Xi"](#8-computing-cross-covariance--the-kfs-xi)
9. [The EM Algorithm (Parameter Estimation)](#9-the-em-algorithm-parameter-estimation)
10. [Causal vs Non-Causal Inference](#10-causal-vs-non-causal-inference)
11. [Use Case 1: Trend + Momentum Smoothing](#11-use-case-1-trend--momentum-smoothing)
12. [Use Case 2: AR(1) Volatility Filtering](#12-use-case-2-ar1-volatility-filtering)
13. [Usage Guide](#13-usage-guide)
14. [API Reference](#14-api-reference)
15. [Future Improvements](#15-future-improvements)

---

## 1. Why a Kalman Filter?

Financial data is inherently noisy. Observed prices, returns, and volatility proxies are all distorted by noise like microstructure noise, market instantaneous over-reaction, etc. What we see is never the "true" state of the market and we can model it as the true state plus noise:

$$z_t = \text{True State}_t + \text{Noise}_t$$

The Kalman Filter is a recursive algorithm that estimates the internal state of a dynamic system from a series of incomplete and noisy measurements. It does this by maintaining a probabilistic belief — a Gaussian distribution over what the true state might be — and updating that belief every time new data arrives.

At its core, the KF acts as a **dynamic Exponentially Weighted Moving Average (EWMA)**. But instead of a fixed, static $\alpha$ parameter, the KF dynamically calculates an optimal $\alpha$ — called the **Kalman Gain** — at each time step based on residuals (also defined as Innovation in this context, which is Observation - Prediction). When market data is noisy, the filter trusts its own prediction more. When its own prediction is uncertain, it snaps to the new observation. This is done through Bayesian inference.

**TLDR:**
- The KF separates **signal** (latent state) from **noise** (measurement error) in continuous time series
- Each latent state is tracked as a **Gaussian distribution** — a mean estimate plus an uncertainty (covariance)
- The filter dynamically adapts how much to trust its model vs. new data, via the Kalman Gain

---

## 2. From HMMs to Kalman Filters

If you've read my HMM package, understanding the Kalman Filter becomes much more intuitive. Both models share the exact same probabilistic structure — the difference is in what kind of "states" they track, which is defined differently.

### The HMM: Discrete States

In the HMM, the market can be in one of $N$ discrete regimes (e.g., Bull, Bear, High-Vol). At each time step, we compute the **probability** of being in each regime. The state space is finite and countable — we can ask "what is $P(q_t = j)$?" for each $j = 1, \ldots, N$. States here are therefore defined as the regimes each with their own mean and variance, as Gaussians.

### The KF: Continuous States

In the KF, the market's hidden state lives in a **continuous** $n$-dimensional space. Instead of "the market is in state 2 with 80% probability," the KF says "the true return is $0.0012 \pm 0.0003$." We cannot assign discrete probabilities to an infinite number of continuous points, so instead we track our estimate of the true state as a **Gaussian**:

$$\text{bel}(\mathbf{x}_t) = \mathcal{N}(\hat{\mathbf{x}}_t, \mathbf{P}_t)$$

Where $\hat{\mathbf{x}}_t$ is our best estimate (the mean) and $\mathbf{P}_t$ is our uncertainty about that estimate (the covariance matrix). The KF is therefore a continuously evolving Gaussian with changing mean and variance. 

States here are therefore the latent variables we want to track.

### Structural Parallels

The two models share a nearly identical algorithmic pipeline:

| Concept | HMM | Kalman Filter |
|---|---|---|
| Hidden states | Discrete regimes $q_t \in \{1, \ldots, N\}$ | Continuous vector $\mathbf{x}_t \in \mathbb{R}^n$ |
| State dynamics | Transition matrix $A$ | Transition matrix $\mathbf{F}$ |
| Observation model | Emission $b_j(O_t)$ per regime | Observation mapping $\mathbf{H}\mathbf{x}_t$ |
| Forward pass | Alpha (forward probabilities) | Forward filter (prediction + correction) |
| Backward pass | Beta (backward probabilities) | RTS smoother |
| State posterior | Gamma $\gamma_t(j)$ | Smoothed state $\hat{\mathbf{x}}_{t\|T}$ |
| Transition posterior | Xi $\xi_t(i,j)$ | Cross-covariance $\mathbf{P}_{t,t-1\|T}$ |
| Parameter learning | Baum-Welch (EM) | EM algorithm |
| Process noise | — (transitions are deterministic probabilities) | $\mathbf{Q}$ (structural uncertainty in dynamics) |
| Observation noise | Implicit in emission width $\sigma_j$ | $\mathbf{R}$ (explicit measurement noise) |

The relationship between components:

```
Observed Data (Z)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                    EM ALGORITHM                     │
│                                                     │
│   ┌─────────── E-Step ──────────┐                   │
│   │                             │                   │
│   │   Forward Filter            │                   │
│   │     ──► x_post, P_post      │                   │
│   │   RTS Smoother              │                   │
│   │     ──► x_smooth, P_smooth  │                   │
│   │   Cross-Covariance          │                   │
│   │     ──► P_{t,t-1|T}         │                   │
│   └─────────────────────────────┘                   │
│                  │                                  │
│                  ▼                                  │
│   ┌─────────── M-Step ──────────┐                   │
│   │                             │                   │
│   │   Moments    ──► F*, H*     │                   │
│   │   Residuals  ──► Q*, R*     │                   │
│   │   Smoothed₀  ──► x₀*, P₀*   │                   │
│   └─────────────────────────────┘                   │
│                  │                                  │
│          Repeat until convergence                   │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Fitted Model (F*, H*, Q*, R*)
        │
        ├──► Causal Filter   ──► Forward-only state estimates (real-time)
        └──► Smooth          ──► Full-data state estimates (historical)
```

---

## 3. The Bayesian Foundation

Just like the HMM, the Kalman Filter is fundamentally an application of **Bayes' Theorem**. The filter operates in a two-step cycle at every time step:

### Step 1: Prediction (The Prior)

Using our system dynamics, we push our current state estimate forward in time. This is what we *believe* will happen before seeing new data:

$$\overline{\text{bel}}(\mathbf{x}_t) = P(\mathbf{x}_t \mid \mathbf{z}_{1:t-1})$$

To compute this, we marginalize over all possible previous states — exactly as in the HMM forward algorithm, but with an integral instead of a sum (because our states are continuous):

$$\overline{\text{bel}}(\mathbf{x}_t) = \int P(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \cdot \text{bel}(\mathbf{x}_{t-1}) \; d\mathbf{x}_{t-1}$$

Because both factors under the integral are Gaussian, and a linear transformation of a Gaussian is another Gaussian, this integral has a **closed-form solution** — we never actually compute it numerically. We just track the mean and covariance algebraically.

### Step 2: Correction (The Posterior)

We observe new data $\mathbf{z}_t$ and apply Bayes' rule:

$$\text{bel}(\mathbf{x}_t) = \eta \cdot P(\mathbf{z}_t \mid \mathbf{x}_t) \cdot \overline{\text{bel}}(\mathbf{x}_t)$$

Where $\eta$ is a normalizing constant. The likelihood $P(\mathbf{z}_t \mid \mathbf{x}_t)$ asks: "how probable is this observation, given a hypothesized state?" The product of two Gaussians is another Gaussian — one that is *tighter* (more certain) than either input. The idea is that incorporating more data always reduces uncertainty.

### Cycle of Prediction and Correction

Therefore, through the continuous updating of posterior beliefs, the theory is that we are able to get a more accurate estimate of the true state of the system.

### The Kalman Gain as Dynamic $\alpha$

To build further intuition, consider the scalarized correction step:

$$x_{\text{post}} = x_{\text{prior}} + K \cdot (z - h \cdot x_{\text{prior}})$$

Rearranging:

$$x_{\text{post}} = x_{\text{prior}} + K \cdot z - K \cdot h \cdot x_{\text{prior}}$$
$$= (1 - K \cdot h) \cdot x_{\text{prior}} + K \cdot z$$

Let $\alpha = K \cdot h$. Then:

$$x_{\text{post}} = (1 - \alpha) \cdot x_{\text{prior}} + \alpha \cdot z$$

This is exactly a weighted average between our prediction and the observation, where $\alpha$ dynamically adjusts based on relative uncertainty. Similarly, the posterior covariance becomes:

$$P_{\text{post}} = (1 - \alpha) \cdot P_{\text{prior}}$$

If $\alpha$ is large (we trust the data), uncertainty shrinks rapidly. If $\alpha$ is small (we trust our model), uncertainty stays close to the prior.

---

## 4. Model Definition

The Linear Gaussian Kalman Filter is defined by two equations and their associated noise parameters. Let $n$ be the number of latent states, $m$ be the number of observations, and $T$ be the number of time steps.

### The State Transition Model

This governs how the hidden state evolves from $t-1$ to $t$:

$$\mathbf{x}_t = \mathbf{F} \mathbf{x}_{t-1} + \mathbf{w}_t$$

| Symbol | Name | Shape | Description |
|---|---|---|---|
| $\mathbf{x}_t$ | True state | $(n \times 1)$ | The unobservable hidden state vector (e.g., true return + momentum) |
| $\mathbf{F}$ | Transition matrix | $(n \times n)$ | How states interact and evolve from $t-1$ to $t$ |
| $\mathbf{w}_t$ | Process noise | $(n \times 1)$ | Structural shocks; $\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$ |
| $\mathbf{Q}$ | Process noise covariance | $(n \times n)$ | How much structural uncertainty exists in the state transition |

$\mathbf{F}$ is essentially the expectation of how $\mathbf{x}_{t+1}$ relates to $\mathbf{x}_t$. The diagonal entries capture each state's autoregressive behaviour (persistence), while off-diagonals capture cross-state interactions (e.g., momentum feeding into returns). $\mathbf{Q}$ represents uncertainty in the "true" state dynamics — its diagonals are the variance of each state's structural shock, and off-diagonals represent correlated shocks across states.

### The Observation Model

This governs how hidden states map to observable measurements:

$$\mathbf{z}_t = \mathbf{H} \mathbf{x}_t + \mathbf{v}_t$$

| Symbol | Name | Shape | Description |
|---|---|---|---|
| $\mathbf{z}_t$ | Observation | $(m \times 1)$ | The noisy measurement we actually see (e.g., market returns) |
| $\mathbf{H}$ | Observation matrix | $(m \times n)$ | Maps the $n$-dimensional state to the $m$-dimensional observation |
| $\mathbf{v}_t$ | Measurement noise | $(m \times 1)$ | Temporary noise; $\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$ |
| $\mathbf{R}$ | Measurement noise covariance | $(m \times m)$ | How severely observations are corrupted by noise |

$\mathbf{H}$ translates the $n$ latent variables into the $m$ observations. If $n = m$ and $\mathbf{H} = \mathbf{I}$, each state is directly observed (with noise). If $m < n$, we observe fewer things than we're estimating — the filter must infer the unobserved states from partial information.

### The Key Distinction: Q vs R

One of the most important conceptual points is the *separation* of noise into two distinct sources:

- **$\mathbf{Q}$ (Process Noise):** Real structural shocks in the underlying system. Even if we could observe the state perfectly, it would still evolve with randomness. This represents fundamental shifts — regime changes, policy announcements, liquidity shocks.

- **$\mathbf{R}$ (Measurement Noise):** Noise in our *observation* of the state, not in the state itself. This represents stale prices, data errors, etc.

This is actually one part that initially confused me as I was working through this project. Try to think of Q and R as realized "+-" Variances around a measurement, something like measuring a length using a ruler and recording it down as 10cm +- 0.1cm. 

The EM algorithm's primary job is to learn where to draw this boundary. Too much allocated to $\mathbf{Q}$ means the filter thinks the underlying state is wildly volatile and will overfit to noise. Too much to $\mathbf{R}$ means the filter thinks its observations are garbage and will be sluggish. The optimal split maximizes the likelihood of the observed data.

---

## 5. Parameter Initialization

Before the EM algorithm can iteratively refine parameters, we need sensible starting points. Poor initialization can lead to convergence at a bad local optimum.

### Initial State Estimate ($\hat{\mathbf{x}}_0$)

We can initialize through a reverse mapping of the first observation into latent space using the pseudo-inverse of $\mathbf{H}$:

$$\hat{\mathbf{x}}_0 = \mathbf{H}^+ \mathbf{z}_0$$

Where $\mathbf{H}^+$ is the Moore-Penrose pseudo-inverse. This handles the case where $\mathbf{H}$ may not be square or full-rank. Essentially, we are backwards transforming the observation to get a reasonable initial state.

### Initial Covariance ($\mathbf{P}_0$)

$$\mathbf{P}_0 = \mathbf{H}^+ \mathbf{R} (\mathbf{H}^+)^\top + \mathbf{Q}$$

This maps the observation noise back into state space and adds the process noise, giving us a reasonable initial uncertainty that accounts for both sources of error.

### Transition Matrix ($\mathbf{F}$)

If no prior knowledge is available, we initialize $\mathbf{F} = \mathbf{I}_n$ (identity matrix). This assumes a **Martingale process** — the best guess for $\mathbf{x}_{t+1}$ is simply $\mathbf{x}_t$, with no interaction between states. This is a maximally neutral prior that the EM algorithm can refine.

### Observation Matrix ($\mathbf{H}$)

If $n = m$: $\mathbf{H} = \mathbf{I}_m$ (direct observation of each state). If $n > m$: $\mathbf{H} = [\mathbf{I}_m \mid \mathbf{0}]$ — we observe only the first $m$ states. If $n < m$: truncated identity.

### Noise Covariances ($\mathbf{Q}$, $\mathbf{R}$)

We estimate the total system variance from the data using `np.cov(Z.T)`, then split it roughly 50/50:

$$\mathbf{R}_{\text{init}} = \text{Cov}(Z) \times 0.5$$
$$\mathbf{Q}_{\text{init}} = \bar{\sigma}^2_{\text{obs}} \times 0.5 \times \mathbf{I}_n$$

Where $\bar{\sigma}^2_{\text{obs}}$ is the mean diagonal of the observation covariance. We want $\mathbf{Q}$ and $\mathbf{R}$ to be roughly somewhere sensible so EM doesn't start from nonsense.

---

## 6. The Forward Pass (Filtering)

The forward pass is the KF's analogue to the HMM's Forward algorithm. It runs chronologically from $t = 0$ to $T-1$, computing the **filtered** state distribution — our best estimate of $\mathbf{x}_t$ using only observations up to and including time $t$.

Think of this system of 2 lines working in parallel, the Expectation track and Variance (or error) track. They do not typically interact, and are computed independently of each other, but all in all contribute towards the system as a whole as an evolving Gaussian at each time step

### Prediction Step

Project the state forward using the transition model:

$$\hat{\mathbf{x}}_{t|t-1} = \mathbf{F} \hat{\mathbf{x}}_{t-1|t-1}$$

$$\mathbf{P}_{t|t-1} = \mathbf{F} \mathbf{P}_{t-1|t-1} \mathbf{F}^\intercal + \mathbf{Q}$$

The first equation propagates the mean estimate forward. The second equation propagates the covariance — note how $\mathbf{P}$ is spatially rotated by $\mathbf{F}$ (the $\mathbf{F} \mathbf{P} \mathbf{F}^\intercal$ term) and then **expanded** by the addition of $\mathbf{Q}$. After prediction, we are always **less certain** than before, because time introduces structural noise.

In the scalarized case, this becomes:

$$P_{\text{prior}} = f^2 \cdot P_{\text{post}} + q$$

Which is simply the variance transformation under a linear mapping ($f^2 P$) plus the additive process noise ($q$). Think of this as the 'Realized Variance' of the system at t.

### Correction Step

Now we observe $\mathbf{z}_t$ and update our belief:

**Innovation (prediction error):**

$$\boldsymbol{\nu}_t = \mathbf{z}_t - \mathbf{H} \hat{\mathbf{x}}_{t|t-1}$$

This is the difference between what we *observed* and what we *expected* to observe. It measures how surprised the model is by the new data. In a well-specified model, the innovations should behave like white noise. If not, the model is not a good fit for the data.

**Innovation covariance (total uncertainty in observation space):**

$$\mathbf{S}_t = \mathbf{H} \mathbf{P}_{t|t-1} \mathbf{H}^\intercal + \mathbf{R}$$

This is the total variance of our prediction error. In scalar terms:

$$S = h^2 \cdot P_{\text{prior}} + R = \text{Var}(z_t) = \text{Var}(h \cdot x_{\text{prior}} + v_t) = h^2 P_{\text{prior}} + R$$

**Kalman Gain:**

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^\intercal \mathbf{S}_t^{-1}$$

The Kalman Gain is the ratio of **prediction uncertainty in observation space** to **total uncertainty**. In scalar terms:

$$K = \frac{h \cdot P_{\text{prior}}}{h^2 \cdot P_{\text{prior}} + R} = \frac{\text{Cov}(z_t, x_{\text{prior}})}{\text{Var}(z_t)}$$

Essentially, the numerator is the extent of covariance between the prior state and the observation, while the denominator is the total 'realized variance' of the observation. 

This is literally a **regression coefficient** — the optimal linear weight for updating our state estimate given the innovation. Note the intuitive limiting behaviours:

- If $\mathbf{R} \to 0$ (perfect measurements): $\mathbf{K} \to \mathbf{H}^{-1}$, and we snap entirely to the observation
- If $\mathbf{P}_{t|t-1} \to 0$ (perfect prediction): $\mathbf{K} \to 0$, and we ignore the observation entirely
- If $\mathbf{R}$ is large relative to $\mathbf{P}$: $\mathbf{K}$ is small — trust the prediction
- If $\mathbf{P}$ is large relative to $\mathbf{R}$: $\mathbf{K}$ is large — trust the observation

**State update:**

$$\hat{\mathbf{x}}_{t|t} = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t \boldsymbol{\nu}_t$$

**Covariance update:**

$$\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{H}) \mathbf{P}_{t|t-1}$$

Note that $\mathbf{P}_{t|t}$ is always **smaller** than $\mathbf{P}_{t|t-1}$ (in the positive-definite sense) — incorporating data always reduces uncertainty.

### Log-Likelihood

After the filter pass, we compute the **innovation log-likelihood**. This is the KF equivalent of the HMM's `sum(log(c))`:

$$\ell = -\frac{1}{2} \sum_{t=0}^{T-1} \left[ m \ln(2\pi) + \ln|\mathbf{S}_t| + \boldsymbol{\nu}_t^\intercal \mathbf{S}_t^{-1} \boldsymbol{\nu}_t \right]$$

Each term in the sum is the log-probability of observing $\mathbf{z}_t$ given everything we've seen so far — it measures how surprised the model is at each step.

### Numerical Stability

A critical implementation detail: we never compute $\mathbf{S}_t^{-1}$ explicitly. Instead, we use the identity:

$$\mathbf{K}_t = \mathbf{P}_{t|t-1} \mathbf{H}^\intercal \mathbf{S}_t^{-1} \iff \mathbf{S}_t^\intercal \mathbf{K}_t^\intercal = \mathbf{H} \mathbf{P}_{t|t-1}$$

And solve via `scipy.linalg.solve(S, H @ P_prior, assume_a='pos').T`. This avoids explicit matrix inversion, which is numerically unstable for near-singular covariance matrices. Similarly, the log-determinant is computed via `np.linalg.slogdet(S)` to avoid overflow/underflow.

---

## 7. The Backward Pass (RTS Smoothing)

The Rauch-Tung-Striebel (RTS) smoother is the KF's analogue to the HMM's Backward algorithm. Just as the HMM's backward beta refines the forward alpha into the full posterior gamma, the RTS smoother refines the forward-filtered estimates $\hat{\mathbf{x}}_{t|t}$ into the **smoothed** estimates $\hat{\mathbf{x}}_{t|T}$ — conditioned on **all** data, past and future.

### Definition

$$\hat{\mathbf{x}}_{t|T} = \mathbb{E}[\mathbf{x}_t \mid \mathbf{z}_{1:T}]$$

This is the best possible estimate of the state at time $t$ using the entire observation sequence. It is always at least as good as (and usually better than) the filtered estimate $\hat{\mathbf{x}}_{t|t}$.

### Recursion

**Base case** ($t = T-1$):

$$\hat{\mathbf{x}}_{T-1|T} = \hat{\mathbf{x}}_{T-1|T-1} \quad \quad \mathbf{P}_{T-1|T} = \mathbf{P}_{T-1|T-1}$$

At the last time step, the smoothed estimate equals the filtered estimate — there is no future data to incorporate. This mirrors HMM's $\beta_{T-1}(j) = 1$.

**Backward recursion** ($t = T-2$ down to $0$):

$$\mathbf{G}_t = \mathbf{P}_{t|t} \mathbf{F}^\intercal \mathbf{P}_{t+1|t}^{-1} \quad \text{(Smoother Gain)}$$

$$\hat{\mathbf{x}}_{t|T} = \hat{\mathbf{x}}_{t|t} + \mathbf{G}_t \left( \hat{\mathbf{x}}_{t+1|T} - \hat{\mathbf{x}}_{t+1|t} \right)$$

$$\mathbf{P}_{t|T} = \mathbf{P}_{t|t} + \mathbf{G}_t \left( \mathbf{P}_{t+1|T} - \mathbf{P}_{t+1|t} \right) \mathbf{G}_t^\intercal$$

### Interpretation of the Smoother Gain

$\mathbf{G}_t$ plays the exact same structural role as the Kalman Gain, but in reverse. It determines how much the future-corrected information at $t+1$ should "pull" the estimate at $t$.

In the scalarized case:

$$G_t = \frac{\text{Cov}(\hat{x}_{\text{post},t}, \hat{x}_{\text{prior},t+1})}{\text{Var}(\hat{x}_{\text{prior},t+1})} = \frac{f \cdot P_{\text{post},t}}{f^2 \cdot P_{\text{post},t} + q}$$

The numerator $\text{Cov}(x_{\text{post},t}, x_{\text{prior},t+1}) = f \cdot P_{\text{post},t}$ measures how correlated our current filtered state is with the next prediction. The denominator $\text{Var}(x_{\text{prior},t+1}) = f^2 P_{\text{post},t} + q$ is the total prediction uncertainty.

If $\text{Cov}(t, t+1)$ is high relative to $\text{Var}(x_{\text{prior},t+1})$, we smooth aggressively — we have high confidence that future corrections should propagate backward. If $\text{Var}(x_{\text{prior},t+1})$ is large (our prediction was very uncertain), we smooth less — the future state tells us little about the past.

### Numerical Stability

Just like the forward pass, we avoid explicit inversion of $\mathbf{P}_{t+1|t}$. In the implementation, we use pseudo-inverse `np.linalg.pinv(P_prior[t+1])` because when $\mathbf{Q}$ has zero entries (e.g., a state that is deterministic, like a constant long-term mean), $\mathbf{P}_{t+1|t}$ can become singular.

---

## 8. Computing Cross-Covariance

In the HMM, $\xi_t(i,j)$ captures the joint posterior probability of being in state $i$ at time $t$ and state $j$ at time $t+1$ — it tells us about **transitions**. In the KF, the analogous quantity is the **smoothed cross-covariance**:

$$\mathbf{P}_{t, t-1|T} = \text{Cov}(\mathbf{x}_t, \mathbf{x}_{t-1} \mid \mathbf{Z}_{1:T})$$

### Why Do We Need This?

The EM M-step for $\mathbf{F}$ requires the expected cross-moment between adjacent states. Think of it this way:

- In HMM, to update $A_{ij}$, you need "how often did transitions $i \to j$ happen?" It is the average rate of transitions from $i \to j$ given all opportunities for the jump from $i \to j$.
- In KF, to update $\mathbf{F}$, you need "what is the expected linear relationship between $\mathbf{x}_{t-1}$ and $\mathbf{x}_t$?" It is the average rate of change of each latent state from t-1 to t given the relationship each latent state has with each other.

### Derivation

The smoothed variables are deterministic estimates — we know them already. We cannot directly compute covariance from point estimates. Instead, we reason through the true latent variable $\mathbf{x}_{\text{true}}$ and the smoothing equation:

$$\mathbf{x}_{\text{true},t} = \hat{\mathbf{x}}_{\text{smooth},t} + \mathbf{e}_t$$

Where $\mathbf{e}_t$ is the estimation error. Substituting into the smoother recursion:

$$\mathbf{x}_{\text{true},t} = \mathbf{G}_{t-1} \cdot \mathbf{x}_{\text{true},t} + (\text{deterministic terms}) + \mathbf{e}_{t-1}$$

Taking the covariance (deterministic terms vanish, error is independent of the true state):

$$\text{Cov}(\mathbf{x}_t, \mathbf{x}_{t-1} \mid \mathbf{Z}) = \mathbf{G}_{t-1} \cdot \text{Cov}(\mathbf{x}_t, \mathbf{x}_t \mid \mathbf{Z}) = \mathbf{G}_{t-1} \cdot \mathbf{P}_{t|T}$$

### Formula

$$\mathbf{P}_{t, t-1|T} = \mathbf{P}_{t|T} \; \mathbf{G}_{t-1}^\intercal$$

This is deceptively simple — the smoother gain $\mathbf{G}_{t-1}$ already encodes the predictive connection between timesteps, and $\mathbf{P}_{t|T}$ is the smoothed covariance at $t$. In code, this is fully vectorized:

```python
cross_cov = P_smooth[1:] @ c[:-1].transpose(0, 2, 1)
```

Where `@` broadcasts across the time dimension, replacing a for loop entirely.

---

## 9. The EM Algorithm (Parameter Estimation)

The EM algorithm for the KF is structurally identical to the HMM's Baum-Welch. We face the same chicken-and-egg problem: to find optimal parameters, we need the hidden states; to find the hidden states, we need the parameters.

### E-Step (Expectation)

Run the forward filter, backward smoother, and cross-covariance computation. This gives us all the smoothed quantities we need: $\hat{\mathbf{x}}_{t|T}$, $\mathbf{P}_{t|T}$, and $\mathbf{P}_{t,t-1|T}$.

Then compute the **expected sufficient statistics** — these are the KF analogue of HMM's gamma-weighted sums:

**Expected second moment** (scalarized: $E[x_t^2] = E[x_t]^2 + \text{Var}(x_t)$):

$$\mathbb{E}[\mathbf{x}_t \mathbf{x}_t^\intercal \mid \mathbf{Z}] = \hat{\mathbf{x}}_{t|T} \hat{\mathbf{x}}_{t|T}^\intercal + \mathbf{P}_{t|T}$$

**Expected cross-moment** (scalarized: $E[x_t x_{t-1}] = \text{Cov}(x_t, x_{t-1}) + E[x_t] E[x_{t-1}]$):

$$\mathbb{E}[\mathbf{x}_t \mathbf{x}_{t-1}^\intercal \mid \mathbf{Z}] = \mathbf{P}_{t,t-1|T} + \hat{\mathbf{x}}_{t|T} \hat{\mathbf{x}}_{t-1|T}^\intercal$$

These are then aggregated into matrix sums for the M-step:

| Symbol | Formula | Role |
|---|---|---|
| $\mathbf{A}$ | $\sum_{t=1}^{T-1} \mathbb{E}[\mathbf{x}_t \mathbf{x}_t^\intercal]$ | Sum of "current" second moments |
| $\mathbf{B}$ | $\sum_{t=1}^{T-1} \mathbb{E}[\mathbf{x}_t \mathbf{x}_{t-1}^\intercal]$ | Sum of cross-moments |
| $\mathbf{C}$ | $\sum_{t=0}^{T-2} \mathbb{E}[\mathbf{x}_t \mathbf{x}_t^\intercal]$ | Sum of "previous" second moments |
| $\mathbf{D}$ | $\sum_{t=0}^{T-1} \mathbf{z}_t \hat{\mathbf{x}}_{t|T}^\intercal$ | Observation–state cross products |
| $\mathbf{E}$ | $\sum_{t=0}^{T-1} \mathbb{E}[\mathbf{x}_t \mathbf{x}_t^\intercal]$ | Sum of all second moments |

### M-Step (Maximization)

**Update $\mathbf{F}$ — State transition:**

$$\mathbf{F}^* = \mathbf{B} \mathbf{C}^{-1}$$

This is a regression. In scalar terms:

$$F^* = \frac{\sum E[x_t \cdot x_{t-1}]}{\sum E[x_{t-1}^2]}$$

**Update $\mathbf{Q}$ — Process noise covariance:**

$$\mathbf{Q}^* = \frac{1}{T-1}\left(\mathbf{A} - \mathbf{F}^* \mathbf{B}^\intercal - \mathbf{B} {\mathbf{F}^*}^\intercal + \mathbf{F}^* \mathbf{C} {\mathbf{F}^*}^\intercal\right)$$

This is the expected squared residual of the state transition, expanded from $\mathbb{E}[(\mathbf{x}_t - \mathbf{F}^* \mathbf{x}_{t-1})^2]$. In scalar terms, the expansion simplifies to $E[x_t^2] - F \cdot E[x_t x_{t-1}]$, since $F^2 E[x_{t-1}^2] = F \cdot E[x_t x_{t-1}]$ by the OLS definition of $F$.

**Update $\mathbf{H}$ — Observation matrix:**

$$\mathbf{H}^* = \mathbf{D} \mathbf{E}^{-1}$$

Same regression logic.

**Update $\mathbf{R}$ — Measurement noise covariance:**

$$\mathbf{R}^* = \frac{1}{T} \sum_{t=0}^{T-1} \left[ (\mathbf{z}_t - \mathbf{H}^* \hat{\mathbf{x}}_{t|T})(\mathbf{z}_t - \mathbf{H}^* \hat{\mathbf{x}}_{t|T})^\intercal + \mathbf{H}^* \mathbf{P}_{t|T} {\mathbf{H}^*}^\intercal \right]$$

The first term is the squared observation residual. The second term is a critical **uncertainty adjustment** — it corrects for the fact that $\hat{\mathbf{x}}_{t|T}$ is an *estimate*, not the true state. Without it, $\mathbf{R}$ would be systematically underestimated.

**Update initial conditions:**

$$\hat{\mathbf{x}}_0^* = \hat{\mathbf{x}}_{0|T} \quad \quad \mathbf{P}_0^* = \mathbf{P}_{0|T}$$

Simply read off the first smoothed values — these are the best estimates of the initial state given all data.

### SPD Enforcement

After computing $\mathbf{Q}^*$ and $\mathbf{R}^*$, we force symmetry and positive definiteness:

```python
Q_new = (Q_new + Q_new.T) / 2 + 1e-6 * np.eye(n)
R_new = (R_new + R_new.T) / 2 + 1e-6 * np.eye(m)
```

Unlike HMM parameters (which are probabilities naturally bounded between 0 and 1), covariance matrices can drift out of the symmetric positive-definite (SPD) cone due to floating-point accumulation. SPD is a hard requirement — a covariance matrix that isn't SPD is not a valid covariance matrix, and Cholesky/inverse operations will fail.

### Convergence

The log-likelihood is monitored at each iteration. EM is guaranteed to produce monotonically non-decreasing log-likelihood — if it ever decreases, there is a bug in the M-step. The algorithm terminates when the change falls below a tolerance threshold or the maximum iteration count is reached.

### Locking F and H

In practice, we often have **domain knowledge** about the system dynamics that should not be overridden by data-driven optimization. The `em_update_FH` flag controls whether EM re-estimates $\mathbf{F}$ and $\mathbf{H}$. When set to `False`, EM only optimizes $\mathbf{Q}$ and $\mathbf{R}$ — finding the optimal noise decomposition for fixed dynamics. This is critical in both demo use cases, where allowing $\mathbf{F}$ and $\mathbf{H}$ to be re-estimated would cause the interpretable state relationships to collapse into a degenerate solution.

---

## 10. Causal vs Non-Causal Inference

Just like the HMM's `mode='infer'`, the KF supports both offline and causal prediction:

### Non-Causal (Default)

Runs the forward filter **and** the backward RTS smoother. The smoothed estimates $\hat{\mathbf{x}}_{t|T}$ use the **entire** dataset — including future observations — to determine the state at time $t$. This is the most accurate mode for historical analysis and research.

### Causal (`mode='infer'`)

Runs **only** the forward filter. The filtered estimates $\hat{\mathbf{x}}_{t|t}$ use only data available at or before time $t$. No future information is used at any point. This is the mode you must use to avoid lookahead bias.

| Mode | Returns | Uses future data? | Use for |
|---|---|---|---|
| Default (offline) | `x_smooth`, `P_smooth` | Yes | Historical analysis, EM training |
| Causal (`mode='infer'`) | `x_post`, `P_post` | No | Live trading, backtesting |

The causal estimates tend to have a wider confidence band than smoothed estimates. Smoothed estimates require future data to confirm the states. Therefore, without that, the causal estimates tend to be less decisive than smoothed estimates.

---

## 11. Use Case 1: Trend + Momentum Smoothing

**Objective:** Decompose noisy S&P 500 daily log returns into an underlying "true return" and a latent momentum component.

### The Model

We assume that the observed market return is a noisy measurement of the true underlying return, and that the true return has a persistent momentum component:

$$R_t = \text{True Returns}_t + \text{Measurement Noise}_t$$

The true returns and momentum co-evolve:

$$\text{True Returns}_t = \text{True Returns}_{t-1} + 1 \cdot \text{Momentum}_{t-1} + \text{Process Noise}$$

$$\text{Momentum}_t = 0 \cdot \text{True Returns}_{t-1} + 1 \cdot \text{Momentum}_{t-1} + \text{Process Noise}$$

### Matrix Form

**State transition:**

$$\begin{bmatrix} \text{Returns}_t \\ \text{Momentum}_t \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} \text{Returns}_{t-1} \\ \text{Momentum}_{t-1} \end{bmatrix} + \mathbf{w}_t$$

**Observation:**

$$\text{Market Returns}_t  = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} \text{Returns}_t \\ \text{Momentum}_t \end{bmatrix} + v_t$$

So we set:

$$\mathbf{F} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

### Why Lock F and H?

The $\mathbf{F}$ matrix encodes our structural belief: momentum persists with a random walk, and it feeds into returns via the off-diagonal. If we let EM re-estimate $\mathbf{F}$, it has no incentive to maintain this interpretable structure — it would find a degenerate solution that minimizes log-likelihood but loses the momentum decomposition.

By setting `em_update_FH = False`, EM learns **only** the optimal noise split: how much of the observed variation is real structural change ($\mathbf{Q}$) versus temporary noise ($\mathbf{R}$).

### Results

The filter converges in ~23 iterations. The fitted $\mathbf{Q}$ converges toward near-zero, meaning the filter learns that the true underlying state changes very slowly, and most of the observed volatility is measurement noise ($\mathbf{R}$). The smoothed state provides a zero-lag noise-free signal.

See `demo_1_filtering.ipynb` for the full implementation and visualizations.

---

## 12. Use Case 2: AR(1) Volatility Filtering

**Objective:** Track unobservable true market volatility using a noisy Yang-Zhang volatility proxy, modelling the true log-variance as a mean-reverting AR(1) process.

### The Observation: Yang-Zhang Daily Variance

Instead of using simple Close-to-Close returns for volatility, we use a Yang-Zhang adapted proxy that combines overnight gaps with intraday Rogers-Satchell extremes:

$$\sigma_{YZ}^2 = [\ln(O_t / C_{t-1})]^2 + \sigma_{RS}^2$$

Where the Rogers-Satchell intraday component is:

$$\sigma_{RS}^2 = \ln(H_t/C_t)\ln(H_t/O_t) + \ln(L_t/C_t)\ln(L_t/O_t)$$

We then take $z_t = \ln(\sigma_{YZ,t}^2)$ as our observation — working in log-space ensures positivity and makes the AR(1) model appropriate.

### The Latent State: AR(1) Log-Variance

Assume the true hidden log-variance follows a mean-reverting process:

$$x_t = \phi \cdot x_{t-1} + (1 - \phi) \cdot \mu + w_t$$

Where:
- $\mu$ is the long-term mean of log-variance (a constant latent state: $\mu_t = \mu_{t-1}$)
- $\phi$ is the mean-reversion coefficient ($|\phi| < 1$ for stationarity)
- $w_t \sim \mathcal{N}(0, q)$ is the process noise

### Matrix Form

Defining the state vector $\boldsymbol{\theta}_t = [x_t, \mu_t]^\intercal$:

$$\mathbf{F} = \begin{bmatrix} \phi & 1-\phi \\ 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad \mathbf{Q} = \begin{bmatrix} q & 0 \\ 0 & 0 \end{bmatrix}$$

### Estimating $\phi$ via Modified Yule-Walker

Standard Yule-Walker equations (which rely on observed variance at lag 0) yield upward-biased estimates because the lag-0 variance $\gamma_{z,0}$ is inflated by measurement noise. We bypass this by using only lags 1 and 2:

For an AR(1) process with measurement noise:

$$\gamma_{z,k} = \text{Cov}(z_t, z_{t-k}) = \text{Cov}(x_t + v_t, x_{t-k} + v_{t-k})$$

For $k \geq 1$, the cross-terms involving independent measurement noise vanish:

$$\gamma_{z,k} = \text{Cov}(x_t, x_{t-k}) = \phi^k \text{Var}(x)$$

Therefore:

$$\phi = \frac{\gamma_{z,2}}{\gamma_{z,1}}$$

This elegantly avoids the noise-contaminated lag-0 term entirely.

### Results

The filter converges in ~77 iterations. The fitted $\mathbf{R} \approx 0.56$ (substantial measurement noise, as expected from a daily volatility proxy), while $\mathbf{Q}_{11} \approx 0.04$ (relatively small process noise — true volatility evolves smoothly).

See `demo_2_vol.ipynb` for the full implementation and visualizations.

---

## 13. Usage Guide

### Installation

No installation required. Just place `KF.py` in your working directory.

**Dependencies:**
```
numpy
scipy
plotly (for visualization)
pandas (for data handling)
yfinance (for data download — Demo 1)
requests (for Tiingo API — Demo 2)
```

### Quick Start (Demo 1: Smoothing)

```python
from KF import KalmanFilter, plot_filtered_price
import yfinance as yf
import numpy as np

# 1. Download data
spy = yf.download('^GSPC', start='2000-12-31', end='2020-12-31')

# 2. Compute log returns and reshape to (T, m)
returns = np.log(spy['Close'] / spy['Close'].shift(1)).dropna()
Z_train = returns.values.reshape(-1, 1)

# 3. Define system dynamics
F = np.array([[1, 1], [0, 1]])    # Returns + Momentum
H = np.array([[1, 0]])             # Observe only returns

# 4. Fit
kf = KalmanFilter(n_states=2, m_obs=1, F=F, H=H,
                  max_iter=200, tol=1e-2, em_update_FH=False)
kf.fit(Z_train)

# 5. Predict (smoothed — offline)
x_smooth, P_smooth = kf.predict(Z_train)

# 6. Predict (causal — for backtesting)
x_filt, P_filt = kf.predict(Z_train, mode='infer')

# 7. Visualize
plot_filtered_price(Z_train, price=spy['Close'], kf=kf,
                    index=returns.index, title="S&P 500 — KF Smoothing")
```

### Visualization

Three plotting functions are provided, mirroring the HMM package's visual style:

| Function | Purpose |
|---|---|
| `plot_filtered_price()` | Multi-panel state estimation with price overlay, confidence bands, and innovations |
| `plot_em_diagnostics()` | EM convergence: log-likelihood curve + Q/R norm evolution |
| `plot_innovation_diagnostics()` | Model validation: normalized innovations (white noise check) + QQ-plot |

`plot_filtered_price()` produces an interactive Plotly chart with up to 4 panels:

| Panel | Content | Appears when |
|---|---|---|
| Top | Price with filtered/smoothed overlay + 95% CI bands | `price` is provided |
| Middle | Observations (dots) vs. filtered/smoothed state (lines) | Always |
| Thin strips | Additional latent states (e.g., momentum) | When `n_states > 1` |
| Bottom | Innovation residuals with ±2σ bounds | Always |

The 95% confidence bands are computed from the diagonal of the covariance matrix: $\hat{x}_t \pm 1.96\sqrt{P_{t,ii}}$.

---

## 14. API Reference

### `KalmanFilter(n_states, m_obs, x0=None, P0=None, F=None, H=None, Q=None, R=None, max_iter=500, tol=1e-6, em_update_FH=False)`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_states` | int | — | Number of latent state dimensions ($n$) |
| `m_obs` | int | — | Number of observation dimensions ($m$) |
| `x0` | array | `None` | Initial state estimate $(n,)$; auto-computed if `None` |
| `P0` | array | `None` | Initial covariance $(n, n)$; auto-computed if `None` |
| `F` | array | `None` | Transition matrix $(n, n)$; defaults to identity |
| `H` | array | `None` | Observation matrix $(m, n)$; defaults to identity/truncated |
| `Q` | array | `None` | Process noise covariance $(n, n)$; estimated from data if `None` |
| `R` | array | `None` | Measurement noise covariance $(m, m)$; estimated from data if `None` |
| `max_iter` | int | 500 | Maximum EM iterations |
| `tol` | float | 1e-6 | Convergence threshold for log-likelihood change |
| `em_update_FH` | bool | `False` | If `True`, EM also re-estimates $\mathbf{F}$ and $\mathbf{H}$ |

### `.fit(Z)`

Fit the KF parameters via EM on observation matrix `Z`.

| Parameter | Type | Description |
|---|---|---|
| `Z` | array-like | Observation matrix, shape $(T, m)$ |

**Fitted attributes:**
- `kf.F` — Transition matrix $(n, n)$
- `kf.H` — Observation matrix $(m, n)$
- `kf.Q` — Process noise covariance $(n, n)$
- `kf.R` — Measurement noise covariance $(m, m)$
- `kf.ll_history` — Log-likelihood at each EM iteration
- `kf.Q_history` — $\mathbf{Q}$ matrix at each iteration
- `kf.R_history` — $\mathbf{R}$ matrix at each iteration

### `.predict(Z, mode=None, initial_state='train_end')`

Run inference on observation matrix `Z`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `Z` | array-like | — | Observation matrix $(T, m)$ |
| `mode` | str or None | `None` | `'infer'` for causal (forward-only); `None` for smoothed |
| `initial_state` | str | `'train_end'` | `'train_end'`: continue from training; `'dynamic'`: re-initialize from first obs |

**Returns:**

| Mode | Returns | Description |
|---|---|---|
| Default | `x_smooth`, `P_smooth` | Smoothed state estimates and covariances |
| `'infer'` | `x_post`, `P_post` | Filtered (causal) state estimates and covariances |

### `plot_filtered_price(Z, price=None, kf=None, overlays=None, mode=None, index=None, dim=0, title=None)`

| Parameter | Type | Description |
|---|---|---|
| `Z` | array | Observation matrix $(T, m)$ |
| `price` | array-like | Close price for top panel overlay |
| `kf` | KalmanFilter | Fitted KF object |
| `overlays` | dict | External price overlays (e.g., `{'SMA 50': sma_50}`) |
| `mode` | str | `'infer'` disables smoothed traces |
| `index` | array-like | X-axis labels (dates) |
| `dim` | int | Which state/observation dimension to plot |
| `title` | str | Chart title |

### `plot_em_diagnostics(ll_history, Q_history=None, R_history=None)`

EM convergence: left panel shows monotonically increasing log-likelihood; right panel shows Frobenius norms of $\mathbf{Q}$ and $\mathbf{R}$ converging. This means model is working as expected.

### `plot_innovation_diagnostics(innov, S, dim=0)`

Model validation: top panel shows normalized innovations $\mathbf{S}_t^{-1/2} \boldsymbol{\nu}_t$ (should resemble white noise with ±2σ bounds); bottom panel shows QQ-plot against standard normal (should be linear if Gaussian assumption holds).

---

## References

- Rauch, H. E., Tung, F., & Striebel, C. T. (1965). *Maximum Likelihood Estimates of Linear Dynamic Systems*. AIAA Journal.
- Shumway, R. H. & Stoffer, D. S. (2000). *Time Series Analysis and Its Applications*. Springer. — EM algorithm for state-space models.
- Yang, D. & Zhang, Q. (2000). *Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices*. Journal of Business. — Yang-Zhang volatility estimator used in Demo 2.
