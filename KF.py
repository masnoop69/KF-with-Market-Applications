import numpy as np
from scipy import linalg

class KalmanFilter:
    def __init__(self, n_states, m_obs, \
                 x0 = None, P0 = None, \
                 F = None, H = None, Q = None, R = None, \
                 max_iter = 500,tol = 1e-6, em_update_FH = False):
        """
        Kalman Filter (KF) is a recursive algorithm that estimates the internal state of a 
        dynamic system from a series of incomplete and noisy measurements. It is essentially the
        continuous version of the Hidden Markov Model (HMM).

        To visualize this, imagine the discrete states of the HMM, where each distinct state 
        has its own fixed mean and variance. Then, for the KF, imagine a continuous spectrum 
        of infinite potential states. Because we cannot assign discrete probabilities to an 
        infinite number of continuous points, we instead track our estimate of the true state. 
        We assume our uncertainty around this estimate is Gaussian, so we represent our belief 
        of the true state similarly. Hence, the KF is a continuously evolving Gaussian with 
        changing mean and variance.
        
        The KF operates in two steps: prediction and update. It does so through Bayesian inference.

        Prediction: 
        The KF projects the current state estimate and its uncertainty forward in time using assumed 
        system dynamics (state transition matrix, state noise covariance, error covariance, etc.)
        
        Update: 
        The KF relies on Bayes' theorem to fuse the predicted state with the new, noisy 
        observation, computing an optimal blended estimate. When the data is noisy due to high
        system error, the model will rely more on its own prediction, and when the data is cleaner, 
        the model will rely more on the new observation. This is done through 
        something called the Kalman gain, of which it can be conceptualized through a dynamic 
        Exponentially Weighted Moving Average (EWMA), where Alpha changes from t to t+1.
        
        The basic KF is a strictly Linear Gaussian State Space Model, assuming both the state 
        transitions and observation mappings are linear operations corrupted by Gaussian noise.

        In Financial markets, this linearity does not hold, hence fitting with Extended Kalman Filter
        might be more optimal.

        """
        self.n_states = n_states # n number of derived (latent) variables from m observations
        self.m_obs = m_obs # m number of noisy observations that we receive
        self.tol = tol
        self.max_iter = max_iter

        if x0 is not None:
            x0 = np.asarray(x0).flatten()
            if x0.shape != (n_states,):
                raise ValueError(f"x0 must have length {n_states}, but got {x0.shape}")
        self.x0 = x0

        if P0 is not None:
            P0 = np.asarray(P0)
            if P0.shape != (n_states, n_states):
                raise ValueError(f"P0 must have shape ({n_states}, {n_states}), but got {P0.shape}")
        self.P0 = P0

        if F is not None:
            F = np.asarray(F)
            if F.shape != (n_states, n_states):
                raise ValueError(f"F must have shape ({n_states}, {n_states}), but got {F.shape}")
        self.F = F

        if H is not None:
            H = np.asarray(H)
            if H.shape != (m_obs, n_states):
                raise ValueError(f"H must have shape ({m_obs}, {n_states}), but got {H.shape}")
        self.H = H

        if Q is not None:
            Q = np.asarray(Q)
            if Q.shape != (n_states, n_states):
                raise ValueError(f"Q must have shape ({n_states}, {n_states}), but got {Q.shape}")
        self.Q = Q

        if R is not None:
            R = np.asarray(R)
            if R.shape != (m_obs, m_obs):
                raise ValueError(f"R must have shape ({m_obs}, {m_obs}), but got {R.shape}")
        self.R = R

        self.em_update_FH = em_update_FH

# ==================== Helper Functions =====================
    def _initialize(self, Z):
        """
        We do not know the exact dynamics of the system, hence we need to make the most neutral or
        educated guess (domain knowledge or otherwise) of the initial parameters. Only P is 
        estimated from the data, and the rest are initialized as neutral as possible.

        Hence, we model the data as such in terms of scalars:
            system dynamics:
                x_prior_t = F * x_post_t-1 + w_t 
                where w_t ~ N(0, Q) (Q is the covariance between variables across variables at t) and is not measureable.
            observation:
                z_t = H * x_prior_t + v_t
                where v_t ~ N(0, R) (R is total observation noise covariance across observations at t) and is not measureable.

        Therefore:
            1. Prediction:
                x_prior_t = F @ x_post_t-1
                P_prior_t = F @ P_post_t-1 @ F.T + Q

            2. Correction:
                K_t = P_prior_t @ H.T @ inv(H @ P_prior_t @ H.T + R)
                x_post_t = x_prior_t + K_t @ (z_t - H @ x_prior_t)
                P_post_t = (I - K_t @ H) @ P_prior_t

            This will be explained further later.

        We are looking at the following parameters:
        x_0: initial state estimate (n x 1) -
            We initialize this through the reverse mapping of the first observation to latent variables using the 
            pseudo-inverse of H, where H may not be square or full-rank.

        P_0: initial state error covariance (state confidence) (n x n) - 
            We initialize this through utilizing the R to "update" the state error.

        F: system dynamics matrix / state transition matrix (n x n) - 
            F is essentially how the variables interact with each other at t to form the next prediction or prior at t+1.
            Essentially it is just the expectatation of how x_t+1 is related to x_t.
            
            We initialize F as an identity matrix, assuming no prior knowledge of system dynamics. 
            We assume x to be Martingale process, therefore expectation of x in t+1 given x in t is x in t.
            Therefore, state transition should be Identity matrix, where there is no interaction between the other
            variables from t to t+1.

        Q: process noise covariance w_t (n x n) - 
            Q represents uncertainty in 'true' state from t to t+1, and how the extent of "co-varyness" of uncertainty
            between variables. Diagonals would be the variance of each variable, and off-diagonals would be the 
            covariance between variables. Hence, if there is a shock in variable i, it will affect variable j with 
            covariance of Q_ij.

            We initialize this through covariance of each measurement in dZ through all observations from [0:T], and multiply
            it by 0.5. We assume that the process noise is half of the estimated total system noise.

        H: observation matrix (m x n) - 
            Essentially, this translates the m number of observations into the n number of latent variables. 
            
            As we do not know the dynamics of how measurements relate to the latent variables, we initialize H as an 
            identity matrix, assuming that each measurement is a direct observation of a latent variable. Where there 
            are more observations than latent variables, we initialize the extra rows as zeros, and vice versa.
            
        R: observation noise covariance v_t (m x m) - 
            Measurements potentially have overlap in information or uncertainties, when translating into latent variables.

            We initialize this by taking the other half of the estimated total system noise.
        """
        # ===========initialize required parameters=============
        n = self.n_states
        m = self.m_obs

        #============== initialize system dynamics================
        if self.F is None:
            self.F = np.eye(n) # F: system dynamics matrix / state transition matrix (n x n)

        #============== initialize noise covariances ================
        if self.R is None or self.Q is None:
            # np.cov returns a scalar for 1-d input — force to (m, m)
            obs_cov = np.atleast_2d(np.cov(Z.T))
            
            if self.R is None:
                self.R = obs_cov * 0.5 # R: observation noise covariance v_t (m x m)
            
            if self.Q is None:
                # Q must be (n x n) — scale from observation variance
                obs_var = np.mean(np.diag(obs_cov))
                self.Q = obs_var * 0.5 * np.eye(n) # Q: process noise covariance w_t (n x n)

        # H must be (m x n) — maps n-dimensional state to m-dimensional observation
        if self.H is None:
            if n == m:
                self.H = np.eye(m)
            elif n > m:
                self.H = np.hstack([np.eye(m), np.zeros((m, n - m))]) # (m x n): observe first m states
            else:
                self.H = np.eye(m, n) # (m x n): truncated identity

        #============== internal method access================
        self.T = len(Z)
        self.Z = Z

    def _forward_pass(self, x0=None, P0=None):
        """
        Forward pass of the Kalman Filter, which computes the conditional distribution of the state
        at time t given observations up to time t.

        state transition model:
            x_prior_t = F * x_post_t-1 + w_t 
        observation:
            z_t = H * x_prior_t + v_t

        Prediction:
            x_prior_t = F @ x_post_t-1
            P_prior_t = F @ P_post_t-1 @ F.T + Q

        Correction:
            K_t = P_prior_t @ H.T @ inv(H @ P_prior_t @ H.T + R)
            x_post_t = x_prior_t + K_t @ (z_t - H @ x_prior_t)
            P_post_t = (I - K_t @ H) @ P_prior_t

            where K_t is the Kalman Gain (KG), which acts as a dynamic weighing ratio between (Uncertainty in Prediction) / (Total Uncertainty),
            where prediction uncertainty would be the prediction covariance. If Prediction Uncertainty is high (P_prior_t high), we trust the
            observations more. However, if the observation noise is high (R high), we trust the prediction more.

            P_prior_t @ H.T can be interpreted as the cov(zt, x_prior_t). Scalarized:
                cov(zt, x_prior_t) = cov(h * x_prior_t + v_t, x_prior_t), where v_t is independent of x_prior_t
                                    = h * var(x_prior_t), and as x_prior_t ~ N(0, P_prior_t),
                                    = h * P_prior_t 
            H @ P_prior_t @ H.T + R can be interpreted as the var(z_t) + var(v_t). Scalarized:
                var(z_t) = var(h * x_prior_t + v_t) + var(v_t)
                            = h^2 * var(x_prior_t) + var(v_t)
                            = h^2 * P_prior_t + R

            Thus, we can rewrite the correction step as such (scalarized):
            x_post_t = x_prior_t + K_t * (z_t - h * x_prior_t)
                     = x_prior_t + K_t * z_t - K_t * h * x_prior_t
                     = (1 - K_t * h) * x_prior_t + K_t * z_t        , let alpha = K_t * h
                     = (1 - alpha) * x_prior_t + alpha * z_t

            P_post_t = (I - K_t * h) * P_prior_t
                     = (1 - alpha) * P_prior_t
            
            Therefore, alpha can be intepreted as dynamic degree of trust in the observations vs our predictive model. 

        In terms of a Bayesian Interpretation, recall that:
            P(H | E) = [P(E | H) * P(H)] / P(E)

        Therefore:
        1. Prediction:
            bel_hat(x[t]) = P(x[t] | z[1:t-1])
                          based on the law of total probability, we integrate over all possible values of x[t-1]:
            bel_hat(x[t]) = integral(P(x[t], x[t-1] | z[1:t-1])) dx[t-1]
                          applying the chain rule of probability:
                          = integral(P(x[t] | x[t-1], z[1:t-1]) * P(x[t-1] | z[1:t-1])) dx[t-1]
                          based on the Markov assumption, x[t] is independent of z[1:t-1] given x[t-1]:
                          = integral(P(x[t] | x[t-1]) * P(x[t-1] | z[1:t-1])) dx[t-1]
                          = integral(P(x[t] | x[t-1]) * bel(x[t-1])) dx[t-1]

            where bel(x[t-1]) is the prior belief of x[t-1] given z[1:t-1].
            
        2. Correction:
            bel(x[t]) = P(x[t] | z[1:t])
                      separating the current measurement z[t] from the history z[1:t-1]:
                      = P(x[t] | z[t], z[1:t-1])
                      applying Bayes' Rule:
                      = [ P(z[t] | x[t], z[1:t-1]) * P(x[t] | z[1:t-1]) ] / P(z[t] | z[1:t-1])
                          letting the denominator be normalization constant n,
            bel(x[t]) = n * P(z[t] | x[t], z[1:t-1]) * P(x[t] | z[1:t-1])
                      based on the sensor Markov assumption, z[t] is independent of z[1:t-1] given x[t]:
                      = n * P(z[t] | x[t]) * P(x[t] | z[1:t-1])
                      = n * P(z[t] | x[t]) * bel_hat(x[t])

            where bel_hat(x[t]) is the predicted belief of x[t] given z[1:t-1].
                     
        
        """
        # ============== Call upon internalized parameters ==================
        # Vectors and Params
        Z = self.Z
        T = self.T
        n = self.n_states
        m = self.m_obs

        # System Dynamics
        F = self.F
        Q = self.Q
        H = self.H
        R = self.R
        
        # =========== initialize starting variables================
        x_0 = x0 if x0 is not None else self.x0
        P_0 = P0 if P0 is not None else self.P0

        if x_0 is None:
            x_0 = np.linalg.pinv(H) @ Z[0] # initial state estimate (n x 1)
        if P_0 is None:
            P_0 = np.linalg.pinv(H) @ R @ np.linalg.pinv(H).T + Q # initial state error covariance (n x n)
        I = np.eye(n)

        # initialize storage arrays
        x_prior = np.zeros((T, n))
        P_prior = np.zeros((T, n, n))
        x_post = np.zeros((T, n))
        P_post = np.zeros((T, n, n))
        K = np.zeros((T, n, m))
        innov = np.zeros((T, m))
        S = np.zeros((T, m, m))

        # ============== execute KF Algorithm ================
        # Prediction step
        # x_prior_t = F @ x_post_t-1
        # P_prior_t = F @ P_post_t-1 @ F.T + Q

        # Correction Step
        # K_t = P_prior_t @ H.T @ inv(H @ P_prior_t @ H.T + R)
        # x_post_t = x_prior_t + K_t @ (z_t - H @ x_prior_t)
        # P_post_t = (I - K_t @ H) @ P_prior_t

        # first run
        # Predict
        x_prior[0] = x_0
        P_prior[0] = P_0

        #correct
        innov[0] = Z[0] - H @ x_prior[0]
        S[0] = H @ P_prior[0] @ H.T + R
        K[0] = linalg.solve(S[0], H @ P_prior[0], assume_a = 'pos').T

        x_post[0] = x_prior[0] + K[0] @ innov[0]
        P_post[0] = (I - K[0] @ H) @ P_prior[0]

        for t in range(1, T):
            # Prediction
            x_prior[t] = F @ x_post[t-1]
            P_prior[t] = F @ P_post[t-1] @ F.T + Q

            # Correction
            innov[t] = Z[t] - H @ x_prior[t]
            S[t] = H @ P_prior[t] @ H.T + R

            K[t] = linalg.solve(S[t], H @ P_prior[t], assume_a = 'pos').T
            x_post[t] = x_prior[t] + K[t] @ innov[t]
            P_post[t] = (I - K[t] @ H) @ P_prior[t]

        self.x_prior = x_prior
        self.P_prior = P_prior
        self.x_post = x_post
        self.P_post = P_post
        self.K = K
        self.innov = innov
        self.S = S
        # =================== Log Likelihood ===================
        Sv = np.linalg.solve(S, innov[:, :, np.newaxis])  # for stability
        vSvT = (innov[:, np.newaxis, :] @ Sv).squeeze()
        _, log_det_S = np.linalg.slogdet(S)
        step_llh = -0.5 * (m * np.log(2 * np.pi) + log_det_S + vSvT)

        self.llh = np.sum(step_llh)

    def _backward_pass(self):
        """
        We use the Rauch-Tung-Striebel Backwards Smoother to smooth the state estimates. The formula is as follows:
        c[t] = P_post[t] @ F.T @ inv(P_prior[t+1])
        x_smooth[t] = x_post[t] + c[t] @ (x_smooth[t+1] - x_prior[t+1])
        P_smooth[t] = P_post[t] + c[t] @ (P_smooth[t+1] - P_prior[t+1]) @ c[t].T

        c[t] can be interpreted as a 'Kalman' gain in this sense, and we name it the Smoothing Gain. Essentially, it is
        the ratio between the covariance of prior and posterior variables at t to t+1 and the total prediction variance 
        at t+1. 
            Scalarized:
                Cov(x_post[t], x_prior[t+1]) = Cov(x_post[t], f * x_post[t])
                                           = f * Cov(x_post[t], x_post[t]), as x_post[t] ~ N(x_post[t], P_post[t])
                                           = f * P_post[t]

                Var(x_prior[t+1]) = Var(f * x_post[t] + w[t])
                                  = f^2 * Var(x_post[t]) + Var(w[t])
                                  = f^2 * P_post[t] + q
                    
        Hence, c[t] is a measure of how much we trust our t+1 prediction. If Covar(t,t+1) is higher (relative to Var(x_prior[t+1])), 
        we smooth more as we are confident that future corrections will not deviate much from our t+1 prediction.
        If Var(x_prior[t+1]) is higher, that means that our t+1 prediction is very wrong, and we do not smooth as much.
        """
        # ============== Call upon internalized parameters ==================
        # Vectors and Params
        T = self.T
        n = self.n_states

        # Filtered Estimates
        x_prior = self.x_prior
        P_prior = self.P_prior
        x_post = self.x_post
        P_post = self.P_post

        # System Dynamics
        F = self.F
        
        # =========== initialize arrays================
        x_smooth = np.zeros((T, n))
        P_smooth = np.zeros((T, n, n))
        c = np.zeros((T, n, n))

        # ============ initialize variables =============
        x_smooth[-1] = x_post[-1]
        P_smooth[-1] = P_post[-1]

        for t in range(T-2, -1, -1):
            # Use pseudo-inverse for numerical stability when states are deterministic (e.g. Q has zeros)
            c[t] = (np.linalg.pinv(P_prior[t+1], rcond=1e-8) @ F @ P_post[t]).T
            x_smooth[t] = x_post[t] + c[t] @ (x_smooth[t+1] - x_prior[t+1])
            P_smooth[t] = P_post[t] + c[t] @ (P_smooth[t+1] - P_prior[t+1]) @ c[t].T

        self.x_smooth = x_smooth
        self.P_smooth = P_smooth
        self.c = c

    def _compute_cross_cov(self):
        """
        Essentially, we are just calculating Cov(x[t], x[t-1] | Z[1:T]), which is the smoothed cross-covariance from the 
        variables in t and t+1. As the smoothed variables are deterministic (ie we know what they are already), we cannot
        directly calculate the cross-covariance, and instead use the true latent x for this.

        Looking at the smoothing equation:
            x_smooth[t] = x_post[t] + c[t] * (x_smooth[t+1] - x_prior[t+1])
            x_true[t] = x_smooth[t] + e[t], where e is error term.

        Hence:
            x_true[t] = x_post[t] + c[t] * (x_true[t+1] - x_prior[t+1]) + e[t]
            x_true[t] = c[t] * x_true[t+1] + (x_post[t] - c[t] * x_prior[t+1]) + e[t]
            
            Cov(x[t], x[t-1] | Z[1:T]) = Cov(x_true[t], c[t-1] * x_true[t] + (x_post[t-1] - c[t-1] * x_prior[t]) + e[t-1] | Z[1:T])
                as x_post[t-1] and x_prior[t] are deterministic and constants, Cov() are 0 for those values,
                                       = Cov(x_true[t], c[t-1 * x_true[t]] + e[t-1]), and as e is independent of x_true[t],
                                       = Cov(x_true[t], c[t-1 * x_true[t]])
                                       = c[t-1] * Cov(x_true[t], x_true[t])
                                       = c[t-1] * P_smooth[t]
        """
        # ============== Call upon internalized parameters ==================
        P_smooth = self.P_smooth
        c = self.c

        # =========== initialize arrays================
        cross_cov = P_smooth[1:] @ c[:-1].transpose(0, 2, 1)
        self.cross_cov = cross_cov
        
    def _em_step(self):
        """
        Baum-welch but for KF instead of HMM. We need to update our system dynamics via Expectation - Maximization.

        We first calculate the Expected Values on what we need by using the smoothed variables, which
        we assume are closer to the true latent variables than the filtered variables.

        Expectation Step:
        We base most of our calculations through these 2 equations (scalarized):
            1. 2nd Moment of x_t
            E[x_t^2] = x_smooth[t] * x_smooth[t].T + P_smooth[t]
                     = E[x_t]^2 + Var(x_t)

            2. cross moment of x_t and x_{t-1}
            E[x_t * x_{t-1} | Z] = cross_cov[t] + x_smooth[t] * x_smooth[t-1]
                                 = Cov(x_t, x_{t-1}) + E[x_t] * E[x_{t-1}]
            
        Maximization Step:
        Therefore, we find the new system dynamics that maximize the fit of the data:
            To compute F_new
            E[x_t * x_{t-1} | Z] / E[x_{t-1}^2] = E[F_new * x_{t-1} * x_{t-1}.T | Z] / E[x_{t-1}^2]
                                                         = F_new * E[x_{t-1}^2] / E[x_{t-1}^2]
                                                         = F_new
            
            To compute Q, it is the expected squared residual of the state transition (sample variance):
            s^2 = sum_{i=1}^{n} (x_i - mu)^2 / (n - 1)
            Q_new = sum_{t=1}^{T-1} E[(x_t - F_new * x_{t-1})^2 | Z] / (T - 1)

            Expanding the full quadratic:
            E[(x_t - F * x_{t-1})^2 | Z] = E[x_t^2] - 2F * E[x_t * x_{t-1}] + F^2 * E[x_{t-1}^2]
                as F^2 * E[x_{t-1}^2] = F * F * E[x_{t-1}^2] = F * E[F * x_{t-1}^2] = F * E[x_t * x_{t-1}]
            E[(x_t - F * x_{t-1})^2 | Z] = E[x_t^2] - F * E[x_t * x_{t-1}]
            
            To compute H, it is the back-mapping of the observation equation:
            Z = H * x + e, where e ~ N(0, R)
            E[Z * x] = E[H * x^2 + e * x]
                     = H * E[x^2]        (as e is independent of x)
            H_new = E[Z * x] / E[x^2] = D @ E^{-1}
            
            To compute R, it is the expected squared observation residual given smoothed states:
            R_new = (1/T) * sum_{t=0}^{T-1} E[(z_t - H_new * x_t)^2 | Z]

            Since x_t is uncertain (we only have the smoothed estimate), this expands to:
                  = (1/T) * sum [ (z_t - H_new * x_smooth_t)^2  +  H_new^2 * P_smooth_t]
                                ^^^^ squared residual ^^^^     ^^^^ uncertainty adjustment ^^^^

            The second term corrects for the fact that x_smooth is an estimate, not the true state.
            Without it, R would be systematically underestimated.
        """
        # =============== E step ===============
        # execute algos
        self._forward_pass()
        self._backward_pass()
        self._compute_cross_cov()

        # prerequisite variables
        x_smooth = self.x_smooth # (T x n)
        cross_cov = self.cross_cov # (T x n x n)
        P_smooth = self.P_smooth # (T x n x n)

        Z = self.Z # (T x m)
        n = self.n_states
        m = self.m_obs

        # Expectation Calculation
        # 2nd moment of smoothing
        E_X2 = x_smooth[:,:,np.newaxis] @ x_smooth[:,np.newaxis,:] + P_smooth
        # cross moment of smoothing
        E_Xt_Xtm1 = cross_cov + x_smooth[1:,:,np.newaxis] @ x_smooth[:-1,np.newaxis,:]

        # Summation of required expectations
        A = E_X2[1:].sum(axis = 0)
        B = E_Xt_Xtm1.sum(axis = 0)
        C = E_X2[:-1].sum(axis = 0)
        D = (Z[:, :, np.newaxis] * x_smooth[:, np.newaxis, :]).sum(axis=0)
        E = E_X2.sum(axis = 0)

        # =============== M step ===============
        if self.em_update_FH is True:
            # update F and H
            F_new = linalg.solve(C.T, B.T).T
            H_new = linalg.solve(E.T, D.T).T
            # update matrices
            self.F = F_new
            self.H = H_new
        else:
            F_new = self.F
            H_new = self.H
        # update Q using the fully expanded quadratic formula (required if F is held constant)
        Q_new = (A - F_new @ B.T - B @ F_new.T + F_new @ C @ F_new.T) / (self.T - 1)
        Q_new = (Q_new + Q_new.T) / 2 + 1e-6 * np.eye(n) # ensure SPD

        # update R, average observation error given new estimates
        resid = Z - x_smooth @ H_new.T
        R_new = (resid[:, :, np.newaxis] @ resid[:, np.newaxis, :] + H_new @ P_smooth @ H_new.T).sum(axis = 0) / self.T
        R_new = (R_new + R_new.T) / 2 + 1e-6 * np.eye(m)#ensure SPD
        
        # =================== Update Matrices ===================
        self.Q = Q_new
        self.R = R_new

        # ============ Update starting Parameters for Next Loop ============
        self.x0 = self.x_smooth[0]
        self.P0 = self.P_smooth[0]

# ==================== Public API =====================
    def fit(self, Z):
        """
        Training for the KF parameters. 
        """
        Z = np.asarray(Z)
        self.Z = Z

        # ============= Parameter Tracking ==============

        self.ll_history = []
        self.Q_history = []
        self.R_history = []

        # ============ Initialization ==============
        
        self._initialize(Z)
        print("="*50)
        print("Initial System Dynamics:")
        print("F:", np.round(self.F, 4))
        print("Q:", np.round(self.Q, 4))
        print("H:", np.round(self.H, 4))
        print("R:", np.round(self.R, 4))
        print("")
        print("Initial State Specific Parameters:")
        print("Measurements:", self.m_obs)
        print("Latent Variables:", self.n_states)
        print("="*50)
        prev_llh = -np.inf

        # ================ Iterative Estimation Stage ==================
        print("="*50)
        print("Commencing iterative estimation...")
        for i in range(self.max_iter):
            self._em_step()
            self.ll_history.append(self.llh)
            self.Q_history.append(self.Q)
            self.R_history.append(self.R)
            print(f"Iteration {(i + 1)}: Log Likelihood: {self.llh:.4f}")
            if np.abs(self.llh - prev_llh) < self.tol:
                print(f"Converged after {(i + 1)} iterations.")
                break
            prev_llh = self.llh

        # ================== Final Dynamics ==================
        print("="*50)
        print("Fitted State Specific Parameters:")
        print("F:", np.round(self.F, 4))
        print("Q:", np.round(self.Q, 4))
        print("H:", np.round(self.H, 4))
        print("R:", np.round(self.R, 4))
        print("="*50)
        
        # Save end-of-training states for out-of-sample prediction
        self.x_post_train = self.x_post[-1].copy()
        self.P_post_train = self.P_post[-1].copy()

    def predict(self, Z, mode = None, initial_state='train_end'):
        """
        Predicting using the KF model.
        """
        Z = np.asarray(Z)
        self.Z = Z
        self.T = len(Z)
        
        x0_pred = None
        P0_pred = None
        if initial_state == 'train_end' and hasattr(self, 'x_post_train'):
            x0_pred = self.x_post_train
            P0_pred = self.P_post_train
        elif initial_state == 'dynamic':
            x0_pred = np.linalg.pinv(self.H) @ Z[0]
            P0_pred = np.linalg.pinv(self.H) @ self.R @ np.linalg.pinv(self.H).T + self.Q
        else:
            x0_pred = self.x0
            P0_pred = self.P0
        
        # ============= causal ================
        if mode == 'infer':
            self._forward_pass(x0=x0_pred, P0=P0_pred)
            return self.x_post, self.P_post

        # ============= non-causal =============
        self._forward_pass(x0=x0_pred, P0=P0_pred)
        self._backward_pass()
        return self.x_smooth, self.P_smooth

# =============== Visualization Tools =====================

def plot_filtered_price(Z, price=None, kf=None, overlays=None, mode=None, index=None, dim=0, title=None):
    """
    Kalman Filter state estimation plot (mirrors HMM's plot_regimes layout).

    Subplot layout:
        Top     — Market returns/price with smoothed state overlay + 95% confidence bands (tall)
        Middle  — State / Observation comparison: obs dots, filtered + smoothed line (medium)
        Bottom  — Innovation sequence with ±2sigma bounds (thin)

    Parameters
    ----------
    Z         : (T, m) array       Observation matrix (e.g. log returns)
    price     : array-like         Close price series (same length as Z)
    kf        : KalmanFilter       Fitted Kalman Filter object
    overlays  : dict               Optional dictionary of external overlays (e.g. SMA/EMA)
    mode      : str                If 'infer', safely disables rendering of smoothed values
    index     : array-like         x-axis labels (dates, integers, etc.), optional
    dim       : int                Which state/obs dimension to plot (default 0)
    title     : str                Chart title, optional
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import pandas as pd

    Z = np.asarray(Z)
    
    # ── Map object properties safely ──
    x_filt = getattr(kf, 'x_post', None)
    if x_filt is None:
        raise ValueError("Kalman Filter object must have 'x_post' populated to plot.")
    x_filt = np.asarray(x_filt)
    
    P_filt = getattr(kf, 'P_post', None)
    innov = getattr(kf, 'innov', None)
    S = getattr(kf, 'S', None)
    H = getattr(kf, 'H', None)

    # Disable smooth lines if purely in inference mode
    x_smooth = getattr(kf, 'x_smooth', None) if mode != 'infer' else None
    P_smooth = getattr(kf, 'P_smooth', None) if mode != 'infer' else None

    T = len(x_filt)

    # x-axis
    if index is not None:
        idx = pd.Index(index)[:T]
    else:
        idx = np.arange(T)

    # ── Determine subplot layout ──
    has_innov = innov is not None and S is not None
    has_smooth = x_smooth is not None
    has_price = price is not None

    # Identify other latent states to plot
    n_states = x_filt.shape[1] if x_filt.ndim > 1 else 1
    other_dims = [j for j in range(n_states) if j != dim]

    # Row config: price (tall) + state/obs (medium) + other states (thin) + innovation (thin)
    n_rows = 1  # always have state/obs row
    row_labels = ['state']  # track what each row is
    if has_price:
        n_rows += 1
        row_labels.insert(0, 'price')   # price goes on top
        
    for j in other_dims:
        n_rows += 1
        row_labels.append(f'other_{j}')

    if has_innov:
        n_rows += 1
        row_labels.append('innov')      # innovation at bottom

    # row heights
    height_map = {'price': 0.45, 'state': 0.25, 'innov': 0.12}
    if not has_price:
        height_map['state'] = 0.50
        
    for j in other_dims:
        height_map[f'other_{j}'] = 0.12

    row_heights = [height_map.get(r, 0.12) for r in row_labels]
    # normalize
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig_height = 400 + 150 * n_rows

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
    )

    # ── Row number lookup ──
    row_of = {label: i + 1 for i, label in enumerate(row_labels)}

    # ── Color palette ──
    price_color = 'rgba(255, 255, 255, 0.85)'
    obs_color = 'rgba(150, 150, 150, 0.45)'
    filt_color = 'rgba(0, 180, 255, 1)'
    smooth_color = 'rgba(255, 100, 200, 1)'
    band_color = 'rgba(0, 180, 255, 0.12)'
    band_smooth_color = 'rgba(255, 100, 200, 0.15)'
    innov_color = 'rgba(255, 200, 50, 0.7)'
    bound_color = 'rgba(255, 80, 80, 0.6)'
    overlay_colors = ['rgba(150, 255, 150, 0.8)', 'rgba(255, 150, 255, 0.8)', 'rgba(255, 255, 150, 0.8)', 'rgba(150, 255, 255, 0.8)', 'rgba(255, 150, 150, 0.8)']

    # ── Precompute observation and state values ──
    obs_dim = min(dim, Z.shape[1] - 1) if Z.ndim > 1 else 0
    obs_vals = Z[:, obs_dim] if Z.ndim > 1 else Z.flatten()

    if H is not None:
        H = np.asarray(H)
        filt_vals = (x_filt @ H.T)[:, dim]
        if has_smooth:
            x_smooth = np.asarray(x_smooth)
            smooth_vals = (x_smooth @ H.T)[:, dim]
        if P_filt is not None:
            P_filt = np.asarray(P_filt)
            P_filt_mapped = np.einsum('ij, tjk, lk -> til', H, P_filt, H)
            filt_std = np.sqrt(P_filt_mapped[:, dim, dim]) if P_filt_mapped.ndim == 3 else np.sqrt(P_filt_mapped.flatten())
        if P_smooth is not None:
            P_smooth = np.asarray(P_smooth)
            P_smooth_mapped = np.einsum('ij, tjk, lk -> til', H, P_smooth, H)
            sm_std = np.sqrt(P_smooth_mapped[:, dim, dim]) if P_smooth_mapped.ndim == 3 else np.sqrt(P_smooth_mapped.flatten())
    else:
        filt_vals = x_filt[:, dim] if x_filt.ndim > 1 else x_filt.flatten()
        if has_smooth:
            x_smooth = np.asarray(x_smooth)
            smooth_vals = x_smooth[:, dim] if x_smooth.ndim > 1 else x_smooth.flatten()
        if P_filt is not None:
            P_filt = np.asarray(P_filt)
            filt_std = np.sqrt(P_filt[:, dim, dim]) if P_filt.ndim == 3 else np.sqrt(P_filt.flatten())
        if P_smooth is not None:
            P_smooth_arr = np.asarray(P_smooth)
            sm_std = np.sqrt(P_smooth_arr[:, dim, dim]) if P_smooth_arr.ndim == 3 else np.sqrt(P_smooth_arr.flatten())

    # ══════════════════════════════════════════════
    # TOP: Market Returns / Price with smooth overlay + confidence bands
    # ══════════════════════════════════════════════
    if has_price:
        pr = row_of['price']
        price_arr = np.asarray(price).flatten()
        # align length
        if len(price_arr) == T + 1:
            p0 = price_arr[0]
            price_arr = price_arr[1:]
        elif len(price_arr) == T:
            p0 = price_arr[0] / np.exp(obs_vals[0])
        else:
            p0 = price_arr[0]
            
        price_arr = price_arr[:T]

        fig.add_trace(go.Scatter(
            x=idx, y=price_arr,
            mode='lines',
            line=dict(color=price_color, width=1),
            name='Close Price',
        ), row=pr, col=1)

        # Filtered price
        filt_price = p0 * np.exp(np.cumsum(filt_vals))
        fig.add_trace(go.Scatter(
            x=idx, y=filt_price,
            mode='lines',
            line=dict(color=filt_color, width=1.1 if has_smooth else 1.3),
            name='Filtered Price',
        ), row=pr, col=1)

        # Bollinger-band style: smoothed price + local uncertainty bands
        if has_smooth and P_smooth is not None:
            # Smoothed price = P0 * exp(cumsum(smoothed log returns))
            smooth_price = p0 * np.exp(np.cumsum(smooth_vals))

            # Local bands: smooth_price * exp(±1.96σ)  (like Bollinger bands)
            upper_price = smooth_price * np.exp(1.96 * sm_std)
            lower_price = smooth_price * np.exp(-1.96 * sm_std)

            fig.add_trace(go.Scatter(
                x=idx, y=smooth_price,
                mode='lines',
                line=dict(color=smooth_color, width=1.3),
                name='Smoothed Price',
            ), row=pr, col=1)

            fig.add_trace(go.Scatter(
                x=idx, y=upper_price,
                mode='lines', line=dict(width=0),
                showlegend=False,
            ), row=pr, col=1)
            fig.add_trace(go.Scatter(
                x=idx, y=lower_price,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=band_smooth_color,
                name='95% CI (Smoothed)',
            ), row=pr, col=1)
        elif not has_smooth and P_filt is not None:
            # If no smooth, at least show the Filtered uncertainty
            upper_filt = filt_price * np.exp(1.96 * filt_std)
            lower_filt = filt_price * np.exp(-1.96 * filt_std)
            
            fig.add_trace(go.Scatter(
                x=idx, y=upper_filt,
                mode='lines', line=dict(width=0),
                showlegend=False,
            ), row=pr, col=1)
            fig.add_trace(go.Scatter(
                x=idx, y=lower_filt,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=band_color,
                name='95% CI (Filtered)',
            ), row=pr, col=1)

        if overlays is not None:
            if isinstance(overlays, dict):
                for i, (name, arr) in enumerate(overlays.items()):
                    ov_arr = np.asarray(arr).flatten()
                    if len(ov_arr) >= T + 1:
                        ov_arr = ov_arr[1:T+1]
                    else:
                        ov_arr = ov_arr[:T]
                    
                    c = overlay_colors[i % len(overlay_colors)]
                    fig.add_trace(go.Scatter(
                        x=idx, y=ov_arr,
                        mode='lines',
                        line=dict(width=1.2, color=c, dash='longdash'),
                        name=name,
                    ), row=pr, col=1)

        fig.update_yaxes(title_text='Price', row=pr, col=1)

    # ══════════════════════════════════════════════
    # MIDDLE: State / Observation comparison
    # ══════════════════════════════════════════════
    sr = row_of['state']

    # Observations
    fig.add_trace(go.Scatter(
        x=idx, y=obs_vals,
        mode='markers',
        marker=dict(size=2, color=obs_color),
        name='Observations',
    ), row=sr, col=1)

    # Filtered state
    fig.add_trace(go.Scatter(
        x=idx, y=filt_vals,
        mode='lines',
        line=dict(color=filt_color, width=1.2),
        name='Filtered State',
    ), row=sr, col=1)

    # Filtered confidence band (±1.96σ)
    if P_filt is not None:
        upper = filt_vals + 1.96 * filt_std
        lower = filt_vals - 1.96 * filt_std

        fig.add_trace(go.Scatter(
            x=idx, y=upper,
            mode='lines', line=dict(width=0),
            showlegend=False,
        ), row=sr, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=lower,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=band_color,
            name='95% CI (Filtered)',
        ), row=sr, col=1)

    # Smoothed state + band on state panel
    if has_smooth:
        fig.add_trace(go.Scatter(
            x=idx, y=smooth_vals,
            mode='lines',
            line=dict(color=smooth_color, width=1.2, dash='dot'),
            name='Smoothed State',
        ), row=sr, col=1)

        if P_smooth is not None:
            fig.add_trace(go.Scatter(
                x=idx, y=smooth_vals + 1.96 * sm_std,
                mode='lines', line=dict(width=0),
                showlegend=False,
            ), row=sr, col=1)
            fig.add_trace(go.Scatter(
                x=idx, y=smooth_vals - 1.96 * sm_std,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=band_smooth_color,
                name='95% CI (Smooth)' if has_price else '95% CI (Smoothed)',
            ), row=sr, col=1)

    fig.update_yaxes(title_text='State / Obs', row=sr, col=1)

    # ══════════════════════════════════════════════
    # THIN STRIPS: Other latent states
    # ══════════════════════════════════════════════
    for j in other_dims:
        r = row_of[f'other_{j}']
        
        fig.add_trace(go.Scatter(
            x=idx, y=x_filt[:, j],
            mode='lines',
            line=dict(color=filt_color, width=1.2),
            name=f'Filtered State {j}',
        ), row=r, col=1)

        if P_filt is not None:
            std_f = np.sqrt(P_filt[:, j, j])
            fig.add_trace(go.Scatter(
                x=idx, y=x_filt[:, j] + 1.96 * std_f,
                mode='lines', line=dict(width=0),
                showlegend=False,
            ), row=r, col=1)
            fig.add_trace(go.Scatter(
                x=idx, y=x_filt[:, j] - 1.96 * std_f,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=band_color,
                showlegend=False,
            ), row=r, col=1)

        if has_smooth:
            fig.add_trace(go.Scatter(
                x=idx, y=x_smooth[:, j],
                mode='lines',
                line=dict(color=smooth_color, width=1.2, dash='dot'),
                name=f'Smoothed State {j}',
            ), row=r, col=1)

            if P_smooth is not None:
                std_s = np.sqrt(P_smooth[:, j, j])
                fig.add_trace(go.Scatter(
                    x=idx, y=x_smooth[:, j] + 1.96 * std_s,
                    mode='lines', line=dict(width=0),
                    showlegend=False,
                ), row=r, col=1)
                fig.add_trace(go.Scatter(
                    x=idx, y=x_smooth[:, j] - 1.96 * std_s,
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor=band_smooth_color,
                    showlegend=False,
                ), row=r, col=1)

        fig.update_yaxes(title_text=f'State {j}', row=r, col=1)

    # ══════════════════════════════════════════════
    # BOTTOM: Innovation sequence with ±2σ bounds (thin)
    # ══════════════════════════════════════════════
    if has_innov:
        ir = row_of['innov']
        innov = np.asarray(innov)
        S = np.asarray(S)
        innov_dim = min(dim, innov.shape[1] - 1) if innov.ndim > 1 else 0
        innov_vals = innov[:, innov_dim] if innov.ndim > 1 else innov.flatten()
        innov_std = np.sqrt(S[:, innov_dim, innov_dim]) if S.ndim == 3 else np.sqrt(S.flatten())

        fig.add_trace(go.Bar(
            x=idx, y=innov_vals,
            marker_color=innov_color,
            marker_line_width=0,
            name='Innovation νₜ',
            showlegend=True,
        ), row=ir, col=1)

        fig.add_trace(go.Scatter(
            x=idx, y=2 * innov_std,
            mode='lines', line=dict(color=bound_color, width=1, dash='dash'),
            name='±2σ Bound',
        ), row=ir, col=1)
        fig.add_trace(go.Scatter(
            x=idx, y=-2 * innov_std,
            mode='lines', line=dict(color=bound_color, width=1, dash='dash'),
            showlegend=False,
        ), row=ir, col=1)

        fig.update_yaxes(title_text='Innovation', row=ir, col=1)

    # ── x-axis label ──
    fig.update_xaxes(title_text='Time', row=n_rows, col=1)

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text=title or f'Kalman Filter State Estimation (dim={dim})',
            font=dict(size=14, family='Arial', color='white'),
        ),
        height=fig_height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(30, 30, 30, 0.9)',
            font_size=12,
            font_family='monospace',
            font_color='white',
            bordercolor='rgba(255, 255, 255, 0.2)'
        ),
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=1.01,
            font=dict(family='monospace', size=11),
        ),
        bargap=0,
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                     zerolinecolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                     zerolinecolor='rgba(128,128,128,0.3)')

    # Remove date gaps if datetime index
    import pandas as pd
    if isinstance(idx, pd.DatetimeIndex):
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    fig.show()

def plot_em_diagnostics(ll_history, Q_history=None, R_history=None,
                        height=550, width=900, title=None):
    """
    EM convergence diagnostics: log-likelihood curve + Q/R norm evolution.

    Parameters
    ----------
    ll_history  : list of float    Log-likelihood at each EM iteration
    Q_history   : list of (n,n)    Q matrix at each iteration, optional
    R_history   : list of (m,m)    R matrix at each iteration, optional
    height      : int              Figure height
    width       : int              Figure width
    title       : str              Custom title, optional
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    has_qr = Q_history is not None and R_history is not None
    n_cols = 2 if has_qr else 1
    subtitles = ['Log-Likelihood']
    if has_qr:
        subtitles.append('‖Q‖ and ‖R‖ (Frobenius Norm)')

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=subtitles,
        horizontal_spacing=0.12,
    )

    iters = np.arange(1, len(ll_history) + 1)

    # ── Left: Log-Likelihood ──
    fig.add_trace(go.Scatter(
        x=iters, y=ll_history,
        mode='lines+markers',
        marker=dict(size=4, color='rgba(0, 200, 255, 0.8)'),
        line=dict(color='rgba(0, 200, 255, 1)', width=2),
        name='Log-Likelihood',
    ), row=1, col=1)

    fig.update_xaxes(title_text='EM Iteration', row=1, col=1)
    fig.update_yaxes(title_text='Log-Likelihood', row=1, col=1)

    # ── Right: Q and R Frobenius norms ──
    if has_qr:
        q_norms = [np.linalg.norm(Q, 'fro') for Q in Q_history]
        r_norms = [np.linalg.norm(R, 'fro') for R in R_history]

        fig.add_trace(go.Scatter(
            x=iters, y=q_norms,
            mode='lines+markers',
            marker=dict(size=4, color='rgba(255, 150, 50, 0.8)'),
            line=dict(color='rgba(255, 150, 50, 1)', width=2),
            name='‖Q‖',
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=iters, y=r_norms,
            mode='lines+markers',
            marker=dict(size=4, color='rgba(100, 255, 150, 0.8)'),
            line=dict(color='rgba(100, 255, 150, 1)', width=2),
            name='‖R‖',
        ), row=1, col=2)

        fig.update_xaxes(title_text='EM Iteration', row=1, col=2)
        fig.update_yaxes(title_text='Frobenius Norm', row=1, col=2)

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text=title or 'EM Convergence Diagnostics',
            font=dict(size=14, family='Arial', color='white'),
        ),
        height=height,
        width=width,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=1.01,
            font=dict(family='monospace', size=11),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                     zerolinecolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                     zerolinecolor='rgba(128,128,128,0.3)')

    fig.show()

def plot_innovation_diagnostics(innov, S, dim=0,
                                height=650, width=900, title=None):
    """
    Model validation via innovation analysis (2 subplots).

    Top     — Normalized innovations (should look like white noise)
    Bottom  — QQ-plot against standard normal — should be linear

    Parameters
    ----------
    innov   : (T, m) array     Innovation residuals from the filter
    S       : (T, m, m) array  Innovation covariances from the filter
    dim     : int              Which observation dimension to analyze
    height  : int              Figure height
    width   : int              Figure width
    title   : str              Custom title, optional
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from scipy import stats

    innov = np.asarray(innov)
    S = np.asarray(S)

    # ── Normalized innovations: S^{-1/2} @ v_t ──
    innov_vals = innov[:, dim] if innov.ndim > 1 else innov.flatten()
    s_vals = S[:, dim, dim] if S.ndim == 3 else S.flatten()
    norm_innov = innov_vals / np.sqrt(s_vals)
    T = len(norm_innov)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
        subplot_titles=[
            'Normalized Innovations (should resemble white noise)',
            'QQ-Plot vs Standard Normal',
        ],
    )

    # ── TOP: Normalized innovations time series ──
    fig.add_trace(go.Scatter(
        x=np.arange(T), y=norm_innov,
        mode='lines',
        line=dict(color='rgba(0, 200, 255, 0.6)', width=0.8),
        name='Norm. Innovation',
    ), row=1, col=1)

    # ±2σ reference lines
    fig.add_hline(y=2, line=dict(color='rgba(255, 80, 80, 0.5)', width=1, dash='dash'),
                  row=1, col=1)
    fig.add_hline(y=-2, line=dict(color='rgba(255, 80, 80, 0.5)', width=1, dash='dash'),
                  row=1, col=1)
    fig.add_hline(y=0, line=dict(color='rgba(128,128,128,0.4)', width=0.5),
                  row=1, col=1)

    fig.update_yaxes(title_text='Std. Residual', row=1, col=1)

    # ── BOTTOM: QQ-plot ──
    sorted_innov = np.sort(norm_innov)
    theoretical_q = stats.norm.ppf(np.linspace(0.001, 0.999, T))

    fig.add_trace(go.Scatter(
        x=theoretical_q, y=sorted_innov,
        mode='markers',
        marker=dict(size=2.5, color='rgba(0, 200, 255, 0.5)'),
        name='QQ Points',
        showlegend=False,
    ), row=2, col=1)

    # Reference 45° line
    qq_min = min(theoretical_q.min(), sorted_innov.min())
    qq_max = max(theoretical_q.max(), sorted_innov.max())
    fig.add_trace(go.Scatter(
        x=[qq_min, qq_max], y=[qq_min, qq_max],
        mode='lines',
        line=dict(color='rgba(255, 80, 80, 0.7)', width=1.5, dash='dash'),
        name='45° Reference',
        showlegend=False,
    ), row=2, col=1)

    fig.update_xaxes(title_text='Theoretical Quantiles', row=2, col=1)
    fig.update_yaxes(title_text='Sample Quantiles', row=2, col=1)

    # ── Layout ──
    fig.update_layout(
        title=dict(
            text=title or 'Innovation Diagnostics',
            font=dict(size=14, family='Arial', color='white'),
        ),
        height=height,
        width=width,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            yanchor='top', y=0.99,
            xanchor='left', x=1.01,
            font=dict(family='monospace', size=11),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                     zerolinecolor='rgba(128,128,128,0.3)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)',
                     zerolinecolor='rgba(128,128,128,0.3)')

    fig.show()
