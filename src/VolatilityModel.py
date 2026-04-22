import numpy as np
import matplotlib.pyplot as plt
import QuantLib as ql

from . import utils

class CourseworkModel:
    def __init__(self, r, kappa, theta, xi, rho, nu0, S0):
        """
        Initialise the Heston model and the QuantLib pricing objects.

        Args:
            r: Risk-free interest rate.
            kappa: Speed of mean reversion.
            theta: Long-run variance level.
            xi: Volatility of volatility.
            rho: Correlation between the asset and variance Brownian motions.
            nu0: Initial variance.
            S0: Initial spot price.
        """
        self.r = float(r)  # Risk-free rate
        self.kappa = float(kappa)  # Speed of mean reversion
        self.theta = float(theta)  # Long-term variance
        self.xi = float(xi)  # Volatility of volatility
        self.rho = float(rho)      # Correlation between W_s and W_nu
        self.nu0 = float(nu0)        # Initial volatility
        self.S0 = float(S0)        # Initial asset price

        self.today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = self.today
        self.calendar = ql.NullCalendar()
        self.day_counter = ql.Actual365Fixed()

        self.spot_quote = ql.SimpleQuote(self.S0)
        self.spot = ql.QuoteHandle(self.spot_quote)
        self.rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.today, self.r, self.day_counter))
        self.div_ts = ql.YieldTermStructureHandle(ql.FlatForward(self.today, 0.0, self.day_counter))

        self.heston_process = ql.HestonProcess(
            self.rf_ts,
            self.div_ts,
            self.spot,
            self.nu0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
        )
        self.heston_model = ql.HestonModel(self.heston_process)
        self.heston_engine = ql.AnalyticHestonEngine(self.heston_model)
        self.heston_black_vol_surface = ql.HestonBlackVolSurface(ql.HestonModelHandle(self.heston_model))

        constant_vol = ql.BlackConstantVol(self.today, self.calendar, 0.20, self.day_counter)
        self.black_scholes_process = ql.BlackScholesMertonProcess(
            self.spot,
            self.div_ts,
            self.rf_ts,
            ql.BlackVolTermStructureHandle(constant_vol),
        )

    def __str__(self):
        return (f"HestonModel(risk_free_rate={self.r}, kappa={self.kappa}, theta={self.theta}, "
                f"xi={self.xi}, rho={self.rho}, nu0={self.nu0}, Feller Condition Satisfied={self.checkFellerCondition()})")
    
    def checkFellerCondition(self):
        """
        Check whether the Heston parameters satisfy the Feller condition.

        Args:
            None.
        """
        # Check Feller Condition for parameters
        lhs = 2 * self.kappa * self.theta
        rhs = self.xi ** 2

        return lhs > rhs
    
    def simulateAssetPaths(self, T=1.0, num_steps=252, num_paths=500, seed = None):
        """
        Simulate asset price and variance paths under the Heston model.

        Args:
            T: Time to maturity.
            num_steps: Number of time steps in the simulation.
            num_paths: Number of simulated paths.
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)
        W_nu = np.random.standard_normal((num_steps, num_paths)) # Brownian increments for variance
        W_s = self.rho*W_nu + np.sqrt(1 - self.rho**2) * np.random.standard_normal((num_steps, num_paths)) # Brownian increments for asset price
        dt = T / num_steps

        log_S = np.zeros((num_steps + 1, num_paths)) # Matrix for log asset prices
        nu = np.zeros((num_steps + 1, num_paths)) # Matrix for volatilities for all paths

        log_S[0, :] = np.log(self.S0) # Setting initial log asset price
        nu[0, :] = self.nu0 # Setting initial volatility for all paths as nu0

        
        # Simulate variance and log-asset price paths using Euler-Maruyama method
        for j in range(1, num_steps + 1):
            # Update variance
            nu_drift = self.kappa * (self.theta - np.maximum(nu[j - 1, :], 0)) * dt
            nu_diffusion = self.xi * np.sqrt(np.maximum(nu[j - 1, :], 0)) * W_nu[j - 1, :] * np.sqrt(dt)
            nu[j, :] = np.maximum(nu[j - 1, :] + nu_drift + nu_diffusion, 0)
            
            # Update log asset price
            S_drift = (self.r - np.maximum(nu[j - 1, :], 0) / 2) * dt
            S_diffusion = np.sqrt(np.maximum(nu[j - 1, :], 0)) * W_s[j - 1, :] * np.sqrt(dt)
            log_S[j, :] = log_S[j - 1, :] + S_drift + S_diffusion
        
           
        # Convert log_S back to asset prices
        S = np.exp(log_S)
        
        return S, nu
    
    def plotPaths(self, S, nu, T, num_steps):
        """
        Plot simulated asset price and variance paths.

        Args:
            S: Simulated asset price paths with shape (num_steps + 1, num_paths).
            nu: Simulated variance paths with shape (num_steps + 1, num_paths).
            T: Time to maturity.
            num_steps: Number of time steps in the simulation.
        """
        t = np.linspace(0, T, num_steps + 1)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot asset price paths
        axes[0].plot(t, S)
        axes[0].axhline(y=self.S0, color='r', linestyle='--', linewidth=2, label=f'Initial S0 = {self.S0}')
        axes[0].set_title('Simulated Paths for asset and volatility under Heston Model')
        axes[0].set_xlabel('Time (Years)')
        axes[0].set_ylabel('Asset Price')
        axes[0].grid()
        axes[0].legend()
        
        # Plot volatility paths
        axes[1].plot(t, nu)
        axes[1].axhline(y=self.nu0, color='r', linestyle='--', linewidth=2, label=f'Initial ν0 = {self.nu0}')
        axes[1].set_xlabel('Time (Years)')
        axes[1].set_ylabel('Volatility')
        axes[1].grid()
        axes[1].legend()
        
        plt.subplots_adjust(hspace=0.2)
        plt.show()

    
    def build_strike_grid(self, num_strikes=50, low_mult=0.7, high_mult=1.7):
        """
        Build the strike grid used for Task 2 pricing.

        Args:
            num_strikes: Number of strike points.
            low_mult: Lower multiple of spot used for the grid start.
            high_mult: Upper multiple of spot used for the grid end.
        """
        # Task 2: keep the forward-moneyness strike grid used for the Heston surface.
        return np.exp(np.linspace(np.log(low_mult * self.S0), np.log(high_mult * self.S0), num_strikes))

    def build_heston_price_surface(self, T_grid, K_grid):
        """
        Build the Heston European call price surface.

        Args:
            T_grid: 1D maturity grid.
            K_grid: 1D strike grid.
        """
        # Task 2: Heston pricing on the maturity/strike grid.
        T_grid = np.asarray(T_grid, dtype=float)
        K_grid = np.asarray(K_grid, dtype=float)
        price_surface = np.zeros((len(T_grid), len(K_grid)), dtype=float)

        for i, T in enumerate(T_grid):
            for j, K in enumerate(K_grid):
                option = ql.VanillaOption(
                    ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
                    ql.EuropeanExercise(self.today + max(1, int(round(float(T) * 365.0))))
                )
                option.setPricingEngine(self.heston_engine)
                price_surface[i, j] = float(option.NPV())

        return price_surface
    
    def plotPriceSurface(self, T_grid, K_surface, price_surface):
        """
        Plot 3D surface of option prices.
        
        Args:
            T_grid: 1D array of maturities.
            K_surface: 2D array of strikes.
            price_surface: 2D array of option prices.
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        T_mesh, K_mesh = np.meshgrid(T_grid, K_surface[0, :], indexing='ij')
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K_mesh, T_mesh, price_surface, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Strike Price ($)', fontsize=11, labelpad=15)
        ax.set_ylabel('Time to Maturity (years)', fontsize=11, labelpad=15)
        ax.set_zlabel('Option Price ($)', fontsize=11, labelpad=15)
        ax.set_title('Heston Model: European Call Price Surface', fontsize=12)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        plt.tight_layout()
        plt.show()

    def build_implied_vol_surface(self, T_grid, K_grid, price_surface=None, min_vol=1e-6, max_vol=4.0):
        """
        Build the implied-volatility surface from the Heston volatility surface.

        Args:
            T_grid: 1D maturity grid.
            K_grid: 1D strike grid.
            price_surface: Optional price surface input, kept for interface consistency.
            min_vol: Minimum allowed implied volatility.
            max_vol: Maximum allowed implied volatility.
        """
        # Task 3: implied vol extraction from the Heston price surface.
        T_grid = np.asarray(T_grid, dtype=float)
        K_grid = np.asarray(K_grid, dtype=float)
        iv_surface = np.full((len(T_grid), len(K_grid)), np.nan, dtype=float)

        for i, T in enumerate(T_grid):
            t_eval = max(float(T), 1.0e-8)
            for j, K in enumerate(K_grid):
                try:
                    iv_surface[i, j] = float(self.heston_black_vol_surface.blackVol(t_eval, float(K), True))
                except RuntimeError:
                    iv_surface[i, j] = np.nan

        return utils.clean_implied_vol_surface(
            T_grid,
            K_grid,
            iv_surface,
            smoothing=2.0,
            min_vol=max(min_vol, 1.0e-3),
            max_vol=max_vol,
        )

    def get_heston_parameters(self):
        """
        Return the current QuantLib Heston parameter set.

        The QuantLib parameter order is (nu0, kappa, xi, rho, theta).
        """
        params = list(self.heston_model.params())
        return {
            "nu0": float(params[0]),
            "kappa": float(params[1]),
            "xi": float(params[2]),
            "rho": float(params[3]),
            "theta": float(params[4]),
        }

    
    def price_heston_call(self, K, T, spot=None):
        """
        Price a European call under the Heston model for a given spot level.

        Args:
            K: Strike price.
            T: Time to maturity.
            spot: Optional spot level to use instead of the current model spot.
        """
        original_spot = self.spot_quote.value()
        try:
            if spot is not None:
                self.spot_quote.setValue(float(spot))

            option = ql.VanillaOption(
                ql.PlainVanillaPayoff(ql.Option.Call, float(K)),
                ql.EuropeanExercise(self.today + max(1, int(round(float(T) * 365.0))))
            )
            option.setPricingEngine(self.heston_engine)
            return float(option.NPV())
        finally:
            if spot is not None:
                self.spot_quote.setValue(float(original_spot))

    def price_local_vol_mc(self, K, T, local_vol_source, num_steps=100, num_paths=10000, seed=None, start_spot=None, start_time=0.0):
        """
        Price a European call by Monte Carlo under local volatility.

        Args:
            K: Strike price.
            T: Time to maturity.
            local_vol_source: Local-volatility surface or callable.
            num_steps: Number of Monte Carlo time steps.
            num_paths: Number of Monte Carlo paths.
            seed: Random seed for reproducibility.
            start_spot: Optional starting spot for the simulation.
            start_time: Time offset used when evaluating the local-vol surface.
        """
        # Task 4: price the off-grid option by Monte Carlo under local volatility.
        if seed is not None:
            np.random.seed(seed)

        dt = float(T) / num_steps
        s0 = self.S0 if start_spot is None else float(start_spot)
        s_paths = np.full(num_paths, s0, dtype=float)

        for step in range(num_steps):
            t = max(float(start_time) + step * dt, 1.0e-8)
            sigmas = np.empty(num_paths, dtype=float)

            for p in range(num_paths):
                try:
                    sigmas[p] = utils.local_vol_value(local_vol_source, t, float(s_paths[p]))
                except RuntimeError:
                    sigmas[p] = np.nan

            sigmas = np.where(np.isfinite(sigmas) & (sigmas > 0.0), sigmas, 0.20)
            z = np.random.standard_normal(num_paths)
            s_paths *= np.exp((self.r - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * z)

        payoff = np.maximum(s_paths - float(K), 0.0)
        return float(np.exp(-self.r * T) * np.mean(payoff))