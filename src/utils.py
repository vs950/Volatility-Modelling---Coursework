import numpy as np
import QuantLib as ql
from scipy.interpolate import RectBivariateSpline

# Spline wrapper used to evaluate the smoothed local-vol surface off-grid.
class BivariateSplineLocalVol:
    def __init__(self, spline, T_grid, K_grid, lower=1.0e-6, upper=4.0):
        """
        Store a smoothed spline together with its grid and clipping bounds.

        Args:
            spline: RectBivariateSpline used to evaluate local volatility.
            T_grid: 1D maturity grid used to build the spline.
            K_grid: 1D strike grid used to build the spline.
            lower: Minimum allowed local-vol value.
            upper: Maximum allowed local-vol value.
        """
        self.spline = spline
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.K_grid = np.asarray(K_grid, dtype=float)
        self.lower = float(lower)
        self.upper = float(upper)

    def localVol(self, t, k, allowExtrapolation=True):
        """
        Evaluate the clipped spline value at a given time and strike.

        Args:
            t: Time coordinate at which to evaluate the surface.
            k: Strike coordinate at which to evaluate the surface.
            allowExtrapolation: Kept for QuantLib-style compatibility.
        """
        t_eval = float(np.clip(t, self.T_grid[0], self.T_grid[-1]))
        k_eval = float(np.clip(k, self.K_grid[0], self.K_grid[-1]))
        value = float(self.spline.ev(t_eval, k_eval))
        return float(np.clip(value, self.lower, self.upper))


# Reinterpolate Task 3 surfaces onto a fixed strike axis for Dupire calibration.
def resample_surface_to_fixed_strike_grid(K_surface, surface, K_fixed_grid):
    """
    Resample a moving-strike surface onto a fixed strike grid.

    Args:
        K_surface: 2D array of strikes for each maturity row.
        surface: 2D array of values defined on K_surface.
        K_fixed_grid: Target strike grid for interpolation.
    """
    K_surface = np.asarray(K_surface, dtype=float)
    surface = np.asarray(surface, dtype=float)
    K_fixed_grid = np.asarray(K_fixed_grid, dtype=float)
    resampled = np.zeros((surface.shape[0], len(K_fixed_grid)), dtype=float)

    for i in range(surface.shape[0]):
        order = np.argsort(K_surface[i, :])
        k_row = K_surface[i, order]
        s_row = surface[i, order]
        resampled[i, :] = np.interp(K_fixed_grid, k_row, s_row)

    return resampled


# Fill missing surface values before clipping and smoothing.
def _fill_nan_surface(surface):
    """
    Fill missing values in a 2D surface by row-wise and column-wise interpolation.

    Args:
        surface: Input surface with possible NaN entries.
    """
    filled = np.array(surface, dtype=float, copy=True)

    for i in range(filled.shape[0]):
        row = filled[i, :]
        mask = np.isfinite(row)
        if np.sum(mask) >= 2:
            x = np.flatnonzero(mask)
            filled[i, ~mask] = np.interp(np.flatnonzero(~mask), x, row[mask])

    for j in range(filled.shape[1]):
        col = filled[:, j]
        mask = np.isfinite(col)
        if np.sum(mask) >= 2:
            x = np.flatnonzero(mask)
            filled[~mask, j] = np.interp(np.flatnonzero(~mask), x, col[mask])

    if np.any(~np.isfinite(filled)):
        finite = filled[np.isfinite(filled)]
        fallback = float(np.median(finite)) if finite.size else 0.20
        filled = np.where(np.isfinite(filled), filled, fallback)

    return filled


# Clean the implied-vol grid before building the local-vol term structure.
def clean_implied_vol_surface(T_grid, K_grid, iv_surface, smoothing=2.0, min_vol=1.0e-3, max_vol=4.0):
    """
    Clean, clip, and smooth an implied-volatility surface.

    Args:
        T_grid: 1D maturity grid.
        K_grid: 1D strike grid.
        iv_surface: Raw implied-vol surface.
        smoothing: Spline smoothing factor.
        min_vol: Lower bound for implied volatility.
        max_vol: Upper bound for implied volatility.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    K_grid = np.asarray(K_grid, dtype=float)
    iv_surface = np.asarray(iv_surface, dtype=float)

    cleaned = np.array(iv_surface, copy=True)
    cleaned = np.where(cleaned > min_vol, cleaned, np.nan)
    cleaned = _fill_nan_surface(cleaned)

    valid = cleaned[np.isfinite(cleaned)]
    if valid.size > 0:
        low_q = np.percentile(valid, 1.0)
        high_q = np.percentile(valid, 99.0)
        low_clip = max(float(min_vol), float(low_q * 0.8))
        high_clip = min(float(max_vol), float(high_q * 1.2))
        cleaned = np.clip(cleaned, low_clip, high_clip)
    else:
        cleaned = np.full_like(cleaned, 0.20)

    spline = RectBivariateSpline(T_grid, K_grid, cleaned, kx=3, ky=3, s=float(smoothing))
    smoothed = np.asarray(spline(T_grid, K_grid), dtype=float)
    smoothed = np.clip(smoothed, min_vol, max_vol)
    return smoothed


# Build the QuantLib variance surface that stores w(T,K)=sigma_impl^2(T,K)T.
def build_black_variance_surface(today, calendar, day_counter, T_grid, K_grid, implied_vol_surface):
    """
    Build a QuantLib BlackVarianceSurface from an implied-volatility grid.

    Args:
        today: QuantLib evaluation date.
        calendar: QuantLib calendar used for date generation.
        day_counter: QuantLib day counter.
        T_grid: 1D maturity grid.
        K_grid: 1D strike grid.
        implied_vol_surface: Smoothed implied-vol surface.
    """
    # QuantLib stores the surface as total variance w(T,K)=sigma^2(T,K)T.
    T_grid = np.asarray(T_grid, dtype=float)
    K_grid = np.asarray(K_grid, dtype=float)
    implied_vol_surface = np.asarray(implied_vol_surface, dtype=float)

    dates = [today + max(1, int(round(float(T) * 365.0))) for T in T_grid]
    vol_matrix = ql.Matrix(len(K_grid), len(T_grid))

    for i in range(len(T_grid)):
        for j in range(len(K_grid)):
            vol = float(implied_vol_surface[i, j])
            if not np.isfinite(vol):
                vol = 0.20
            vol_matrix[j][i] = float(np.clip(vol, 1e-6, 4.0))

    black_var_surface = ql.BlackVarianceSurface(
        today,
        calendar,
        dates,
        [float(k) for k in K_grid],
        vol_matrix,
        day_counter,
    )
    black_var_surface.enableExtrapolation()
    return black_var_surface


# Wrap the variance surface in QuantLib's local-vol machinery for Dupire pricing.
def build_local_vol_surface(today, calendar, day_counter, risk_free_ts, div_ts, spot, T_grid, K_grid, implied_vol_surface):
    """
    Build QuantLib local-volatility objects from a variance surface.

    Args:
        today: QuantLib evaluation date.
        calendar: QuantLib calendar used for date generation.
        day_counter: QuantLib day counter.
        risk_free_ts: Risk-free yield term structure handle.
        div_ts: Dividend yield term structure handle.
        spot: Spot quote handle.
        T_grid: 1D maturity grid.
        K_grid: 1D strike grid.
        implied_vol_surface: Smoothed implied-vol surface.
    """
    black_var_surface = build_black_variance_surface(today, calendar, day_counter, T_grid, K_grid, implied_vol_surface)
    local_vol_surface = ql.LocalVolSurface(
        ql.BlackVolTermStructureHandle(black_var_surface),
        risk_free_ts,
        div_ts,
        spot,
    )
    return black_var_surface, local_vol_surface


def build_local_vol_surface_from_implied_surface(today, calendar, day_counter, risk_free_ts, div_ts, spot, T_grid, K_surface, implied_vol_surface, K_fixed_grid=None):
    """
    Build local-vol objects from an implied-vol surface defined on a moving strike grid.

    This resamples the Task 2/3 moving-strike surface onto a fixed strike grid so Dupire
    local-vol construction can use a single strike axis across maturities.

    Args:
        today: QuantLib evaluation date.
        calendar: QuantLib calendar used for date generation.
        day_counter: QuantLib day counter.
        risk_free_ts: Risk-free yield term structure handle.
        div_ts: Dividend yield term structure handle.
        spot: Spot quote handle.
        T_grid: 1D maturity grid.
        K_surface: 2D strike grid from the Task 2 surface.
        implied_vol_surface: Implied-vol surface on K_surface.
        K_fixed_grid: Optional fixed strike grid for resampling.
    """
    # Reuse the Task 3 IV surface directly; only the strike axis is interpolated.
    if K_fixed_grid is None:
        K_fixed_grid = K_surface[0, :]

    implied_vol_fixed_grid = resample_surface_to_fixed_strike_grid(K_surface, implied_vol_surface, K_fixed_grid)
    return build_local_vol_surface(today, calendar, day_counter, risk_free_ts, div_ts, spot, T_grid, K_fixed_grid, implied_vol_fixed_grid)


def local_vol_value(local_vol_source, t, s):
    """
    Evaluate a local-vol source at a given time and spot level.

    Args:
        local_vol_source: Object exposing localVol(), ev(), or callable behaviour.
        t: Time at which to evaluate local volatility.
        s: Spot level at which to evaluate local volatility.
    """
    if hasattr(local_vol_source, "localVol"):
        return float(local_vol_source.localVol(float(t), float(s), True))
    if hasattr(local_vol_source, "ev"):
        return float(local_vol_source.ev(float(t), float(s)))
    if callable(local_vol_source):
        return float(local_vol_source(float(t), float(s)))
    raise TypeError("local_vol_source must expose localVol(), ev(), or be callable.")


def build_implied_vol_surface_on_surface(heston_black_vol_surface, T_grid, K_surface, min_vol=1.0e-6, max_vol=4.0):
    """
    Build implied volatility on a moving-strike surface directly from a Heston black-vol surface.

    Args:
        heston_black_vol_surface: QuantLib HestonBlackVolSurface.
        T_grid: 1D maturity grid.
        K_surface: 2D strike grid from the Task 2 surface.
        min_vol: Minimum allowed implied volatility.
        max_vol: Maximum allowed implied volatility.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    K_surface = np.asarray(K_surface, dtype=float)
    iv_surface = np.full_like(K_surface, np.nan, dtype=float)

    for i, T in enumerate(T_grid):
        t_eval = max(float(T), 1.0e-8)
        for j, K in enumerate(K_surface[i, :]):
            try:
                iv_surface[i, j] = float(heston_black_vol_surface.blackVol(t_eval, float(K), True))
            except RuntimeError:
                iv_surface[i, j] = np.nan

    return clean_implied_vol_surface(
        T_grid,
        K_surface[0, :],
        iv_surface,
        smoothing=2.0,
        min_vol=max(min_vol, 1.0e-3),
        max_vol=max_vol,
    )


def build_heston_calibration_basket(T_grid, K_surface, implied_vol_surface, maturity_indices=None, strike_indices=None, min_vol=1.0e-3, max_vol=4.0):
    """
    Build a calibration basket from the Task 2 surface.

    Each basket entry stores the selected maturity, strike, and market implied vol.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    K_surface = np.asarray(K_surface, dtype=float)
    implied_vol_surface = np.asarray(implied_vol_surface, dtype=float)

    if maturity_indices is None:
        maturity_indices = np.linspace(3, len(T_grid) - 4, 5, dtype=int)
    else:
        maturity_indices = np.asarray(maturity_indices, dtype=int)

    if strike_indices is None:
        strike_indices = np.linspace(4, K_surface.shape[1] - 5, 5, dtype=int)
    else:
        strike_indices = np.asarray(strike_indices, dtype=int)

    basket = []
    for i in maturity_indices:
        maturity_days = max(1, int(round(float(T_grid[i]) * 365.0)))
        for j in strike_indices:
            market_vol = float(implied_vol_surface[i, j])
            if not np.isfinite(market_vol):
                continue

            basket.append({
                "i": int(i),
                "j": int(j),
                "T": float(T_grid[i]),
                "maturity_days": int(maturity_days),
                "K": float(K_surface[i, j]),
                "vol": float(np.clip(market_vol, min_vol, max_vol)),
            })

    return basket


def calibrate_heston_model(model, calibration_basket, error_type=None, max_iterations=200, function_evals=100, root_eps=1.0e-8, ftol=1.0e-8, gtol=1.0e-8):
    """
    Calibrate a CourseworkModel instance to an explicit calibration basket.

    Args:
        model: CourseworkModel instance.
        calibration_basket: List of basket entries containing maturity_days, K, and vol.
        error_type: QuantLib calibration error type.
        max_iterations: Maximum optimizer iterations.
        function_evals: Stationary-state iteration cap used by QuantLib EndCriteria.
        root_eps: Root epsilon.
        ftol: Function epsilon.
        gtol: Gradient epsilon.
    """
    if error_type is None:
        error_type = ql.BlackCalibrationHelper.RelativePriceError

    helpers = []
    basket = list(calibration_basket)
    for entry in basket:
        if isinstance(entry, dict):
            maturity_days = int(entry["maturity_days"])
            strike = float(entry["K"])
            market_vol = float(entry["vol"])
        else:
            maturity_days, strike, market_vol = entry[:3]
            maturity_days = int(maturity_days)
            strike = float(strike)
            market_vol = float(market_vol)

        helper = ql.HestonModelHelper(
            ql.Period(maturity_days, ql.Days),
            model.calendar,
            model.S0,
            strike,
            ql.QuoteHandle(ql.SimpleQuote(market_vol)),
            model.rf_ts,
            model.div_ts,
            error_type,
        )
        helpers.append(helper)

    for helper in helpers:
        helper.setPricingEngine(model.heston_engine)

    before_params = list(model.heston_model.params())
    before = {
        "nu0": float(before_params[0]),
        "kappa": float(before_params[1]),
        "xi": float(before_params[2]),
        "rho": float(before_params[3]),
        "theta": float(before_params[4]),
    }

    optimiser = ql.LevenbergMarquardt()
    stationary_iterations = min(int(function_evals), max(1, int(max_iterations) - 1))
    end_criteria = ql.EndCriteria(max_iterations, stationary_iterations, root_eps, ftol, gtol)
    model.heston_model.calibrate(helpers, optimiser, end_criteria)

    after_params = list(model.heston_model.params())
    after = {
        "nu0": float(after_params[0]),
        "kappa": float(after_params[1]),
        "xi": float(after_params[2]),
        "rho": float(after_params[3]),
        "theta": float(after_params[4]),
    }

    errors = np.array([abs(float(helper.calibrationError())) for helper in helpers], dtype=float)

    # Keep the public attributes aligned with the calibrated QuantLib model.
    model.nu0 = after["nu0"]
    model.kappa = after["kappa"]
    model.xi = after["xi"]
    model.rho = after["rho"]
    model.theta = after["theta"]

    return {
        "before": before,
        "after": after,
        "basket": basket,
        "num_helpers": len(helpers),
        "mean_abs_calibration_error": float(np.mean(errors)) if errors.size else float("nan"),
        "max_abs_calibration_error": float(np.max(errors)) if errors.size else float("nan"),
        "helpers": helpers,
    }


def evaluate_local_vol_grid(T_grid, K_grid, local_vol_surface):
    """
    Evaluate the local-vol surface on a regular maturity/strike grid.

    Args:
        T_grid: 1D maturity grid.
        K_grid: 1D strike grid.
        local_vol_surface: QuantLib local-vol object.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    K_grid = np.asarray(K_grid, dtype=float)
    local_vol_grid = np.zeros((len(T_grid), len(K_grid)), dtype=float)

    for i, T in enumerate(T_grid):
        t_eval = max(float(T), 1.0e-8)
        for j, K in enumerate(K_grid):
            try:
                lv = float(local_vol_surface.localVol(t_eval, float(K), True))
            except RuntimeError:
                lv = np.nan
            local_vol_grid[i, j] = lv

    return local_vol_grid


def smooth_local_vol_surface(T_grid, K_grid, local_vol_grid, smoothing=0.5, lower=1.0e-6, upper=4.0):
    """
    Smooth and clip a raw local-volatility grid.

    Args:
        T_grid: 1D maturity grid.
        K_grid: 1D strike grid.
        local_vol_grid: Raw local-vol values sampled from QuantLib.
        smoothing: Spline smoothing factor.
        lower: Minimum allowed local-vol value.
        upper: Maximum allowed local-vol value.
    """
    # Remove NaNs, clip extreme values, then smooth the raw local-vol grid.
    T_grid = np.asarray(T_grid, dtype=float)
    K_grid = np.asarray(K_grid, dtype=float)
    local_vol_grid = np.asarray(local_vol_grid, dtype=float)

    cleaned = _fill_nan_surface(local_vol_grid)
    cleaned = np.clip(cleaned, lower, upper)

    valid = cleaned[np.isfinite(cleaned)]
    if valid.size > 0:
        low_q = np.percentile(valid, 1.0)
        high_q = np.percentile(valid, 99.0)
        low_clip = max(float(lower), float(low_q * 0.5))
        high_clip = min(float(upper), float(high_q * 1.5))
        cleaned = np.clip(cleaned, low_clip, high_clip)

    spline = RectBivariateSpline(T_grid, K_grid, cleaned, kx=3, ky=3, s=float(smoothing))
    smoothed = np.asarray(spline(T_grid, K_grid), dtype=float)
    smoothed = np.clip(smoothed, lower, upper)
    spline_wrapper = BivariateSplineLocalVol(spline, T_grid, K_grid, lower=lower, upper=upper)
    return cleaned, spline, smoothed, spline_wrapper