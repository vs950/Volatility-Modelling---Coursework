import time

import numpy as np
import matplotlib.pyplot as plt

from .VolatilityModel import CourseworkModel
from . import utils as heston_utils

# Initial parameters for the Heston model
r = 0.05
kappa = 1.5
theta = 0.04
xi = 0.3
rho = -0.7
nu0 = 0.04
S0 = 200

# Simulation parameters
T = 1.0
num_steps = 504
num_paths = 10

# Create an instance of the Heston model with the specified parameters
heston_model = CourseworkModel(r=r, kappa=kappa, theta=theta, xi=xi, rho=rho, nu0=nu0, S0=S0)

# Print model parameters and Feller condition status
print(heston_model)

#================ Task 1: Simulate asset price and volatility paths ================#
S, nu = heston_model.simulateAssetPaths(T=T, num_steps=num_steps, num_paths=num_paths, seed=69)
heston_model.plotPaths(S, nu, T=T, num_steps=num_steps)



#================ Task 2: Price surface: 50x50 grid over maturity and forward moneyness to compute Strikes ================#
T_grid = np.linspace(0.08, 2.0, 50)
m_f_grid = np.linspace(0.7, 1.3, 50)

price_surface = np.zeros((len(T_grid), len(m_f_grid)))
K_surface = np.zeros((len(T_grid), len(m_f_grid)))

for i, T in enumerate(T_grid):
    F = S0 * np.exp(r * T)
    K_vals = m_f_grid * F
    K_surface[i, :] = K_vals
    price_surface[i, :] = heston_model.build_heston_price_surface([T], K_vals)[0]

print("\nTask 2: QuantLib Heston price surface")
print(f"  Forward moneyness grid: [{m_f_grid[0]:.2f}, {m_f_grid[-1]:.2f}]")
print(f"  Price surface min / max: ${np.min(price_surface):.4f} / ${np.max(price_surface):.4f}")

# Plot the price surface
heston_model.plotPriceSurface(T_grid, K_surface, price_surface)



#================ Task 3: Compute implied volatilities for the price surface ================#
implied_vol_surface = heston_utils.build_implied_vol_surface_on_surface(
    heston_model.heston_black_vol_surface,
    T_grid,
    K_surface,
)

print("\nTask 3: QuantLib implied volatility surface")
print(f"  Implied vol min / max: {np.nanmin(implied_vol_surface):.4f} / {np.nanmax(implied_vol_surface):.4f}")
print(f"  NaN count: {np.isnan(implied_vol_surface).sum()}")
print(f"  Short maturity wing IVs (left/right): {implied_vol_surface[0, 0]:.4f} / {implied_vol_surface[0, -1]:.4f}")

# Plot the implied volatility surface
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
T_mesh, K_mesh = np.meshgrid(T_grid, K_surface[0, :], indexing='ij')
surf = ax.plot_surface(K_mesh, T_mesh, implied_vol_surface * 100, cmap='inferno', alpha=0.8)
ax.set_xlabel('Strike Price ($)', fontsize=11, labelpad=15)
ax.set_ylabel('Time to Maturity (years)', fontsize=11, labelpad=15)
ax.set_zlabel('Implied Volatility (%)', fontsize=11, labelpad=15)
ax.set_title('Implied Volatility Surface', fontsize=12)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
plt.show()



# Plot volatility smile curves at key maturities
# Select 4 representative maturities: short, medium-short, medium-long, long
smile_indices = [5, 15, 30, 45]  # Indices into T_grid for distinct maturities
smile_labels = [f'T={T_grid[i]:.2f}y' for i in smile_indices]

fig, ax = plt.subplots(figsize=(12, 7))

for idx, t_idx in enumerate(smile_indices):
    iv_curve = implied_vol_surface[t_idx, :] * 100
    valid_mask = np.isfinite(iv_curve)
    # Convert strike to moneyness S0/F(T) for each maturity slice.
    F_t = S0 * np.exp(r * T_grid[t_idx])
    moneyness = K_surface[t_idx, valid_mask] / F_t
    ax.plot(moneyness, iv_curve[valid_mask],
            marker='o', linewidth=2, markersize=4, label=smile_labels[idx], alpha=0.8)

ax.set_xlabel('Forward Moneyness K / F(T)', fontsize=12)
ax.set_ylabel('Implied Volatility (%)', fontsize=12)
ax.set_title('Volatility Smile at Different Maturities', fontsize=13)
ax.grid(True, alpha=0.3)
ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='ATM')
ax.legend(fontsize=11, loc='best')
plt.show()

#================ Task 4: Calibrate Dupire Local Volatility Model ================#
print("\n" + "="*80)
print("DUPIRE LOCAL VOLATILITY MODEL - OFF-GRID OPTION PRICING")
print("="*80)

print("\nCalibrating Dupire local volatility model on 50x50 option price grid...")

# Task 4 converts the Task 2 moving-strike grid to a fixed strike grid before applying Dupire,
# because the local-vol construction needs a stable strike axis for interpolation and evaluation.
# The helper also returns the black variance surface for reference.
black_var_surface, local_vol_surface = heston_utils.build_local_vol_surface_from_implied_surface(
    heston_model.today,
    heston_model.calendar,
    heston_model.day_counter,
    heston_model.rf_ts,
    heston_model.div_ts,
    heston_model.spot,
    T_grid,
    K_surface,
    implied_vol_surface,
) 
# Reuse the fixed strike axis extracted from the Task 2 surface after the moving-to-fixed conversion above.
K_fixed_grid = K_surface[0, :]
local_vol_grid = heston_utils.evaluate_local_vol_grid(T_grid, K_fixed_grid, local_vol_surface)
local_vol_clean_grid, local_vol_spline, local_vol_smoothed_grid, local_vol_wrapper = heston_utils.smooth_local_vol_surface(
    T_grid,
    K_fixed_grid,
    local_vol_grid,
    smoothing=200.0,
    lower=1.0e-6,
    upper=4.0,
)

print("Task 4: Dupire local volatility surface")
print(f"  Local vol min / max: {np.nanmin(local_vol_grid):.4f} / {np.nanmax(local_vol_grid):.4f}")
print(f"  NaN count: {np.isnan(local_vol_grid).sum()}")
print(f"  Cleaned local vol min / max: {np.min(local_vol_clean_grid):.4f} / {np.max(local_vol_clean_grid):.4f}")
print(f"  Smoothed local vol min / max: {np.min(local_vol_smoothed_grid):.4f} / {np.max(local_vol_smoothed_grid):.4f}")

# Plot Dupire Local Volatility Surface 
print("Plotting Dupire calibrated local volatility surface...")
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_mesh, T_mesh, local_vol_smoothed_grid * 100.0, cmap='viridis', alpha=0.8)
ax.set_xlabel('Strike Price ($)', fontsize=11, labelpad=15)
ax.set_ylabel('Time to Maturity (years)', fontsize=11, labelpad=15)
ax.set_zlabel('Local Volatility (%)', fontsize=11, labelpad=15)
ax.set_title('Dupire Local Volatility Surface (Smoothed)', fontsize=12)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
plt.tight_layout()
plt.show()


# Price off-grid option using Dupire local volatility model and Monte Carlo.
# Use a maturity above 1 year that is not on the original 50x50 grid.
T_offgrid = 1.17
K_offgrid = 210.5

print(f"\nOff-Grid Option Pricing:")
print(f"  Current asset price (S0): ${S0:.2f}")
print(f"  Strike price (K): ${K_offgrid:.2f} (off-grid)")
print(f"  Time to maturity (T): {T_offgrid:.2f} years (off-grid)")
print(f"  Risk-free rate (r): {r:.4f}")

print(f"\nComputing price using Dupire local volatility + Monte Carlo...")
price_dupire = heston_model.price_local_vol_mc(
    K_offgrid, T_offgrid, local_vol_surface, num_steps=100, num_paths=10000, seed=42
)

print(f"\n>> Off-grid option price (Dupire MC): ${price_dupire:.4f}")

# Validate accuracy against the direct Heston price
print("\n" + "-"*80)
print("Validation: Compare Dupire price to direct Heston price at the same K and T")
print("-"*80)

# Price the same off-grid option directly with the Heston model.
heston_price_direct = heston_model.build_heston_price_surface([T_offgrid], [K_offgrid])[0, 0]

print(f"\nOff-grid point: (K=${K_offgrid:.2f}, T={T_offgrid:.4f})")
print(f"  Heston direct price: ${heston_price_direct:.4f}")
print(f"  Dupire MC price: ${price_dupire:.4f}")
print(f"\nPrice difference from direct Heston price: ${abs(price_dupire - heston_price_direct):.4f}")


#================ Task 5: Reprice the same option along one future Heston path ================#
print("\n" + "="*80)
print("Task 5: FUTURE-STATE PRICING ALONG ONE SIMULATED HESTON PATH")
print("="*80)

path_index = 0
num_observation_points = 6

dt_path = T / num_steps
max_obs_index = max(1, int(np.floor(T_offgrid / dt_path)) - 1)
obs_indices = np.unique(np.round(np.linspace(0, max_obs_index, num_observation_points)).astype(int))
obs_times = obs_indices * dt_path
obs_spots = S[obs_indices, path_index]
remaining_maturities = T_offgrid - obs_times

future_heston_prices = []
future_dupire_prices = []

for i, (t_now, spot_now, tau_now) in enumerate(zip(obs_times, obs_spots, remaining_maturities)):
    heston_future = heston_model.price_heston_call(K_offgrid, tau_now, spot=spot_now)
    dupire_future = heston_model.price_local_vol_mc(
        K_offgrid,
        tau_now,
        local_vol_surface,
        num_steps=60,
        num_paths=3000,
        seed=100 + i,
        start_spot=spot_now,
        start_time=t_now,
    )

    future_heston_prices.append(heston_future)
    future_dupire_prices.append(dupire_future)

future_heston_prices = np.asarray(future_heston_prices, dtype=float)
future_dupire_prices = np.asarray(future_dupire_prices, dtype=float)
future_abs_gap = np.abs(future_dupire_prices - future_heston_prices)
future_rel_gap = 100.0 * future_abs_gap / np.maximum(future_heston_prices, 1.0e-12)
moneyness_obs = obs_spots / K_offgrid
true_vol_obs = np.sqrt(np.maximum(nu[obs_indices, path_index], 0.0))
dupire_local_vol_obs = np.array([
    float(local_vol_surface.localVol(max(float(t_now), 1.0e-8), float(spot_now), True))
    for t_now, spot_now in zip(obs_times, obs_spots)
], dtype=float)

print(f"\nPath used for Task 5: {path_index}")
print(f"Observation dates before execution: {len(obs_times)}")
print("\n  t       S_t    Moneyness    Heston      Dupire      Abs gap     Rel gap   Heston vol  Dupire vol")
for t_now, spot_now, mon_now, h_price, d_price, abs_gap, rel_gap, h_vol_now, d_vol_now in zip(
    obs_times,
    obs_spots,
    moneyness_obs,
    future_heston_prices,
    future_dupire_prices,
    future_abs_gap,
    future_rel_gap,
    true_vol_obs,
    dupire_local_vol_obs,
):
    print(
        f"  {t_now:6.4f}  {spot_now:8.3f}    {mon_now:7.3f}  "
        f"{h_price:9.4f}  {d_price:9.4f}  {abs_gap:9.4f}  {rel_gap:8.3f}%  "
        f"{h_vol_now * 100.0:10.3f}%  {d_vol_now * 100.0:9.3f}%"
    )

print(f"\nTask 5 summary:")
print(f"  Mean absolute gap: ${np.mean(future_abs_gap):.4f}")
print(f"  Max absolute gap:  ${np.max(future_abs_gap):.4f}")
print(f"  Mean relative gap:  {np.mean(future_rel_gap):.3f}%")
print(f"  Max relative gap:   {np.max(future_rel_gap):.3f}%")

time_full = np.arange(max_obs_index + 1) * dt_path
path_slice = S[:max_obs_index + 1, path_index]
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(obs_times)))

fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
fig.suptitle('Repricing one simulated Heston path without recalibration', fontsize=15)

# Panel 1: underlying path and valuation dates.
ax = axes[0, 0]
ax.plot(time_full, path_slice, color='dimgray', linewidth=2, label='Heston path')
ax.scatter(obs_times, obs_spots, c=colors, s=70, edgecolors='black', linewidths=0.5, zorder=3, label='Repricing dates')
ax.axhline(K_offgrid, color='tab:red', linestyle='--', linewidth=1.8, alpha=0.8, label='Strike')
ax.axvline(T_offgrid, color='tab:blue', linestyle=':', linewidth=2.0, alpha=0.9, label='Execution date')
ax.set_title('Underlying asset path', pad=12)
ax.set_xlabel('Time', labelpad=10)
ax.set_ylabel('Asset price', labelpad=10)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc='best')

# Panel 2: model prices through time.
ax = axes[0, 1]
ax.plot(obs_times, future_heston_prices, marker='o', color='tab:blue', linewidth=2.2, label='Heston price')
ax.plot(obs_times, future_dupire_prices, marker='s', color='tab:orange', linewidth=2.2, label='Dupire price')
ax.fill_between(obs_times, future_heston_prices, future_dupire_prices, color='gray', alpha=0.15, label='Pricing gap')
ax.set_title('Future repricing of the same option', pad=12)
ax.set_xlabel('Calendar time', labelpad=10)
ax.set_ylabel('Option price', labelpad=10)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc='best')

# Panel 3: absolute pricing gap.
ax = axes[1, 0]
ax.plot(obs_times, future_abs_gap, marker='o', color='tab:red', linewidth=2.0, label='Absolute gap')
ax.set_title('Model gap as time evolves', pad=12)
ax.set_xlabel('Calendar time', labelpad=10)
ax.set_ylabel('Absolute price gap', labelpad=10)
ax.grid(True, alpha=0.25)
ax.legend(loc='best')

# Panel 4: true Heston instantaneous vol versus Dupire local vol.
ax = axes[1, 1]
ax.plot(obs_times, true_vol_obs * 100.0, marker='o', linewidth=2.2, color='tab:blue', label=r'Heston instantaneous vol $\sqrt{\nu_t}$')
ax.plot(obs_times, dupire_local_vol_obs * 100.0, marker='s', linewidth=2.2, color='tab:orange', label='Dupire local volatility')
ax.fill_between(
    obs_times,
    true_vol_obs * 100.0,
    dupire_local_vol_obs * 100.0,
    color='gray',
    alpha=0.15,
    label='Volatility gap',
)
ax.set_title('Heston instantaneous volatility versus Dupire local volatility', pad=12)
ax.set_xlabel('Calendar time', labelpad=10)
ax.set_ylabel('Volatility (%)', labelpad=10)
ax.grid(True, alpha=0.25)
ax.legend(loc='best')

plt.show()

#================ Task 6: Calibrate a new Heston model to the Task 2 price surface ================#
print("\n" + "="*80)
print("TASK 6: CALIBRATE A NEW HESTON MODEL TO THE TASK 2 PRICE SURFACE")
print("="*80)

# Start from a blind but still Feller-satisfying parameter guess that is deliberately offset from Task 1.
calibration_model = CourseworkModel(r=r, kappa=0.8, theta=0.07, xi=0.3, rho=-0.3, nu0=0.01, S0=S0)
print("Blind initial model:")
print(calibration_model)

# Build a 7 x 7 interior calibration basket and minimise the relative price error over those helpers.
cal_maturity_indices = np.array([5, 11, 17, 24, 31, 38, 44], dtype=int)
cal_strike_indices = np.array([5, 11, 17, 24, 31, 38, 44], dtype=int)
calibration_basket = heston_utils.build_heston_calibration_basket(
    T_grid,
    K_surface,
    implied_vol_surface,
    maturity_indices=cal_maturity_indices,
    strike_indices=cal_strike_indices,
)

blind_time0_price = calibration_model.price_heston_call(K_offgrid, T_offgrid)
blind_future_prices = np.array([
    calibration_model.price_heston_call(K_offgrid, tau_now, spot=spot_now)
    for spot_now, tau_now in zip(obs_spots, remaining_maturities)
], dtype=float)

calibration_start = time.perf_counter()
calibration_report = heston_utils.calibrate_heston_model(
    calibration_model,
    calibration_basket,
    max_iterations=50,
)
calibration_elapsed = time.perf_counter() - calibration_start

cal_time0_price = calibration_model.price_heston_call(K_offgrid, T_offgrid)
cal_future_prices = np.array([
    calibration_model.price_heston_call(K_offgrid, tau_now, spot=spot_now)
    for spot_now, tau_now in zip(obs_spots, remaining_maturities)
], dtype=float)

blind_time0_error = abs(blind_time0_price - heston_price_direct)
cal_time0_error = abs(cal_time0_price - heston_price_direct)
blind_future_abs_err = np.abs(blind_future_prices - future_heston_prices)
cal_future_abs_err = np.abs(cal_future_prices - future_heston_prices)
dupire_time0_error = abs(future_dupire_prices[0] - future_heston_prices[0])
dupire_future_abs_err = np.abs(future_dupire_prices - future_heston_prices)

rel_dupire_future_err = 100.0 * dupire_future_abs_err / np.maximum(future_heston_prices, 1.0e-12)
rel_cal_future_err = 100.0 * cal_future_abs_err / np.maximum(future_heston_prices, 1.0e-12)

print(f"\nCalibration basket size: {calibration_report['num_helpers']} points")
print(f"Calibration runtime: {calibration_elapsed:.2f} seconds")
print(f"Mean basket calibration error (implied vol points): {calibration_report['mean_abs_calibration_error']:.6f}")
print(f"Max basket calibration error  (implied vol points): {calibration_report['max_abs_calibration_error']:.6f}")

print("\nCalibrated Heston parameters:")
for label in ["nu0", "kappa", "xi", "rho", "theta"]:
    before_val = calibration_report['before'][label]
    after_val = calibration_report['after'][label]
    print(f"  {label:>5s}: {before_val:.6f} -> {after_val:.6f}")

print("\nTask 6 pricing accuracy against the true Task 1 Heston model")
print("  t       moneyness    True price    Dupire price   Dupire err  Dupire %   Calibrated    Cal err     Cal %")
for t_now, mon_now, true_price, dupire_price, cal_price, dupire_gap, cal_gap in zip(
    obs_times,
    moneyness_obs,
    future_heston_prices,
    future_dupire_prices,
    cal_future_prices,
    dupire_future_abs_err,
    cal_future_abs_err,
):
    dupire_pct = 100.0 * dupire_gap / max(true_price, 1.0e-12)
    cal_pct = 100.0 * cal_gap / max(true_price, 1.0e-12)
    print(
        f"  {t_now:6.4f}    {mon_now:7.3f}     {true_price:9.4f}   {dupire_price:11.4f}   {dupire_gap:9.4f}   {dupire_pct:8.3f}%   {cal_price:9.4f}   {cal_gap:8.4f}   {cal_pct:7.3f}%"
    )

print("\nTask 6 summary versus the true Heston model")
print(f"  Mean Dupire future error:   ${np.mean(dupire_future_abs_err):.4f}")
print(f"  Mean calibrated error:      ${np.mean(cal_future_abs_err):.4f}")
print(f"  Max Dupire future error:    ${np.max(dupire_future_abs_err):.4f}")
print(f"  Max calibrated error:       ${np.max(cal_future_abs_err):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
fig.suptitle('Task 6: True Heston, Dupire, and calibrated Heston comparison', fontsize=15)

# Panel 1: prices through time.
ax = axes[0]
ax.plot(obs_times, future_heston_prices, marker='o', linewidth=2.2, color='tab:blue', label='True Heston')
ax.plot(obs_times, future_dupire_prices, marker='s', linewidth=2.2, color='tab:orange', label='Dupire')
ax.plot(obs_times, cal_future_prices, marker='^', linewidth=2.2, color='black', linestyle='--', label='Calibrated Heston')
ax.set_title('Option prices along the future path', pad=12)
ax.set_xlabel('Calendar time', labelpad=10)
ax.set_ylabel('Option price', labelpad=10)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc='best')

# Panel 2: relative pricing error versus the true model.
ax = axes[1]
ax.plot(obs_times, rel_dupire_future_err, marker='s', linewidth=2.0, color='tab:orange', label='Dupire relative error')
ax.plot(obs_times, rel_cal_future_err, marker='^', linewidth=2.0, color='black', linestyle='--', label='Calibrated relative error')
ax.set_title('Relative pricing error versus the true model', pad=12)
ax.set_xlabel('Calendar time', labelpad=10)
ax.set_ylabel('Relative error (%)', labelpad=10)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc='best')

plt.show()

print("\n" + "="*80)
print("="*80)