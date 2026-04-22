# Heston and Dupire Coursework Project

## Introduction

This repository contains a Python coursework project for modelling and pricing options with the Heston stochastic volatility model and a Dupire local volatility model derived from it. The code simulates Heston asset paths, builds option price and implied volatility surfaces, constructs a Dupire local volatility surface, reprices an off-grid option, and compares the Heston, Dupire, and calibrated Heston results.

The project is organised around a single executable workflow in the source folder, with supporting model and utility code separated for clarity.

## Repository Contents

### `src/main.py`

This is the main script that runs the coursework workflow end to end. It:

- sets the initial Heston model parameters,
- simulates asset and variance paths,
- builds the Heston option price surface,
- extracts the implied volatility surface,
- constructs the Dupire local volatility surface,
- prices an off-grid option with Monte Carlo,
- reprices the same option along a future Heston path, and
- performs the Task 6 calibration and comparison.

### `src/VolatilityModel.py`

This file contains the `CourseworkModel` class, which wraps the core Heston model logic. It is responsible for:

- setting up the QuantLib Heston process and pricing engine,
- simulating asset and variance paths,
- pricing European calls under Heston Characteristic function,
- pricing European calls under Dupire local volatility model using Monte Carlo,
- plotting simulated paths and price surfaces for each model, and
- exposing the model parameters in a convenient form.

### `src/utils.py`

This file contains helper functions used by the main workflow. It provides utilities for:

- cleaning and smoothing implied volatility data,
- resampling moving-strike surfaces onto a fixed strike grid,
- building Black variance and local volatility surfaces,
- evaluating local volatility values in a model-agnostic way,
- constructing the calibration basket for Task 6, and
- calibrating the new Heston model against the generated market surface.

## Running the Project

Install the dependencies listed in `requirements.txt`, then run:

```bash
python -m src.main
```

The script will print the task outputs and display the figures used in the coursework.

## Notes

- The project uses a forward-moneyness strike grid for the Heston surface and converts it to a fixed strike grid for Dupire construction.
- QuantLib is required for the Heston pricing and calibration components.
