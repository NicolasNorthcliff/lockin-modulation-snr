# Lock-in Modulation SNR Simulator

An interactive Python/Streamlit simulation comparing sinusoidal and square-wave optical modulation for lock-in detection.

## Features

- Time-domain comparison with identical noise
- Frequency-domain (FFT) analysis
- Digital lock-in detection using X/Y demodulation
- Monte Carlo estimation of recovered signal and SNR
- Adjustable modulation frequency, detector bandwidth, integration time, and noise level
- Comparison under different modulation normalization conditions

## Digital Lock-in Model

The simulated lock-in detector calculates

X = (2/N) Σ V_i cos(ωt_i)

Y = (2/N) Σ V_i sin(ωt_i)

R = √(X² + Y²)

The SNR is estimated from repeated Monte Carlo measurements as

SNR = mean(R) / std(R)

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
