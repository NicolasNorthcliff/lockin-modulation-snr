import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sine vs Square Lock-in SNR", layout="wide")


# ============================================================
# Signal and noise functions
# ============================================================

def square_wave(phase):
    return np.where(np.sin(phase) >= 0.0, 1.0, -1.0)


def make_modulations(t, f0, mode):
    """Return physically explicit sine and square optical-power waveforms."""
    phase = 2.0 * np.pi * f0 * t

    if mode == "Equal peak and average optical power (0 to 1)":
        # Both range from 0 to 1, average 0.5, modulation depth 100%.
        p_sine = 0.5 * (1.0 + np.sin(phase))
        p_square = 0.5 * (1.0 + square_wave(phase))
        expected_square_over_sine = 4.0 / np.pi
        interpretation = "equal_optical_limits"

    elif mode == "Equal fundamental amplitude at f0":
        # Both have a fundamental Fourier amplitude of 1.
        # A unit square has fundamental amplitude 4/pi, so scale by pi/4.
        dc = 1.5
        p_sine = dc + np.sin(phase)
        p_square = dc + (np.pi / 4.0) * square_wave(phase)
        expected_square_over_sine = 1.0
        interpretation = "equal_fundamental"

    elif mode == "Equal RMS AC modulation":
        # Both zero-mean AC components have RMS = 1.
        dc = 1.5
        p_sine = dc + np.sqrt(2.0) * np.sin(phase)
        p_square = dc + square_wave(phase)
        expected_square_over_sine = 2.0 * np.sqrt(2.0) / np.pi
        interpretation = "equal_rms"

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return (
        p_sine,
        p_square,
        expected_square_over_sine,
        interpretation,
    )


def one_over_f_noise(n, fs, rng):
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = rng.normal(size=len(freqs)) + 1j * rng.normal(size=len(freqs))
    shaping = np.zeros_like(freqs)
    shaping[1:] = 1.0 / np.sqrt(freqs[1:])
    spectrum *= shaping
    spectrum[0] = 0.0
    x = np.fft.irfft(spectrum, n=n)
    x -= np.mean(x)
    sd = np.std(x)
    return x / sd if sd > 0 else x


def make_noise(n, fs, white_sigma, pink_ratio, rng):
    white = rng.normal(0.0, white_sigma, n)
    pink = np.zeros(n)
    if pink_ratio > 0 and white_sigma > 0:
        pink = one_over_f_noise(n, fs, rng) * white_sigma * pink_ratio
    return white + pink


def rc_lowpass(x, fc, fs):
    dt = 1.0 / fs
    tau = 1.0 / (2.0 * np.pi * fc)
    alpha = dt / (tau + dt)
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
    return y


def lockin_xy(v, t, f0):
    phase = 2.0 * np.pi * f0 * t
    x = (2.0 / len(v)) * np.sum(v * np.cos(phase))
    y = (2.0 / len(v)) * np.sum(v * np.sin(phase))
    r = np.hypot(x, y)
    return x, y, r


def phase_aligned_channel(x, y):
    """Project X/Y results onto their mean signal direction."""
    phi = np.arctan2(np.mean(y), np.mean(x))
    signal = x * np.cos(phi) + y * np.sin(phi)
    quadrature = -x * np.sin(phi) + y * np.cos(phi)
    return signal, quadrature, phi


def signed_snr(samples):
    sd = np.std(samples, ddof=1)
    return np.mean(samples) / sd if sd > 0 else np.inf


def fft_amplitude(x, fs):
    x = x - np.mean(x)
    n = len(x)
    amp = 2.0 * np.abs(np.fft.rfft(x)) / n
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amp[0] = 0.0
    return freq, amp


def fundamental_amplitude(x, t, f0):
    """Calculate the amplitude of the Fourier component at f0."""
    phase = 2.0 * np.pi * f0 * t
    x_ac = x - np.mean(x)
    a_cos = (2.0 / len(x_ac)) * np.sum(x_ac * np.cos(phase))
    a_sin = (2.0 / len(x_ac)) * np.sum(x_ac * np.sin(phase))
    return np.hypot(a_cos, a_sin)


# ============================================================
# Streamlit controls
# ============================================================

st.title("Pure Sine vs Square-Wave Chopper: Lock-in SNR Comparison")
st.sidebar.header("Simulation parameters")

mode = st.sidebar.selectbox(
    "Comparison normalization",
    [
        "Equal peak and average optical power (0 to 1)",
        "Equal fundamental amplitude at f0",
        "Equal RMS AC modulation",
    ],
)

f0 = st.sidebar.slider("Modulation frequency f0 (Hz)", 10, 500, 100, 10)
T_req = st.sidebar.slider("Lock-in integration time (s)", 0.10, 2.00, 0.50, 0.05)
fc = st.sidebar.slider("Detector low-pass bandwidth (Hz)", 20, 2000, 500, 20)
white_sigma = st.sidebar.slider("White-noise sigma (signal units)", 0.0, 2.0, 0.30, 0.05)
pink_ratio = st.sidebar.slider("1/f noise amplitude / white-noise sigma", 0.0, 3.0, 0.0, 0.1)
mc_trials = st.sidebar.slider("Monte Carlo trials", 50, 1000, 300, 50)
seed = st.sidebar.number_input("Random seed", min_value=0, value=1234, step=1)


# ============================================================
# Time grid: integer number of cycles prevents spectral leakage
# ============================================================

n_cycles = max(1, int(round(T_req * f0)))
T = n_cycles / f0
fs = max(50 * f0, 10 * fc, 5000)
n = int(round(T * fs))

if n > 120000:
    n = 120000
    fs = n / T

t = np.arange(n) / fs


# ============================================================
# Signals and theoretical fundamental amplitudes
# ============================================================

p_sine, p_square, expected_ratio, interpretation = make_modulations(t, f0, mode)

a1_sine = fundamental_amplitude(p_sine, t, f0)
a1_square = fundamental_amplitude(p_square, t, f0)
a1_ratio = a1_square / a1_sine
expected_db = 20.0 * np.log10(expected_ratio)


# ============================================================
# One displayed realization with identical additive noise
# ============================================================

rng_display = np.random.default_rng(seed)
shared_noise = make_noise(n, fs, white_sigma, pink_ratio, rng_display)
v_sine = rc_lowpass(p_sine + shared_noise, fc, fs)
v_square = rc_lowpass(p_square + shared_noise, fc, fs)


# ============================================================
# Monte Carlo lock-in measurements
# ============================================================

x_sine = np.empty(mc_trials)
y_sine = np.empty(mc_trials)
r_sine = np.empty(mc_trials)
x_square = np.empty(mc_trials)
y_square = np.empty(mc_trials)
r_square = np.empty(mc_trials)

for k in range(mc_trials):
    # The same realization is used for the two waveforms in each paired trial.
    rng = np.random.default_rng(seed + 1000 + k)
    noise = make_noise(n, fs, white_sigma, pink_ratio, rng)

    vs = rc_lowpass(p_sine + noise, fc, fs)
    vq = rc_lowpass(p_square + noise, fc, fs)

    x_sine[k], y_sine[k], r_sine[k] = lockin_xy(vs, t, f0)
    x_square[k], y_square[k], r_square[k] = lockin_xy(vq, t, f0)


# Phase-align each waveform so the primary result is a single lock-in channel.
s_sine, q_sine, phi_sine = phase_aligned_channel(x_sine, y_sine)
s_square, q_square, phi_square = phase_aligned_channel(x_square, y_square)

snr_sine = signed_snr(s_sine)
snr_square = signed_snr(s_square)
snr_ratio = snr_square / snr_sine


# ============================================================
# Summary
# ============================================================

st.subheader("Summary: phase-aligned single-channel lock-in")

c1, c2, c3, c4 = st.columns(4)
c1.metric("SNR — Pure sine", f"{snr_sine:.2f}")
c2.metric("SNR — Square chopper", f"{snr_square:.2f}")
c3.metric("Measured square / sine", f"{snr_ratio:.3f}")
c4.metric("Ideal square / sine", f"{expected_ratio:.3f}")

st.caption(
    f"Actual integration time = {T:.4f} s ({n_cycles} complete cycles); "
    f"sample rate = {fs:.0f} Hz; N = {n:,}."
)

st.info(
    f"At f0: sine amplitude = {a1_sine:.4f}, square amplitude = {a1_square:.4f}, "
    f"square/sine = {a1_ratio:.4f}. The ideal SNR ratio is {expected_ratio:.4f} "
    f"({expected_db:+.2f} dB) when the noise PSD and lock-in bandwidth at f0 are equal."
)

st.markdown(
    r"""
The phrase **equal amplitude** is not unique. It may refer to equal peak
modulation, equal RMS modulation, or equal fundamental amplitude at the
lock-in reference frequency. The selected normalization determines the
expected SNR comparison.
"""
)


# ============================================================
# Figure 1: time domain
# ============================================================

st.subheader("1. Time domain: modulation + identical noise")
cycles_to_show = min(5, n_cycles)
mask = t <= cycles_to_show / f0

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(t[mask] * 1e3, v_sine[mask], label="Pure sine + noise", lw=1.5)
ax1.plot(t[mask] * 1e3, v_square[mask], label="Square chopper + same noise", lw=1.2, alpha=0.85)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Detected signal (a.u.)")
ax1.set_title("After detector low-pass")
ax1.grid(alpha=0.25)
ax1.legend()
st.pyplot(fig1)
plt.close(fig1)


# ============================================================
# Figure 2: FFT
# ============================================================

st.subheader("2. FFT: noiseless modulation spectrum")
f_s, a_s = fft_amplitude(p_sine, fs)
f_q, a_q = fft_amplitude(p_square, fs)
fmax = min(9 * f0, fs / 2)
maskf = f_s <= fmax

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(f_s[maskf], a_s[maskf], label="Pure sine", lw=1.8)
ax2.plot(f_q[maskf], a_q[maskf], label="Square chopper", lw=1.4)

ymax = max(a_q[maskf].max(), a_s[maskf].max())
for harmonic in [1, 3, 5, 7, 9]:
    if harmonic * f0 <= fmax:
        ax2.axvline(harmonic * f0, ls=":" if harmonic > 1 else "--", lw=1, alpha=0.55)
        ax2.text(harmonic * f0, ymax * 0.90, f"{harmonic}f0", rotation=90, va="top", ha="right")

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("One-sided amplitude (a.u.)")
ax2.set_title("Square wave contains odd harmonics; sine is concentrated at f0")
ax2.grid(alpha=0.25)
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)


# ============================================================
# Figure 3: Monte Carlo
# ============================================================

st.subheader("3. Digital lock-in Monte Carlo")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**3A. Lock-in X/Y plane**")
    fig3a, ax3a = plt.subplots(figsize=(6, 6))
    ax3a.scatter(x_sine, y_sine, alpha=0.35, s=18, label="Pure sine")
    ax3a.scatter(x_square, y_square, alpha=0.35, s=18, label="Square chopper")
    ax3a.scatter(np.mean(x_sine), np.mean(y_sine), s=180, marker="x", linewidths=3, label="Mean — sine")
    ax3a.scatter(np.mean(x_square), np.mean(y_square), s=180, marker="x", linewidths=3, label="Mean — square")
    ax3a.axhline(0, lw=1, alpha=0.4)
    ax3a.axvline(0, lw=1, alpha=0.4)
    ax3a.set_xlabel("Lock-in X")
    ax3a.set_ylabel("Lock-in Y")
    ax3a.set_title("Monte Carlo points in the X/Y plane")
    ax3a.grid(alpha=0.25)
    ax3a.legend(fontsize=9)
    st.pyplot(fig3a)
    plt.close(fig3a)

with col2:
    st.markdown("**3B. Phase-aligned single-channel signal**")
    fig3b, ax3b = plt.subplots(figsize=(6, 6))
    ax3b.hist(s_sine, bins=35, alpha=0.60, label=f"Pure sine\nSNR = {snr_sine:.2f}")
    ax3b.hist(s_square, bins=35, alpha=0.60, label=f"Square\nSNR = {snr_square:.2f}")
    ax3b.axvline(np.mean(s_sine), lw=2, label="Mean S — sine")
    ax3b.axvline(np.mean(s_square), lw=2, label="Mean S — square")
    ax3b.set_xlabel("Recovered phase-aligned signal S")
    ax3b.set_ylabel("Monte Carlo count")
    ax3b.set_title("Single-channel distribution after phase alignment")
    ax3b.grid(alpha=0.25)
    ax3b.legend(fontsize=9)
    st.pyplot(fig3b)
    plt.close(fig3b)


# ============================================================
# Numerical results and equations
# ============================================================

st.markdown("### Monte Carlo numerical results")
m1, m2 = st.columns(2)

with m1:
    st.markdown("**Pure sine**")
    st.write(f"Reference phase = {np.degrees(phi_sine):.2f} deg")
    st.write(f"Mean S = {np.mean(s_sine):.4f}")
    st.write(f"Std(S) = {np.std(s_sine, ddof=1):.4f}")
    st.write(f"Mean quadrature Q = {np.mean(q_sine):.4e}")
    st.write(f"SNR = Mean(S) / Std(S) = {snr_sine:.2f}")

with m2:
    st.markdown("**Square chopper**")
    st.write(f"Reference phase = {np.degrees(phi_square):.2f} deg")
    st.write(f"Mean S = {np.mean(s_square):.4f}")
    st.write(f"Std(S) = {np.std(s_square, ddof=1):.4f}")
    st.write(f"Mean quadrature Q = {np.mean(q_square):.4e}")
    st.write(f"SNR = Mean(S) / Std(S) = {snr_square:.2f}")

with st.expander("Optional magnitude R diagnostic"):
    st.write(
        f"Sine: Mean(R) = {np.mean(r_sine):.4f}, Std(R) = {np.std(r_sine, ddof=1):.4f}"
    )
    st.write(
        f"Square: Mean(R) = {np.mean(r_square):.4f}, Std(R) = {np.std(r_square, ddof=1):.4f}"
    )
    st.warning(
        "R = sqrt(X^2 + Y^2) is useful for phase-independent visualization, "
        "but it has a positive Rice-distribution bias at low SNR. It is not used "
        "as the primary single-channel SNR metric here."
    )

st.subheader("Digital lock-in equations")
st.latex(r"X=\frac{2}{N}\sum_i V_i\cos(\omega t_i)")
st.latex(r"Y=\frac{2}{N}\sum_i V_i\sin(\omega t_i)")
st.latex(r"\phi=\operatorname{atan2}(\langle Y\rangle,\langle X\rangle)")
st.latex(r"S=X\cos\phi+Y\sin\phi")
st.latex(r"\mathrm{SNR}=\frac{\operatorname{mean}(S)}{\operatorname{std}(S)}")


# ============================================================
# Mode-specific interpretation
# ============================================================

st.subheader("Interpretation")

if interpretation == "equal_optical_limits":
    st.markdown(r"""
Both waveforms range from 0 to 1 and have the same average optical power,
0.5. Their modulation depths and peak optical powers are therefore also equal.

The fundamental amplitudes are

\[
A_{1,\mathrm{sine}}=\frac{1}{2},\qquad
A_{1,\mathrm{square}}=\frac{2}{\pi}.
\]

Thus,

\[
\frac{A_{1,\mathrm{square}}}{A_{1,\mathrm{sine}}}
=\frac{4}{\pi}\approx1.273.
\]

With the same noise PSD and lock-in bandwidth at \(f_0\), the ideal square-wave
SNR is about 27.3% higher, or \(+2.10\ \mathrm{dB}\). This advantage comes from
the larger fundamental component under the stated optical constraints, not
from the higher harmonics being detected by a single-frequency lock-in.
""")

elif interpretation == "equal_fundamental":
    st.markdown(r"""
The square wave is scaled by \(\pi/4\), so its fundamental component equals the
unit-amplitude sine component:

\[
A_{1,\mathrm{sine}}=A_{1,\mathrm{square}}=1.
\]

A sinusoidal single-frequency lock-in detects only the component at \(f_0\).
Therefore, if both signals experience the same noise PSD and measurement
bandwidth at \(f_0\), their ideal SNRs are equal. The square wave still contains
odd harmonics, but those harmonics do not improve this single-frequency result.
""")

else:
    st.markdown(r"""
The zero-mean AC components have the same RMS value. The sine wave places all
of its AC modulation power at \(f_0\), whereas the square wave distributes some
of its power among \(3f_0,5f_0,7f_0,\ldots\).

The fundamental-amplitude ratio is

\[
\frac{A_{1,\mathrm{square}}}{A_{1,\mathrm{sine}}}
=\frac{2\sqrt{2}}{\pi}\approx0.900.
\]

Consequently, a single-frequency lock-in ideally favors the sine wave by

\[
\frac{A_{1,\mathrm{sine}}}{A_{1,\mathrm{square}}}
=\frac{\pi}{2\sqrt{2}}\approx1.111,
\]

or approximately \(+0.91\ \mathrm{dB}\) in favor of the sine wave.
""")

st.caption(
    "Simplified linear model: additive white and 1/f noise are included. "
    "Shot noise, laser RIN proportional to optical power, chopper vibration, "
    "detector saturation, and nonlinear modulation transfer are not included."
)
