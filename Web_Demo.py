import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='Sine vs Square Lock-in SNR', layout='wide')


# ============================================================
# Basic functions
# ============================================================

def square_wave(phase):
    return np.where(np.sin(phase) >= 0, 1.0, -1.0)


def make_modulations(t, f0, mode):
    phase = 2 * np.pi * f0 * t

    if mode == 'Same peak & average optical power':
        # Both signals go from 0 to 1 and have average = 0.5
        p_sine = 0.5 * (1 + np.sin(phase))
        p_square = 0.5 * (1 + square_wave(phase))

        expected_label = 'Expected ideal advantage: square / sine = 4/pi'
        expected_ratio = 4 / np.pi

    else:
        # Same RMS AC modulation
        dc = 1.5
        p_square = dc + square_wave(phase)
        p_sine = dc + np.sqrt(2) * np.sin(phase)

        expected_label = 'Expected ideal advantage: sine / square = pi/(2sqrt(2))'
        expected_ratio = np.pi / (2 * np.sqrt(2))

    expected_db = 20 * np.log10(expected_ratio)

    return (
        p_sine,
        p_square,
        expected_label,
        expected_ratio,
        expected_db
    )


def one_over_f_noise(n, fs, rng):
    freqs = np.fft.rfftfreq(n, d=1/fs)

    spectrum = (
        rng.normal(size=len(freqs))
        + 1j * rng.normal(size=len(freqs))
    )

    shaping = np.zeros_like(freqs)
    shaping[1:] = 1 / np.sqrt(freqs[1:])

    spectrum *= shaping
    spectrum[0] = 0

    x = np.fft.irfft(spectrum, n=n)

    x -= np.mean(x)

    s = np.std(x)

    return x / s if s > 0 else x


def make_noise(n, fs, white_sigma, pink_ratio, rng):

    white = rng.normal(0, white_sigma, n)

    pink = np.zeros(n)

    if pink_ratio > 0 and white_sigma > 0:
        pink = (
            one_over_f_noise(n, fs, rng)
            * white_sigma
            * pink_ratio
        )

    return white + pink


def rc_lowpass(x, fc, fs):

    dt = 1 / fs
    tau = 1 / (2 * np.pi * fc)

    alpha = dt / (tau + dt)

    y = np.empty_like(x, dtype=float)

    y[0] = x[0]

    for i in range(1, len(x)):
        y[i] = y[i-1] + alpha * (x[i] - y[i-1])

    return y


def lockin_xy(v, t, f0):

    w = 2 * np.pi * f0

    X = (2 / len(v)) * np.sum(v * np.cos(w * t))
    Y = (2 / len(v)) * np.sum(v * np.sin(w * t))

    R = np.sqrt(X**2 + Y**2)

    return X, Y, R


def fft_amplitude(x, fs):

    x = x - np.mean(x)

    n = len(x)

    amp = 2 * np.abs(np.fft.rfft(x)) / n

    f = np.fft.rfftfreq(n, d=1/fs)

    amp[0] = 0

    return f, amp


def snr_from_r(r):

    sd = np.std(r, ddof=1)

    return np.mean(r) / sd if sd > 0 else np.inf


# ============================================================
# Streamlit interface
# ============================================================

st.title('Pure Sine vs Square-Wave Chopper: Lock-in SNR Comparison')

st.sidebar.header('Simulation parameters')


mode = st.sidebar.selectbox(
    'Comparison normalization',
    [
        'Same peak & average optical power',
        'Same RMS AC modulation'
    ]
)


f0 = st.sidebar.slider(
    'Modulation frequency f0 (Hz)',
    10,
    500,
    100,
    10
)


T_req = st.sidebar.slider(
    'Lock-in integration time (s)',
    0.10,
    2.00,
    0.50,
    0.05
)


fc = st.sidebar.slider(
    'Detector low-pass bandwidth (Hz)',
    20,
    2000,
    500,
    20
)


white_sigma = st.sidebar.slider(
    'White-noise sigma (signal units)',
    0.0,
    2.0,
    0.30,
    0.05
)


pink_ratio = st.sidebar.slider(
    '1/f noise amplitude / white-noise sigma',
    0.0,
    3.0,
    0.0,
    0.1
)


mc_trials = st.sidebar.slider(
    'Monte Carlo trials',
    50,
    1000,
    300,
    50
)


seed = st.sidebar.number_input(
    'Random seed',
    min_value=0,
    value=1234,
    step=1
)


# ============================================================
# Time grid
# ============================================================

# Snap to an integer number of modulation cycles
# to avoid FFT leakage

n_cycles = max(
    1,
    int(round(T_req * f0))
)

T = n_cycles / f0


fs = max(
    50 * f0,
    10 * fc,
    5000
)


n = int(round(T * fs))


if n > 120000:
    n = 120000
    fs = n / T


t = np.arange(n) / fs


# ============================================================
# Signals
# ============================================================

(
    p_sine,
    p_square,
    expected_label,
    expected_ratio,
    expected_db
) = make_modulations(
    t,
    f0,
    mode
)


# ============================================================
# Display noise
# ============================================================

# Same noise for both time traces
# so they can be directly compared

rng_display = np.random.default_rng(seed)

shared_noise = make_noise(
    n,
    fs,
    white_sigma,
    pink_ratio,
    rng_display
)


v_sine = rc_lowpass(
    p_sine + shared_noise,
    fc,
    fs
)


v_square = rc_lowpass(
    p_square + shared_noise,
    fc,
    fs
)


# ============================================================
# Monte Carlo
# ============================================================

# Now save X, Y, AND R

x_sine = np.empty(mc_trials)
y_sine = np.empty(mc_trials)
r_sine = np.empty(mc_trials)

x_square = np.empty(mc_trials)
y_square = np.empty(mc_trials)
r_square = np.empty(mc_trials)


for k in range(mc_trials):

    rng = np.random.default_rng(
        seed + 1000 + k
    )

    noise = make_noise(
        n,
        fs,
        white_sigma,
        pink_ratio,
        rng
    )


    vs = rc_lowpass(
        p_sine + noise,
        fc,
        fs
    )


    vq = rc_lowpass(
        p_square + noise,
        fc,
        fs
    )


    (
        x_sine[k],
        y_sine[k],
        r_sine[k]
    ) = lockin_xy(
        vs,
        t,
        f0
    )


    (
        x_square[k],
        y_square[k],
        r_square[k]
    ) = lockin_xy(
        vq,
        t,
        f0
    )


# ============================================================
# SNR
# ============================================================

snr_sine = snr_from_r(r_sine)
snr_square = snr_from_r(r_square)


# ============================================================
# Summary
# ============================================================

st.subheader('Summary')

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    'SNR — Pure sine',
    f'{snr_sine:.2f}'
)


c2.metric(
    'SNR — Square chopper',
    f'{snr_square:.2f}'
)


c3.metric(
    'SNR sine / square',
    f'{snr_sine / snr_square:.3f}'
)


c4.metric(
    'SNR square / sine',
    f'{snr_square / snr_sine:.3f}'
)


st.caption(
    f'Actual integration time = {T:.4f} s '
    f'({n_cycles} complete cycles); '
    f'sample rate = {fs:.0f} Hz; '
    f'N = {n:,}.'
)


st.info(
    f'{expected_label} = '
    f'{expected_ratio:.4f}, '
    f'or {expected_db:+.2f} dB '
    f'under the ideal normalization assumption.'
)


# ============================================================
# FIGURE 1 — Time domain
# ============================================================

st.subheader(
    '1. Time domain: modulation + identical noise'
)


cycles_to_show = min(
    5,
    n_cycles
)


mask = (
    t <= cycles_to_show / f0
)


fig1, ax1 = plt.subplots(
    figsize=(10, 4)
)


ax1.plot(
    t[mask] * 1e3,
    v_sine[mask],
    label='Pure sine + noise',
    lw=1.5
)


ax1.plot(
    t[mask] * 1e3,
    v_square[mask],
    label='Square chopper + same noise',
    lw=1.2,
    alpha=0.85
)


ax1.set_xlabel(
    'Time (ms)'
)


ax1.set_ylabel(
    'Detected signal (a.u.)'
)


ax1.set_title(
    'After detector low-pass'
)


ax1.grid(
    alpha=0.25
)


ax1.legend()


st.pyplot(fig1)


# ============================================================
# FIGURE 2 — FFT
# ============================================================

st.subheader(
    '2. FFT: noiseless modulation spectrum'
)


f_s, a_s = fft_amplitude(
    p_sine,
    fs
)


f_q, a_q = fft_amplitude(
    p_square,
    fs
)


fmax = min(
    9 * f0,
    fs / 2
)


maskf = (
    f_s <= fmax
)


fig2, ax2 = plt.subplots(
    figsize=(10, 4)
)


ax2.plot(
    f_s[maskf],
    a_s[maskf],
    label='Pure sine',
    lw=1.8
)


ax2.plot(
    f_q[maskf],
    a_q[maskf],
    label='Square chopper',
    lw=1.4
)


# Include 9f0
for h in [1, 3, 5, 7, 9]:

    if h * f0 <= fmax:

        ax2.axvline(
            h * f0,
            ls=':' if h > 1 else '--',
            lw=1,
            alpha=0.55
        )

        ymax = max(
            a_q[maskf].max(),
            a_s[maskf].max()
        )

        ax2.text(
            h * f0,
            ymax * 0.90,
            f'{h}f0',
            rotation=90,
            va='top',
            ha='right'
        )


ax2.set_xlabel(
    'Frequency (Hz)'
)


ax2.set_ylabel(
    'One-sided amplitude (a.u.)'
)


ax2.set_title(
    'Square wave shows odd harmonics; '
    'sine is concentrated at f0'
)


ax2.grid(
    alpha=0.25
)


ax2.legend()


st.pyplot(fig2)


# ============================================================
# FIGURE 3 — X/Y scatter + R histogram
# ============================================================

st.subheader(
    '3. Digital lock-in Monte Carlo'
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# FIGURE 3A — X/Y lock-in plane
# ------------------------------------------------------------

with col1:

    st.markdown(
        '**3A. Lock-in X/Y plane**'
    )

    fig3a, ax3a = plt.subplots(
        figsize=(6, 6)
    )


    ax3a.scatter(
        x_sine,
        y_sine,
        alpha=0.35,
        s=18,
        label='Pure sine'
    )


    ax3a.scatter(
        x_square,
        y_square,
        alpha=0.35,
        s=18,
        label='Square chopper'
    )


    # Mean positions
    mean_x_sine = np.mean(x_sine)
    mean_y_sine = np.mean(y_sine)

    mean_x_square = np.mean(x_square)
    mean_y_square = np.mean(y_square)


    ax3a.scatter(
        mean_x_sine,
        mean_y_sine,
        s=180,
        marker='x',
        linewidths=3,
        label='Mean — sine'
    )


    ax3a.scatter(
        mean_x_square,
        mean_y_square,
        s=180,
        marker='x',
        linewidths=3,
        label='Mean — square'
    )


    # Origin
    ax3a.axhline(
        0,
        lw=1,
        alpha=0.4
    )


    ax3a.axvline(
        0,
        lw=1,
        alpha=0.4
    )


    ax3a.set_xlabel(
        'Lock-in X'
    )


    ax3a.set_ylabel(
        'Lock-in Y'
    )


    ax3a.set_title(
        'Monte Carlo points in the X/Y plane'
    )


    ax3a.grid(
        alpha=0.25
    )


    ax3a.legend(
        fontsize=9
    )


    st.pyplot(fig3a)


# ------------------------------------------------------------
# FIGURE 3B — R histogram
# ------------------------------------------------------------

with col2:

    st.markdown(
        '**3B. Recovered magnitude R**'
    )

    fig3b, ax3b = plt.subplots(
        figsize=(6, 6)
    )


    ax3b.hist(
        r_sine,
        bins=35,
        alpha=0.60,
        label=(
            f'Pure sine\n'
            f'SNR = {snr_sine:.2f}'
        )
    )


    ax3b.hist(
        r_square,
        bins=35,
        alpha=0.60,
        label=(
            f'Square\n'
            f'SNR = {snr_square:.2f}'
        )
    )


    mean_r_sine = np.mean(r_sine)
    mean_r_square = np.mean(r_square)

    std_r_sine = np.std(
        r_sine,
        ddof=1
    )

    std_r_square = np.std(
        r_square,
        ddof=1
    )


    ax3b.axvline(
        mean_r_sine,
        lw=2,
        label='Mean R — sine'
    )


    ax3b.axvline(
        mean_r_square,
        lw=2,
        label='Mean R — square'
    )


    ax3b.set_xlabel(
        'Recovered lock-in magnitude R'
    )


    ax3b.set_ylabel(
        'Monte Carlo count'
    )


    ax3b.set_title(
        'Distribution of recovered R'
    )


    ax3b.grid(
        alpha=0.25
    )


    ax3b.legend(
        fontsize=9
    )


    st.pyplot(fig3b)


# ============================================================
# Numerical Monte Carlo results
# ============================================================

st.markdown(
    '### Monte Carlo numerical results'
)


m1, m2 = st.columns(2)


with m1:

    st.markdown('**Pure sine**')

    st.write(
        f'Mean X = {mean_x_sine:.4f}'
    )

    st.write(
        f'Mean Y = {mean_y_sine:.4f}'
    )

    st.write(
        f'Mean R = {mean_r_sine:.4f}'
    )

    st.write(
        f'Std(R) = {std_r_sine:.4f}'
    )

    st.write(
        f'SNR = Mean(R) / Std(R) '
        f'= {snr_sine:.2f}'
    )


with m2:

    st.markdown('**Square chopper**')

    st.write(
        f'Mean X = {mean_x_square:.4f}'
    )

    st.write(
        f'Mean Y = {mean_y_square:.4f}'
    )

    st.write(
        f'Mean R = {mean_r_square:.4f}'
    )

    st.write(
        f'Std(R) = {std_r_square:.4f}'
    )

    st.write(
        f'SNR = Mean(R) / Std(R) '
        f'= {snr_square:.2f}'
    )


# ============================================================
# Lock-in equations
# ============================================================

st.subheader(
    'Digital lock-in equations'
)


st.latex(
    r'X=\frac{2}{N}\sum_i V_i\cos(\omega t_i)'
)


st.latex(
    r'Y=\frac{2}{N}\sum_i V_i\sin(\omega t_i)'
)


st.latex(
    r'R=\sqrt{X^2+Y^2}'
)


st.latex(
    r'\mathrm{SNR}'
    r'=\frac{\mathrm{mean}(R)}{\mathrm{std}(R)}'
)


st.warning(
    'SNR here is estimated from repeated Monte Carlo '
    'measurements as mean(R) / std(R). '
    'At very low SNR, R has a positive Rice-distribution bias.'
)


# ============================================================
# Interpretation
# ============================================================

st.subheader(
    'Interpretation'
)


if mode == 'Same peak & average optical power':

    st.markdown(r'''
With both waveforms constrained from 0 to the same peak optical power,
they also have the same average optical power.

The fundamental amplitudes are

- sine: \(0.5\)
- 50% square: \(2/\pi \approx 0.637\)

Therefore,

\[
\frac{A_{\rm square}(f_0)}
{A_{\rm sine}(f_0)}
=
\frac{4}{\pi}
\approx 1.273
\]

so the square-wave modulation provides about 27.3% more signal
at the lock-in reference frequency.

If the noise PSD and lock-in bandwidth are the same,
the ideal SNR advantage is therefore approximately

\[
20\log_{10}(4/\pi)
\approx +2.10\ {\rm dB}.
\]
''')


else:

    st.markdown(r'''
When the zero-mean AC components are normalized to have the same
RMS modulation, the sine wave places all of its AC modulation power
at \(f_0\).

The square wave distributes some of its modulation power into

\[
3f_0,\;5f_0,\;7f_0,\ldots
\]

A lock-in detecting only \(f_0\) therefore favors the sine wave.

The ideal ratio is

\[
\frac{A_{\rm sine}(f_0)}
{A_{\rm square}(f_0)}
=
\frac{\pi}{2\sqrt{2}}
\approx 1.111
\]

corresponding to approximately

\[
+0.91\ {\rm dB}
\]

in favor of the sine wave.
''')


st.caption(
    'Simplified linear model: does not yet include shot noise, '
    'laser RIN, chopper vibration, detector saturation, '
    'or nonlinear modulation transfer.'
)