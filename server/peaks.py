"""Chromatographic peak detection and integration on one trace (XIC, SRM transition, TIC).

Used by the explorer (click-to-integrate), targeted quantification, and gap filling in untargeted
analysis, so a peak area means the same thing everywhere in the app.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def smooth(y: np.ndarray, window: int = 7) -> np.ndarray:
    """Savitzky–Golay smoothing (keeps peak height and width better than a moving average)."""
    y = np.asarray(y, float)
    w = min(window, len(y) - (1 - len(y) % 2))
    if w < 5:
        return y.copy()
    if w % 2 == 0:
        w -= 1
    return np.clip(savgol_filter(y, w, 2), 0, None)


def baseline(y: np.ndarray, quantile: float = 0.1, window: int = 101) -> np.ndarray:
    """Rolling low-quantile baseline — follows slow drift without cutting into peaks narrower than the window."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 5:
        return np.zeros(n)
    # the window must be several peak widths wide or it follows the peak itself; a third of the trace is
    # wide enough for any LC peak in a full-run XIC and still tracks gradient drift
    w = min(max(window, n // 3), n)
    half = w // 2
    pad = np.pad(y, half, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    base = np.quantile(sliding_window_view(pad, w), quantile, axis=1)[:n]
    return smooth(base, 21) if n > 25 else base


def noise_level(y: np.ndarray) -> float:
    """Robust noise estimate: median absolute deviation of the first difference (insensitive to peaks)."""
    d = np.diff(np.asarray(y, float))
    if not len(d):
        return 0.0
    mad = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    if mad > 0:
        return mad
    # Most neighbouring points are identical: a zero-filled trace (high-resolution XIC, thresholded SRM). The SD
    # of the differences is then dominated by the peaks themselves — one tall peak made every small peak in the
    # same trace fall below min_snr. Use the lower quartile of the non-zero intensities (≈ the instrument's
    # reporting threshold / the noise spikes) when that is smaller.
    cands = [float(np.std(d))]
    pos = np.asarray(y, float)
    pos = pos[pos > 0]
    if len(pos) >= 4:
        cands.append(float(np.percentile(pos, 25)))
    cands = [c for c in cands if c > 0]
    return min(cands) if cands else 1.0


def integrate(t: np.ndarray, y: np.ndarray, rt: float | None = None, window: float | None = None,
              smooth_window: int = 5, min_snr: float = 3.0, subtract_baseline: bool = True) -> dict | None:
    """Find and integrate one chromatographic peak.

    If rt is given, the peak whose apex is closest to rt (within ±window s) is chosen; otherwise the
    tallest. Boundaries run from the apex down to where the smoothed trace reaches the baseline (or 1%
    of the apex, or a valley). Area is the trapezoid integral above a straight baseline drawn between
    the two boundaries — the convention most vendor software uses ("drop to baseline").
    Returns None when nothing rises above min_snr × noise.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 3 or not np.any(y > 0):
        return None
    ys = smooth(y, smooth_window)
    base = baseline(ys) if subtract_baseline and len(ys) > 30 else np.zeros_like(ys)
    yc = np.clip(ys - base, 0, None)
    nz = noise_level(y)
    if rt is not None and window:
        cand = np.flatnonzero((t >= rt - window) & (t <= rt + window))
        if not len(cand):
            return None
    else:
        cand = np.arange(len(t))
    pk, _ = find_peaks(np.concatenate([[0], yc, [0]]))
    pk = pk - 1
    pk = pk[np.isin(pk, cand)]
    if not len(pk):
        if not len(cand):
            return None
        pk = np.array([cand[np.argmax(yc[cand])]])
    if rt is not None:
        # nearest tall-enough peak: among peaks > 30% of the tallest candidate, the closest to the expected RT
        tall = pk[yc[pk] >= 0.3 * yc[pk].max()]
        apex = int(tall[np.argmin(np.abs(t[tall] - rt))])
    else:
        apex = int(pk[np.argmax(yc[pk])])
    h = yc[apex]
    snr = h / nz if nz > 0 else np.inf
    if h <= 0 or snr < min_snr:
        return None
    stop = max(0.01 * h, nz)
    lo = _walk(yc, apex, -1, stop, h, nz)
    hi = _walk(yc, apex, +1, stop, h, nz)
    seg_t = t[lo:hi + 1]
    seg_y = y[lo:hi + 1]
    if subtract_baseline:
        # Straight line between the trace's level at the two boundaries. Where the peak came back down to
        # baseline that level is the local mean of the raw trace just outside the boundary; where it stopped
        # in a valley between co-eluting peaks it is the rolling baseline under the valley ("drop to
        # baseline"). (This used to be the minimum of the raw segment for both ends, which on a noisy
        # baseline sits ~2 SD below the true level and inflates every area by 2 SD × peak width.)
        def _level(i, outward):
            if yc[i] > stop:
                return float(base[i])
            j0, j1 = (max(i - 2, 0), i + 1) if outward < 0 else (i, min(i + 3, len(y)))
            return float(np.mean(y[j0:j1]))
        b0, b1 = _level(lo, -1), _level(hi, +1)
        line = np.interp(seg_t, [t[lo], t[hi]], [b0, b1])
    else:
        b0 = b1 = 0.0
        line = np.zeros_like(seg_t)
    area = float(np.trapezoid(np.clip(seg_y - line, 0, None), seg_t)) if len(seg_t) > 1 else float(h)
    # full width at half maximum from the smoothed, baseline-corrected trace, interpolated between the
    # scans either side of half height (without interpolation it is overestimated by up to one scan interval)
    half = h / 2
    l = apex
    while l > lo and yc[l] > half:
        l -= 1
    r = apex
    while r < hi and yc[r] > half:
        r += 1
    if r > l:
        tl = t[l] + (half - yc[l]) / (yc[l + 1] - yc[l]) * (t[l + 1] - t[l]) if yc[l] < half < yc[l + 1] else t[l]
        tr = t[r] - (half - yc[r]) / (yc[r - 1] - yc[r]) * (t[r] - t[r - 1]) if yc[r] < half < yc[r - 1] else t[r]
        fwhm = float(tr - tl)
    else:
        fwhm = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return {
        "rt": float(t[apex]), "height": float(y[apex]), "area": area, "start": float(t[lo]), "end": float(t[hi]),
        "fwhm": fwhm, "snr": float(min(snr, 1e6)), "points": int(hi - lo + 1),
        "i_start": int(lo), "i_end": int(hi), "i_apex": apex,
        "base_start": float(b0), "base_end": float(b1),   # the straight baseline under the area (for plots)
    }


def _walk(yc, apex, step, stop, h, nz):
    """From the apex outwards until the trace falls to `stop`, or rises again by more than noise after
    dropping below half height (a valley between co-eluting peaks). Returns the lowest point passed."""
    i = apex
    best, best_i = yc[apex], apex
    rise = max(2 * nz, 0.05 * h)
    while 0 <= i + step < len(yc) and yc[i + step] > stop:
        if best < 0.5 * h and yc[i + step] > best + rise:
            return best_i
        i += step
        if yc[i] <= best:
            best, best_i = yc[i], i
    return i + step if 0 <= i + step < len(yc) else i


def detect_all(t: np.ndarray, y: np.ndarray, min_snr: float = 5.0, min_width_s: float = 2.0, smooth_window: int = 7):
    """Every peak in a trace above min_snr × noise (for TIC peak lists and quick overviews)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 5:
        return []
    ys = smooth(y, smooth_window)
    base = baseline(ys)
    yc = np.clip(ys - base, 0, None)
    nz = noise_level(y)
    dt = float(np.median(np.diff(t))) or 1.0
    pk, prop = find_peaks(yc, height=min_snr * nz, width=max(1, min_width_s / dt / 2), prominence=min_snr * nz)
    out = []
    for p in pk:
        r = integrate(t, y, rt=float(t[p]), window=dt * 1.5, smooth_window=smooth_window, min_snr=min_snr)
        if r:
            out.append(r)
    return out
