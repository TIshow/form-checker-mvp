"""core/audio.py: 合成信号で、有声/無声・F0・フォルマント帯の向きが取れることを確かめる。"""
import numpy as np

from core import audio


def _tone(sr, seconds, f0, boost_band=None):
    t = np.arange(int(sr * seconds)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    if boost_band:
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1, len(t))
        spec = np.fft.rfft(noise)
        fr = np.fft.rfftfreq(len(t), 1 / sr)
        spec[(fr < boost_band[0]) | (fr > boost_band[1])] = 0
        band = np.fft.irfft(spec, n=len(t))
        x = x + 0.5 * band / (np.std(band) + 1e-9)
    return x


def test_audio_f0_voicing_and_formant_direction():
    sr, fps = 22050, 30
    a = _tone(sr, 1.0, 220.0)                                   # 素の声
    b = _tone(sr, 1.0, 220.0, boost_band=audio.FORMANT_BAND)   # 3kHz 帯を足した声
    s = np.zeros(int(sr * 0.5))                                 # 無音
    x = np.concatenate([a, s, b])
    n = int(round(len(x) / sr * fps))
    r = audio.analyze_samples(x, sr, fps, n)
    v = r["voiced"]; f0 = r["series"]["f0_hz"]; form = r["series"]["formant_db"]
    fa, fs, fb = slice(2, 28), slice(32, 43), slice(47, 73)     # 端を避ける
    assert v[fa].all() and v[fb].all() and not v[fs].any()
    assert np.nanmedian(np.abs(f0[fa] - 220)) < 5 and np.nanmedian(np.abs(f0[fb] - 220)) < 5
    assert np.nanmean(form[fb]) > np.nanmean(form[fa]) + 3     # 帯を足した方が上に出る
    assert r["summary"]["voiced_fraction"] > 0.6
    assert r["summary"]["f0_mean_hz"] is not None and abs(r["summary"]["f0_mean_hz"] - 220) < 10


def test_audio_vibrato_detected_on_modulated_tone():
    sr, fps = 22050, 30
    t = np.arange(int(sr * 2.0)) / sr
    f_inst = 220 * 2 ** (0.5 * np.sin(2 * np.pi * 5.5 * t) / 12)   # ±50 cent, 5.5 Hz
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    x = sum(np.sin(k * phase) / k for k in range(1, 10))
    n = int(round(2.0 * fps))
    r = audio.analyze_samples(x, sr, fps, n)
    assert r["summary"]["vibrato_rate_hz"] is not None
    assert abs(r["summary"]["vibrato_rate_hz"] - 5.5) < 0.8
