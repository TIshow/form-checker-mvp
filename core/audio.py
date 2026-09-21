"""動画の音声トラックから「響き」の代用量を、映像のフレームごとに出す（issue 013）。

## 何を測るか

| 量 | 何の代用か | 計算 |
|---|---|---|
| `rms_db` | 声量 | 短時間 RMS。**同一録音内の相対値**（最大を 0 dB） |
| `spr_db` | 響き（通る声） | Singing Power Ratio: 2〜4 kHz のピーク − 0〜2 kHz のピーク [dB] |
| `formant_db` | 響き | 2.8〜3.4 kHz 帯のエネルギー / 0〜8 kHz [dB]（シンガーズフォルマント帯） |
| `tilt_db` | 張り・明るさ | 1〜4 kHz / 0〜1 kHz [dB] |
| `f0_hz` | 音程 | 自己相関。有声区間のみ、それ以外は NaN |

全部 numpy。ffmpeg は動画から wav に落とすところだけ（imageio-ffmpeg 同梱）。

## 帯域と定義の出どころ（**未確認**。閾値は置かない）

- シンガーズフォルマントは 3 kHz 付近（訓練された男声で 2,800〜3,400 Hz とされる）。
  Sundberg 1974 が原典だが本文未確認。帯域は `FORMANT_BAND` で変えられる
- SPR は Omori et al. 1996 の定義（2〜4 kHz と 0〜2 kHz のピーク差）。本文未確認

だからここでは**良し悪しを判定しない**。同じ歌手・同じフレーズ・同じ録音条件の中で
「姿勢を変えたとき、どちらへ動いたか」を並べるところまで。絶対 dB は
マイク距離・会場・AAC 圧縮で変わるので比べない。

## 時間軸

映像のフレーム i の中心時刻 (i + 0.5) / fps を窓の中心にする。関節列と同じ長さの
配列を返すので、表示側はフレーム番号で引くだけでよい。スロー映像（撮影 fps と
再生 fps が違う）では音の時間軸が合わないので使わない。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

#: シンガーズフォルマント帯 [Hz]。出典未確認（docstring）。
FORMANT_BAND = (2800.0, 3400.0)
#: SPR の帯域 [Hz]
SPR_LOW, SPR_HIGH = (0.0, 2000.0), (2000.0, 4000.0)
#: 有声とみなす条件: 最大 RMS からの落ち込みと、自己相関ピークの高さ
VOICED_MIN_DB = -40.0
VOICED_MIN_AC = 0.5
#: F0 の探索範囲 [Hz]
F0_RANGE = (80.0, 1000.0)

#: 表示用の言葉（ドメインの metric_labels と同じ並び: ラベル, 単位, 桁, 補足）
SERIES_LABELS = {
    "rms_db": ("声量", "dB", 0, "同一録音内の相対値（最大 0）"),
    "spr_db": ("響き（SPR）", "dB", 1, "2〜4kHz ピーク − 0〜2kHz ピーク。大きいほど倍音が通る"),
    "formant_db": ("フォルマント帯", "dB", 1, "2.8〜3.4kHz / 全体"),
    "tilt_db": ("スペクトル傾斜", "dB", 1, "1〜4kHz / 0〜1kHz。大きいほど明るい"),
    "f0_hz": ("基本周波数", "Hz", 0, "有声区間のみ"),
}
SUMMARY_LABELS = {
    "voiced_fraction": ("有声区間の割合", "", 2, ""),
    "spr_mean_db": ("響き（SPR）平均", "dB", 1, "有声区間"),
    "formant_mean_db": ("フォルマント帯 平均", "dB", 1, "有声区間"),
    "tilt_mean_db": ("スペクトル傾斜 平均", "dB", 1, "有声区間"),
    "f0_mean_hz": ("基本周波数 平均", "Hz", 0, "有声区間"),
    "f0_jump_fraction": ("F0 の跳び", "", 2, "隣接フレームで3半音超の割合。大きいと伴奏・倍音の取り違え"),
    "f0_sd_semitones": ("音程の揺れ", "半音", 2, "有声区間の標準偏差"),
    "vibrato_rate_hz": ("ビブラート周期", "Hz", 1, "1秒以上続く有声区間の F0 変調"),
    "vibrato_extent_cents": ("ビブラート幅", "cent", 0, "±"),
}


def extract_samples(video: str | Path, sr: int = 22050, start: float = 0.0) -> np.ndarray:
    """動画の音声を mono float32 で取り出す。ffmpeg は imageio-ffmpeg の同梱バイナリ。"""
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-ss", f"{start:.3f}", "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _band(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def analyze_samples(x: np.ndarray, sr: int, fps: float, n_frames: int, win: int = 2048,
                    f0_range: tuple[float, float] = F0_RANGE) -> dict:
    """サンプル列から、映像フレームごとの系列と要約を返す。

    f0_range: F0 を探す範囲 [Hz]。**伴奏が入っている録音では歌手の声域に絞ること**
    （実測: ソプラノの 10 秒で、既定の 80〜1000 Hz だと伴奏の 84〜225 Hz を拾って
    F0 が跳び回った）。絞っても伴奏の影響は残るので、`f0_jump_fraction` を見る。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("mono のサンプル列を渡すこと")
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    b_form = _band(freqs, *FORMANT_BAND)
    b_all = _band(freqs, 0.0, min(8000.0, sr / 2))
    b_lo, b_hi = _band(freqs, *SPR_LOW), _band(freqs, *SPR_HIGH)
    b_t_lo, b_t_hi = _band(freqs, 0.0, 1000.0), _band(freqs, 1000.0, 4000.0)
    lag_lo, lag_hi = int(sr / f0_range[1]), int(sr / f0_range[0])

    rms = np.zeros(n_frames); spr = np.full(n_frames, np.nan); form = np.full(n_frames, np.nan)
    tilt = np.full(n_frames, np.nan); f0 = np.full(n_frames, np.nan); ac_peak = np.zeros(n_frames)
    eps = 1e-12
    for i in range(n_frames):
        c = int(round((i + 0.5) / fps * sr))
        lo = c - win // 2
        seg = np.zeros(win)
        a, b = max(lo, 0), min(lo + win, len(x))
        if b > a:
            seg[a - lo:b - lo] = x[a:b]
        rms[i] = np.sqrt(np.mean(seg ** 2))
        w = seg * hann
        S = np.fft.rfft(w)
        P = (np.abs(S) ** 2)
        if P[b_all].sum() > eps:
            spr[i] = 10 * np.log10(P[b_hi].max() + eps) - 10 * np.log10(P[b_lo].max() + eps)
            form[i] = 10 * np.log10((P[b_form].sum() + eps) / (P[b_all].sum() + eps))
            tilt[i] = 10 * np.log10((P[b_t_hi].sum() + eps) / (P[b_t_lo].sum() + eps))
        # 自己相関（パワースペクトルの逆 FFT）で F0。1.5 kHz より上は落としてから
        # 取る。フォルマント帯（3 kHz 付近）が強い声だと、その帯の周期性が
        # 自己相関の短い遅れに出て F0 を上に釣り上げる。F0 は低い倍音で決まる
        ac = np.fft.irfft(P * (freqs < 1500.0), n=win)
        if ac[0] > eps:
            ac = ac / ac[0]
            seg_ac = ac[lag_lo:lag_hi]
            # 自己相関は周期の整数倍にもピークが立つ。最大値を取ると 1 オクターブ〜
            # 1 オクターブ半下（周期の 2〜3 倍）を拾う（実測: ソプラノで 84/111 Hz が
            # 頻発）。最大の 85% 以上あるピークのうち**最も短い遅れ**を採る
            top = float(seg_ac.max())
            if top <= 0:            # 探索範囲に周期性が無い（無声）
                continue
            cand = np.flatnonzero(seg_ac >= 0.85 * top)
            k = int(cand[0])
            # 隣が高ければそちらへ寄せる（離散ピークの頂点）
            while k + 1 < len(seg_ac) and seg_ac[k + 1] > seg_ac[k]:
                k += 1
            ac_peak[i] = float(seg_ac[k])
            f0[i] = sr / (lag_lo + k)
    rms_db = 20 * np.log10(rms / (rms.max() + eps) + eps)
    voiced = (rms_db > VOICED_MIN_DB) & (ac_peak > VOICED_MIN_AC)
    f0[~voiced] = np.nan
    # 5 フレームの中央値で単発の誤りを落とす（有声区間の中だけ）
    f0m = f0.copy()
    for i in range(len(f0)):
        seg = f0[max(0, i - 2):i + 3]
        seg = seg[np.isfinite(seg)]
        if np.isfinite(f0[i]) and len(seg) >= 3:
            f0m[i] = np.median(seg)
    f0 = f0m

    def smooth3(v):
        out = v.copy()
        for i in range(1, len(v) - 1):
            seg = v[i - 1:i + 2]
            if np.isfinite(seg).all():
                out[i] = seg.mean()
        return out

    spr, form, tilt = smooth3(spr), smooth3(form), smooth3(tilt)

    def vmean(v):
        m = voiced & np.isfinite(v)
        return float(v[m].mean()) if m.any() else None

    summary = {
        "voiced_fraction": float(voiced.mean()),
        "spr_mean_db": vmean(spr), "formant_mean_db": vmean(form), "tilt_mean_db": vmean(tilt),
        "f0_mean_hz": vmean(f0),
    }
    fv = f0[voiced & np.isfinite(f0)]
    summary["f0_sd_semitones"] = float(np.std(12 * np.log2(fv / fv.mean()))) if len(fv) > 1 else None
    # F0 の信頼度: 隣接フレームで 3 半音を超えて跳ぶ割合。歌声は滑らかに動くので、
    # これが大きいときは伴奏や倍音の取り違えが混ざっている
    jumps = np.abs(12 * np.log2(f0[1:] / f0[:-1]))
    ok = np.isfinite(jumps)
    summary["f0_jump_fraction"] = float((jumps[ok] > 3).mean()) if ok.any() else None
    summary.update(_vibrato(f0, voiced, fps))
    return {
        "sr": sr, "fps": fps, "n_frames": n_frames, "window": win,
        "formant_band_hz": list(FORMANT_BAND), "f0_range_hz": list(f0_range),
        "series": {"rms_db": rms_db, "spr_db": spr, "formant_db": form, "tilt_db": tilt, "f0_hz": f0},
        "voiced": voiced,
        "summary": summary,
        "note": "同一録音内の相対値。帯域の出典は未確認（core/audio.py）。良し悪しは判定しない",
    }


def _vibrato(f0: np.ndarray, voiced: np.ndarray, fps: float, min_s: float = 1.0) -> dict:
    """1 秒以上続く有声区間ごとに F0 の 4〜8 Hz 変調を見る。無ければ None。"""
    rates, extents = [], []
    i = 0
    n = len(f0)
    while i < n:
        if not (voiced[i] and np.isfinite(f0[i])):
            i += 1; continue
        j = i
        # 3 半音を超える跳びで区間を切る（伴奏や倍音誤りをビブラートに数えない）
        while j < n and voiced[j] and np.isfinite(f0[j]) and (j == i or abs(12 * np.log2(f0[j] / f0[j - 1])) <= 3):
            j += 1
        if (j - i) / fps >= min_s and fps >= 16:
            cents = 1200 * np.log2(f0[i:j] / np.mean(f0[i:j]))
            cents = cents - np.mean(cents)
            spec = np.abs(np.fft.rfft(cents * np.hanning(len(cents))))
            fr = np.fft.rfftfreq(len(cents), 1.0 / fps)
            band = (fr >= 4.0) & (fr <= 8.0)
            if band.any() and spec[band].max() > 0:
                k = np.flatnonzero(band)[int(np.argmax(spec[band]))]
                rates.append(float(fr[k]))
                extents.append(float((cents.max() - cents.min()) / 2))
        i = j
    return {"vibrato_rate_hz": float(np.mean(rates)) if rates else None,
            "vibrato_extent_cents": float(np.mean(extents)) if extents else None}


def analyze(video: str | Path, fps: float, n_frames: int, start: float = 0.0, sr: int = 22050,
            f0_range: tuple[float, float] = F0_RANGE) -> dict:
    """動画から直接。`start` は動画の何秒目を関節列のフレーム 0 にするか。"""
    return analyze_samples(extract_samples(video, sr, start), sr, fps, n_frames, f0_range=f0_range)
