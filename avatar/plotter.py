from __future__ import annotations
from typing import Dict, List
import time

def plotter_process(q, labels: List[str], seconds: int, hz: float, start_evt, stop_evt):
    # Import Matplotlib *inside* the child; choose backend BEFORE pyplot import
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import numpy as _np
        from collections import deque as _deque
    except ImportError as e:
        print(f"Plotter disabled: missing dependency {e}")
        if start_evt:
            start_evt.set()
        return

    labels = list(sorted(set(labels)))
    max_pts = int(seconds * hz)

    t = _deque(maxlen=max_pts)
    val = _deque(maxlen=max_pts)
    aro = _deque(maxlen=max_pts)
    lat_pos = _deque(maxlen=max_pts)
    lat_emo = _deque(maxlen=max_pts)
    lat_asr = _deque(maxlen=max_pts)
    lat_rtc = _deque(maxlen=max_pts)
    nx = _deque(maxlen=max_pts)
    ny = _deque(maxlen=max_pts)
    prob_hist: Dict[str, _deque] = {lb: _deque(maxlen=max_pts) for lb in labels}
    txt: List[str] = []

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3)
    ax_v = fig.add_subplot(gs[0, 0]); ax_v.set_title("Valence"); ax_v.set_ylim(-1, 1)
    ax_a = fig.add_subplot(gs[0, 1]); ax_a.set_title("Arousal"); ax_a.set_ylim(-1, 1)
    ax_lat = fig.add_subplot(gs[0, 2]); ax_lat.set_title("Latencies (ms)")
    ax_prob = fig.add_subplot(gs[1, :2]); ax_prob.set_title("Emotion probabilities"); ax_prob.set_ylim(0.0, 1.0)
    ax_txt = fig.add_subplot(gs[1, 2]); ax_txt.set_title("Transcription History"); ax_txt.axis('off')
    ax_fx = fig.add_subplot(gs[2, 0]); ax_fx.set_title("Face X (norm)"); ax_fx.set_ylim(-1, 1)
    ax_fy = fig.add_subplot(gs[2, 1]); ax_fy.set_title("Face Y (norm)"); ax_fy.set_ylim(-1, 1)

    (l_v,) = ax_v.plot([], [])
    (l_a,) = ax_a.plot([], [])
    (l_lp,) = ax_lat.plot([], [], label='pos')
    (l_le,) = ax_lat.plot([], [], label='emo')
    (l_la,) = ax_lat.plot([], [], label='asr')
    (l_lr,) = ax_lat.plot([], [], label='rtc')
    ax_lat.legend(loc='upper left')

    l_probs = {lb: ax_prob.plot([], [], label=lb)[0] for lb in labels}
    ax_prob.legend(loc='upper left', ncol=2)

    (l_fx,) = ax_fx.plot([], [])
    (l_fy,) = ax_fy.plot([], [])

    va_fallback_flags = _deque(maxlen=max_pts)

    def drain_queue():
        drained = 0
        while True:
            try:
                m = q.get_nowait()
            except Exception:
                break
            drained += 1
            tt = m.get("ts", time.time())
            t.append(tt)
            val.append(float(m.get("valence", 0.0)))
            aro.append(float(m.get("arousal", 0.0)))
            lat_pos.append(float(m.get("lat_pos", 0.0)))
            lat_emo.append(float(m.get("lat_emo", 0.0)))
            lat_asr.append(float(m.get("lat_asr", 0.0)))
            lat_rtc.append(float(m.get("lat_rtc", 0.0)))
            nx.append(float(m.get("nx", 0.0)))
            ny.append(float(m.get("ny", 0.0)))
            probs = m.get("probs", {})
            va_fallback_flags.append(bool(m.get("va_fallback", False)))
            for lb in labels:
                prob_hist[lb].append(float(probs.get(lb, 0.0)))
            text = m.get("text", "")
            if text:
                if txt and text.startswith(txt[-1]):
                    txt[-1] = text
                elif not txt or text != txt[-1]:
                    txt.append(text)
        return drained

    def update(_):
        if stop_evt.is_set():
            plt.close(fig)
            return
        drained = drain_queue()
        if drained == 0:
            return
        xs = _np.arange(len(t))
        l_v.set_data(xs, list(val)); ax_v.set_xlim(max(0, len(xs)-max_pts), len(xs)+1)
        l_a.set_data(xs, list(aro)); ax_a.set_xlim(max(0, len(xs)-max_pts), len(xs)+1)
        l_lp.set_data(xs, list(lat_pos)); l_le.set_data(xs, list(lat_emo)); l_la.set_data(xs, list(lat_asr)); l_lr.set_data(xs, list(lat_rtc))
        ax_lat.relim(); ax_lat.autoscale_view(); ax_lat.set_xlim(max(0, len(xs)-max_pts), len(xs)+1)
        for lb, line in l_probs.items():
            line.set_data(xs, list(prob_hist[lb]))
        ax_prob.set_xlim(max(0, len(xs)-max_pts), len(xs)+1)
        l_fx.set_data(xs, list(nx)); ax_fx.set_xlim(max(0, len(xs)-max_pts), len(xs)+1)
        l_fy.set_data(xs, list(ny)); ax_fy.set_xlim(max(0, len(xs)-max_pts), len(xs)+1)
        # Show fallback indicator in title if active in any of the last points
        try:
            if any(va_fallback_flags):
                ax_v.set_title("Valence [fallback]")
                ax_a.set_title("Arousal [fallback]")
            else:
                ax_v.set_title("Valence")
                ax_a.set_title("Arousal")
        except Exception:
            pass
        ax_txt.clear(); ax_txt.axis('off'); ax_txt.text(0.0, 1.0, " ".join(txt), va='top', fontsize=9, wrap=True)

    anim = FuncAnimation(fig, update, interval=int(1000.0 / max(1.0, hz)), blit=False, cache_frame_data=False, repeat=True)
    _anim_ref = anim  # keep alive
    start_evt.set()

    def on_close(_evt):
        stop_evt.set()
    fig.canvas.mpl_connect("close_event", on_close)
    try:
        plt.tight_layout()
        plt.show()
    except KeyboardInterrupt:
        pass
