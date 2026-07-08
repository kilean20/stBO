"""live_plots.py
==============
Two live popup plot windows for use alongside the stBO BOController loop.

  live_monitor_plot  — scrolling strip-chart per PV group, mirroring what
                     generate_phoebus_plt / Phoebus would display
  live_history_plot  — live version of bo.plot_history() (objective + timing)

Architecture
------------
Each class:
  1. Spawns a *subprocess* that owns a matplotlib figure and redraws it via
     FuncAnimation.
  2. Starts a *background thread* in the main process that polls data sources
     every `poll_interval_s` seconds and serialises new data into a Queue.
  3. The subprocess drains the Queue inside every animation frame.
  4. On stop(), a poison-pill freezes the animation but leaves the window open.

Data extraction — two sources, always 1-to-1 aligned
------------------------------------------------------
OracleEvaluator.__call__ returns {'x', 'y', 'x_set', 't_start', 't_end'}
and optional timing diagnostics to BOController; monitor PV readings live in
oracle.history rather than bo.history.

The full PV data lives in oracle.history["mean"] — a list of pd.Series
populated by EvaluatorBase._set_and_read every time the oracle is evaluated
with a non-None x.  Each Series contains the column-wise mean of the fetched
DataFrame: control_CSETs + control_RDs + monitor_PVs.

live_monitor_plot therefore reads:
  • Monitor PV values      — oracle.history["mean"][i][pv_name]
  • Control CSET values    — bo.history[i]["x_set"][j]   (what was commanded)
  • Control RD values      — oracle.history["mean"][i][rd_pv]  (what was read back)
  • Timestamps             — bo.history[i]["t_end"].timestamp()

Usage
-----
    from live_plots import live_monitor_plot, live_history_plot

    monitor_live = live_monitor_plot(
        bo,
        oracle,
        monitor_groups         = monitor_groups,
        control_CSETs_groups   = control_CSETs_groups,
        control_RDs_groups     = control_RDs_groups,   # parallel to CSETs groups
    )
    history_live = live_history_plot(bo)

    monitor_live.start()
    history_live.start()

    for _ in range(n_local):
        bo.step(mode="local", acq_type="qEI", fresh_train=True, plot_acq=True)
        plt.show()

    bo.finalize()
    monitor_live.stop()   # freezes popup (does NOT close it)
    history_live.stop()
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — module-level so multiprocessing 'spawn' can pickle them
# ─────────────────────────────────────────────────────────────────────────────

def _pick_backend() -> str:
    """Return the first interactive matplotlib backend available.

    Also clears MPLBACKEND from the environment so that Jupyter's inline
    backend setting (module://matplotlib_inline.backend_inline) does not
    override matplotlib.use() inside the subprocess / forked process.
    """
    import os, importlib
    os.environ.pop("MPLBACKEND", None)   # clear Jupyter's inline backend
    for backend in ("TkAgg", "Qt5Agg", "Qt6Agg", "wxAgg", "MacOSX"):
        try:
            importlib.import_module(
                {"TkAgg": "tkinter", "Qt5Agg": "PyQt5", "Qt6Agg": "PyQt6",
                 "wxAgg": "wx", "MacOSX": "AppKit"}.get(backend, backend)
            )
            return backend
        except ImportError:
            continue
    return "Agg"


def _series_get(series, pv_name: str) -> float:
    """Safely extract a float from a pandas Series by PV name."""
    try:
        val = series[pv_name]
        return float(np.squeeze(val))
    except (KeyError, TypeError, ValueError):
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess worker — live_monitor_plot
# ─────────────────────────────────────────────────────────────────────────────

def _monitor_worker(
    queue: mp.Queue,
    monitor_groups: List[List[str]],
    control_CSETs_groups: List[List[str]],
    control_RDs_groups: List[List[str]],   # parallel to control_CSETs_groups
    max_points: int,
    update_interval_ms: int,
    window_title: str,
) -> None:
    """
    Subprocess entry point for the Phoebus-style strip chart.

    Layout
    ------
    One subplot per monitor_group  (white background)
    One subplot per control group  (lavender background)
      └─ solid lines  : CSETs (what was commanded)
      └─ dashed lines : RDs   (what was read back), same colour per pair

    The window stays open after stop() — the animation simply freezes.
    """
    import matplotlib
    matplotlib.use(_pick_backend())
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from collections import deque

    # ── Build the flat list of all groups in display order ───────────────────
    # Each entry: (pv_list, label, is_control, rd_list_or_None)
    group_specs = []
    for i, g in enumerate(monitor_groups):
        suffixes = [p.split(":")[-1] for p in g]
        label = suffixes[0].replace("_", " ") if len(set(suffixes)) == 1 else f"Monitor {i+1}"
        group_specs.append((g, label, False, None))

    for i, (cset_g, rd_g) in enumerate(zip(control_CSETs_groups, control_RDs_groups)):
        suffixes = [p.split(":")[-1] for p in cset_g]
        label = suffixes[0].replace("_", " ") if len(set(suffixes)) == 1 else f"Control {i+1}"
        group_specs.append((cset_g, label, True, rd_g))

    n_groups = len(group_specs)

    # ── Data buffers ─────────────────────────────────────────────────────────
    all_pvs = set()
    for pvs, _, _, rds in group_specs:
        all_pvs.update(pvs)
        if rds:
            all_pvs.update(rds)

    ts_buf: deque = deque(maxlen=max_points)
    pv_buf: Dict[str, deque] = {pv: deque(maxlen=max_points) for pv in all_pvs}
    t0_ref: List[Optional[float]] = [None]
    stopped: List[bool] = [False]   # flag: stop animating but keep window open

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_h = max(2.2 * n_groups, 4.0)
    fig, axes = plt.subplots(n_groups, 1, figsize=(11, fig_h),
                             sharex=False, squeeze=False)
    axes = axes.flatten()

    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── Pre-create Line2D objects ─────────────────────────────────────────────
    pv_lines: Dict[str, Any] = {}

    for ax, (pvs, label, is_ctrl, rds) in zip(axes, group_specs):
        ax.set_title(label, fontsize=9, pad=3)
        ax.set_ylabel("Value", fontsize=8)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=7)
        if is_ctrl:
            ax.set_facecolor("#f5f0ff")

        for k, pv in enumerate(pvs):
            color = prop_cycle[k % len(prop_cycle)]
            device = pv.split(":")[0]
            suffix = pv.split(":")[-1]
            lbl = f"{device} • {suffix}" if len(pvs) > 1 else pv

            # Solid line for CSET (or plain monitor) — always in legend
            (line,) = ax.plot([], [], lw=1.8, color=color,
                              linestyle="-", label=lbl)
            pv_lines[pv] = line

            # Dashed line for the paired RD — same colour, NOT in legend
            if is_ctrl and rds is not None and k < len(rds):
                rd_pv = rds[k]
                (rd_line,) = ax.plot([], [], lw=1.4, color=color,
                                     linestyle="--", label="_nolegend_")
                pv_lines[rd_pv] = rd_line

        ncol = 2 if len(pvs) > 5 else 1
        ax.legend(fontsize=7, ncol=ncol,
                  loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, framealpha=0.7)

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.suptitle(window_title, fontsize=10, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    # ── Animation ─────────────────────────────────────────────────────────────
    def _drain_queue() -> None:
        """Drain all pending queue messages.  Sets stopped[0]=True on poison pill."""
        while True:
            try:
                msg = queue.get_nowait()
            except Exception:
                return
            if msg is None:
                stopped[0] = True   # freeze — do NOT close
                return
            for snap in msg:
                ts = snap["timestamp"]
                if t0_ref[0] is None:
                    t0_ref[0] = ts
                ts_buf.append(ts - t0_ref[0])
                for pv, val in snap["pv_values"].items():
                    if pv in pv_buf:
                        pv_buf[pv].append(val)

    def _animate(_frame):
        if stopped[0]:
            return   # window stays open, animation freezes

        _drain_queue()

        if len(ts_buf) == 0:
            return

        t_arr = np.asarray(ts_buf)

        for ax, (pvs, _, is_ctrl, rds) in zip(axes, group_specs):
            all_group_pvs = list(pvs) + (list(rds) if (is_ctrl and rds) else [])
            y_min, y_max = float("inf"), float("-inf")

            for pv in all_group_pvs:
                if pv not in pv_lines:
                    continue
                vals = np.asarray(pv_buf[pv])
                if len(vals) == 0:
                    continue
                n = min(len(t_arr), len(vals))
                displayed = vals[-n:]                       # the slice actually drawn
                pv_lines[pv].set_data(t_arr[-n:], displayed)
                finite = displayed[np.isfinite(displayed)]  # y-range from displayed only
                if len(finite):
                    y_min = min(y_min, finite.min())
                    y_max = max(y_max, finite.max())

            if len(t_arr) >= 2:
                ax.set_xlim(t_arr[0], t_arr[-1] + 1e-6)
            elif len(t_arr) == 1:
                ax.set_xlim(t_arr[0] - 1, t_arr[0] + 1)

            if np.isfinite(y_min) and np.isfinite(y_max):
                span = max(y_max - y_min, 1e-16)
                margin = span * 0.12
                ax.set_ylim(y_min - margin, y_max + margin)

    _ani = FuncAnimation(fig, _animate, interval=update_interval_ms,
                         cache_frame_data=False, blit=False)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess worker — live_history_plot
# ─────────────────────────────────────────────────────────────────────────────

def _history_worker(
    queue: mp.Queue,
    maximize: bool,
    update_interval_ms: int,
    window_title: str,
) -> None:
    """
    Subprocess entry point.
    Top panel  : objective + best-so-far.
    Bottom panel: timing breakdown.
    Window stays open after stop().
    """
    import matplotlib
    matplotlib.use(_pick_backend())
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 1.0, "grid.linewidth": 0.5, "lines.linewidth": 1.8,
    })

    fig, (ax_obj, ax_time) = plt.subplots(2, 1, figsize=(7.5, 6.5),
                                          sharex=False, constrained_layout=True)
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    (line_obj,)  = ax_obj.plot([], [], "o-", markersize=4, label="Objective", alpha=0.8)
    (line_best,) = ax_obj.plot([], [], lw=2.5, label="Best so far")
    scat_best    = ax_obj.scatter([], [], zorder=6, s=80, label="Best")
    ax_obj.set_ylabel("Objective")
    ax_obj.set_title("Objective over evaluations")
    ax_obj.grid(True, alpha=0.3)
    ax_obj.legend(loc="best", frameon=False)

    timing_spec = [("train","Train GP"),("search","Acq opt"),
                   ("oracle","Oracle"),("total","Total")]
    timing_lines: Dict[str, Any] = {}
    for key, lbl in timing_spec:
        (l,) = ax_time.plot([], [], "o-", markersize=3, label=lbl)
        timing_lines[key] = l
    ax_time.set_xlabel("Iteration")
    ax_time.set_ylabel("Time (s)")
    ax_time.set_title("Timing per iteration")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(loc="best", frameon=False)

    state: Dict[str, Any] = {"data": None}
    stopped: List[bool] = [False]

    def _drain_queue() -> None:
        while True:
            try:
                msg = queue.get_nowait()
            except Exception:
                return
            if msg is None:
                stopped[0] = True
                return
            state["data"] = msg

    def _animate(_frame):
        if stopped[0]:
            return   # freeze, don't close

        _drain_queue()
        d = state["data"]
        if d is None:
            return

        ys = np.asarray(d["ys"], dtype=float)
        if len(ys) == 0:
            return

        x = np.arange(1, len(ys) + 1)
        best = np.maximum.accumulate(ys) if maximize else np.minimum.accumulate(ys)
        best_idx = int(np.argmax(best) if maximize else np.argmin(best))

        line_obj.set_data(x, ys)
        line_best.set_data(x, best)
        scat_best.set_offsets([[x[best_idx], best[best_idx]]])
        scat_best.set_facecolors([line_best.get_color()])
        scat_best.set_label(f"Best @ iter {x[best_idx]}  ({best[best_idx]:.4g})")
        ax_obj.set_xlim(0.5, len(ys) + 0.5)
        finite = ys[np.isfinite(ys)]
        if len(finite) >= 2:
            span = finite.max() - finite.min()
            margin = max(span * 0.12, 1e-15)
            ax_obj.set_ylim(finite.min() - margin, finite.max() + margin)
        elif len(finite) == 1:
            v = finite[0]; ax_obj.set_ylim(v - 1, v + 1)
        ax_obj.legend(loc="best", frameon=False)

        timing = d["timing"]
        n_time = max((len(v) for v in timing.values()), default=0)
        if n_time == 0:
            return
        xt = np.arange(1, n_time + 1)
        t_max = 0.0
        for key, line in timing_lines.items():
            arr = np.asarray(timing.get(key, []), dtype=float)
            if len(arr) == 0:
                line.set_data([], []); continue
            arr = arr[:n_time]
            line.set_data(xt[:len(arr)], arr)
            finite_t = arr[np.isfinite(arr)]
            if len(finite_t):
                t_max = max(t_max, finite_t.max())
        ax_time.set_xlim(0.5, n_time + 0.5)
        if t_max > 0:
            ax_time.set_ylim(-0.05 * t_max, t_max * 1.18)

    _ani = FuncAnimation(fig, _animate, interval=update_interval_ms,
                         cache_frame_data=False, blit=False)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Public class — live_monitor_plot
# ─────────────────────────────────────────────────────────────────────────────

class live_monitor_plot:
    """
    Popup strip-chart mirroring Phoebus, reading from oracle.history["mean"].

    Parameters
    ----------
    bo : BOController
    oracle : OracleEvaluator
        Provides oracle.history["mean"] (list of pd.Series with all PV columns).
    monitor_groups : list[list[str]]
        Same structure as passed to generate_phoebus_plt.
    control_CSETs_groups : list[list[str]]
        Same structure as passed to generate_phoebus_plt.  Solid lines.
    control_RDs_groups : list[list[str]] | None
        Parallel to control_CSETs_groups.  Each inner list gives the RD PV
        name for the corresponding CSET at the same position.  Plotted as
        dashed lines of the same colour as the paired CSET.
        Example: if control_CSETs_groups = [['X1:I_CSET','X2:I_CSET']]
                 then control_RDs_groups  = [['X1:I_RD',  'X2:I_RD'  ]]
    max_points : int
        Rolling window length.
    update_interval_ms : int
        Redraw period of popup (ms).
    poll_interval_s : float
        Poll period of background thread (s).
    window_title : str
    """

    def __init__(
        self,
        bo,
        oracle,
        monitor_groups: List[List[str]],
        control_CSETs_groups: List[List[str]],
        control_RDs_groups: Optional[List[List[str]]] = None,
        *,
        max_points: int = 200,
        update_interval_ms: int = 1000,
        poll_interval_s: float = 0.5,
        window_title: str = "Live Monitor (Phoebus-style)",
    ) -> None:
        self.bo = bo
        self.oracle = oracle
        self.monitor_groups = monitor_groups
        self.control_CSETs_groups = control_CSETs_groups
        # If RDs not provided, use empty lists (no dashed lines drawn)
        if control_RDs_groups is None:
            control_RDs_groups = [[] for _ in control_CSETs_groups]
        self.control_RDs_groups = control_RDs_groups

        self._csets_flat: List[str] = [pv for g in control_CSETs_groups for pv in g]
        self._rds_flat:   List[str] = [pv for g in control_RDs_groups   for pv in g]

        self.max_points = max_points
        self.update_interval_ms = update_interval_ms
        self.poll_interval_s = poll_interval_s
        self.window_title = window_title

        self._queue: mp.Queue = mp.Queue()
        self._process: Optional[mp.Process] = None
        self._thread: Optional[threading.Thread] = None
        self._stop: threading.Event = threading.Event()
        self._last_len: int = 0

    def start(self) -> "live_monitor_plot":
        # "fork" on Linux/Mac: child inherits parent memory — worker functions
        # are already loaded, no re-import of __main__ needed (which would fail
        # in Jupyter where __main__ is the kernel built-in, not a file).
        # "spawn" on Windows: fork is not available there.
        import sys
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        self._process = ctx.Process(
            target=_monitor_worker,
            args=(
                self._queue,
                self.monitor_groups,
                self.control_CSETs_groups,
                self.control_RDs_groups,
                self.max_points,
                self.update_interval_ms,
                self.window_title,
            ),
            daemon=True,
        )
        self._process.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Freeze the popup (leaves window open) and stop the background thread."""
        self._stop.set()
        try:
            self._queue.put_nowait(None)   # poison pill → animation freezes
        except Exception:
            pass
        # Do NOT terminate the process — let the user close the window manually

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def _poll_loop(self) -> None:
        """
        Background thread.

        Source mapping
        --------------
        Monitor PVs    → oracle.history["mean"][i]  (pd.Series)
        Control CSETs  → bo.history[i]["x_set"]     (np.ndarray, positional)
        Control RDs    → oracle.history["mean"][i]  (pd.Series)
        Timestamps     → bo.history[i]["t_end"]
        """
        monitor_pvs = [pv for g in self.monitor_groups for pv in g]

        while not self._stop.is_set():
            bo_history   = self.bo.history
            oracle_means = self.oracle.history["mean"]   # list[pd.Series]

            current_len = min(len(bo_history), len(oracle_means))

            if current_len > self._last_len:
                snapshots = []

                for i in range(self._last_len, current_len):
                    series   = oracle_means[i]
                    bo_entry = bo_history[i]
                    pv_vals: Dict[str, float] = {}

                    # Monitor PVs — from oracle Series
                    for pv in monitor_pvs:
                        pv_vals[pv] = _series_get(series, pv)

                    # Control CSETs — positional from bo x_set
                    x_set = bo_entry.get("x_set")
                    for j, pv in enumerate(self._csets_flat):
                        if x_set is not None:
                            try:
                                pv_vals[pv] = float(np.squeeze(x_set)[j])
                                continue
                            except (IndexError, TypeError, ValueError):
                                pass
                        pv_vals[pv] = _series_get(series, pv)   # fallback

                    # Control RDs — from oracle Series (same source as monitors)
                    for pv in self._rds_flat:
                        pv_vals[pv] = _series_get(series, pv)

                    t_end = bo_entry.get("t_end")
                    ts = t_end.timestamp() if t_end is not None else time.time()

                    snapshots.append({"index": i + 1, "pv_values": pv_vals,
                                      "timestamp": ts})

                self._last_len = current_len
                if snapshots:
                    try:
                        self._queue.put_nowait(snapshots)
                    except Exception:
                        pass

            self._stop.wait(self.poll_interval_s)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Public class — live_history_plot
# ─────────────────────────────────────────────────────────────────────────────

class live_history_plot:
    """
    Popup showing a live version of bo.plot_history().

    Parameters
    ----------
    bo : BOController
    maximize : bool
    update_interval_ms : int
    poll_interval_s : float
    window_title : str
    """

    def __init__(
        self,
        bo,
        *,
        maximize: bool = True,
        update_interval_ms: int = 1000,
        poll_interval_s: float = 0.5,
        window_title: str = "Live BO History",
    ) -> None:
        self.bo = bo
        self.maximize = maximize
        self.update_interval_ms = update_interval_ms
        self.poll_interval_s = poll_interval_s
        self.window_title = window_title

        self._queue: mp.Queue = mp.Queue()
        self._process: Optional[mp.Process] = None
        self._thread: Optional[threading.Thread] = None
        self._stop: threading.Event = threading.Event()
        self._last_len: int = 0

    def start(self) -> "live_history_plot":
        import sys
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        self._process = ctx.Process(
            target=_history_worker,
            args=(self._queue, self.maximize,
                  self.update_interval_ms, self.window_title),
            daemon=True,
        )
        self._process.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Freeze the popup (leaves window open) and stop the background thread."""
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        # Do NOT terminate the process — window stays open for inspection

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            history = self.bo.history
            current_len = len(history)

            if current_len != self._last_len:
                self._last_len = current_len

                ys: List[float] = []
                for h in history:
                    y = h.get("y")
                    if y is None:
                        continue
                    try:
                        if hasattr(y, "detach"):
                            y = y.detach().cpu().item()
                        ys.append(float(y))
                    except (TypeError, ValueError):
                        pass

                timing_clean: Dict[str, List[float]] = {}
                for k, v in (self.bo.timing or {}).items():
                    timing_clean[k] = [
                        float(x) if x is not None else float("nan") for x in v
                    ]

                try:
                    self._queue.put_nowait({"ys": ys, "timing": timing_clean})
                except Exception:
                    pass

            self._stop.wait(self.poll_interval_s)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
