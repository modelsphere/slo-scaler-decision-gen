"""PD pdengine simulation harness. NOT a pytest — a driver script.

Run:
    .venv/bin/python docs/pd_simulations.py
Or filtered:
    .venv/bin/python docs/pd_simulations.py traffic_both_comfy
    .venv/bin/python docs/pd_simulations.py --list

Every scenario stamps a fresh ServiceView against a PD Placement,
then replays a tick schedule (each entry: signals for that tick, or
a mutation to the placement/bounds) and prints one row per tick so
the eye can diff scenarios:

    tick  t  shape  phys   verdicts  gates  proposal  hr  wire
      0   0  2p1d  2p1d  t~ o~  r~   up~ shed~   -     -

Legend:
  shape/phys      prefill p decode d engines per slice
  verdicts        ttft / otps  x=VIOLATED ✓=COMFORTABLE ~=GREY
  gates           up = up-cooldown open   shed = shed-ready
  proposal        r1a-fire / r1b-steady / r1c-shed
  hr              hold reason ('-' means clean commit)
  wire            what /decisions would serve this tick
                  (`<omitted>` when the row was skipped)
  `← mut`         a DS spec or CR bound was mutated right before this tick

Keep scenarios short (3–6 ticks visible) and label each with the edge
it's meant to expose.
"""

import argparse
import sys
from dataclasses import dataclass

from decision_gen import pd, serviceview as sv
from decision_gen.k8s_state import PDSpec, Placement
from decision_gen.signals import Reading
from decision_gen.thresholds import Verdict


@dataclass
class Tick:
    ttft: float | None = None
    ttft_state: str = "ok"
    otps: float | None = None
    otps_state: str = "ok"
    rejection: float | None = 0.0
    rejection_count: float | None = 0
    request_count: float | None = 0
    physical_p: int | None = None
    physical_d: int | None = None
    physical_total: int = 1
    new_bounds: tuple | None = None
    new_live: tuple | None = None
    expect: str | None = None


PD_CR = {
    "serviceId": "svc",
    "priority": 5,
    "minimumDeployment": {"type": "pdengine", "value": 6},
    "maximumDeployment": {"type": "pdengine", "value": 24},
    "ttft": {"default": {"metrics": [{"type": "p80", "threshold": 30.0}]}},
    "otps": {"default": {"metrics": [{"type": "p80", "threshold": 400.0}]}},
}


def _placement(live_p=2, live_d=1):
    return Placement(
        namespace="ns", name="svc", kind="disaggregatedset",
        pool="p", gpus_per_replica=24, workload_name="svc",
        spec_replicas=1,
        pd=PDSpec(slices=1, prefill=live_p, decode=live_d,
                  role_group_gpus={"prefill": 8, "decode": 8}),
    )


def _cr(bounds):
    cr = dict(PD_CR)
    cr["minimumDeployment"] = {"type": "pdengine", "value": bounds[0]}
    cr["maximumDeployment"] = {"type": "pdengine", "value": bounds[1]}
    return cr


def _readings(tick):
    return {
        "ttft": {"p80": Reading(tick.ttft, tick.ttft_state)},
        "otps": {"p80": Reading(tick.otps, tick.otps_state)},
        "rejection": Reading(tick.rejection, "ok"),
        "rejection_count": Reading(tick.rejection_count, "ok"),
        "request_count": Reading(tick.request_count, "ok"),
        "replicas_ready_prefill": Reading(tick.physical_p, "ok")
            if tick.physical_p is not None else Reading(None, "missing_series"),
        "replicas_ready_decode": Reading(tick.physical_d, "ok")
            if tick.physical_d is not None else Reading(None, "missing_series"),
    }


def run(name, bounds=(6, 24), live=(2, 1), ticks=(), verbose=True):
    view = sv.ServiceView("ns", "svc")
    if verbose:
        print(f"\n── {name}  bounds={bounds}  live={live} "
              f"{'─' * max(0, 60 - len(name))}")
        print(_header())
    t_now = 0.0
    for i, tk in enumerate(ticks):
        if tk.new_bounds is not None:
            bounds = tk.new_bounds
        if tk.new_live is not None:
            live = tk.new_live
        cr = _cr(bounds)
        placement = _placement(*live)
        t = view.step(_readings(tk), placement, cr, tk.physical_total, t_now)
        if verbose:
            print(_row(i, t_now, tk, t, view))
        t_now += 60.0
    return view


def _header():
    return (f"{'tick':>4} {'t':>5} {'shape':>7} {'phys':>7} "
            f"{'verdicts':>10} {'gates':>12} {'proposal':>16} "
            f"{'hold':<28} {'wire':<20}")


def _v_mark(v):
    return {Verdict.VIOLATED: "x", Verdict.COMFORTABLE: "✓"}.get(v, "~")


def _row(i, t_now, tk, tr, view):
    shape = view.committed or "-"
    if shape != "-" and hasattr(shape, "prefill"):
        shape = f"{shape.prefill}p{shape.decode}d"
    phys = "-"
    if isinstance(tr.physical, tuple) and len(tr.physical) == 2:
        a, b = tr.physical
        if isinstance(a, int) and isinstance(b, int):
            phys = f"{a}p{b}d"
        else:
            phys = f"{a if isinstance(a, int) else '?'}p{b if isinstance(b, int) else '?'}d"
    elif isinstance(tr.physical, int):
        phys = str(tr.physical)
    verdicts = "-"
    if tr.verdicts:
        tv = tr.verdicts.get("ttft", {}).get("p80")
        ov = tr.verdicts.get("otps", {}).get("p80")
        verdicts = f"t{_v_mark(tv)} o{_v_mark(ov)}"
    gates = f"up{'✓' if tr.gates.up_cooldown_open else '~'} " \
            f"shed{'✓' if tr.gates.shed_ready else '~'}"
    prop = tr.proposal.rule if tr.proposal else "-"
    hold = tr.hold_reason or "-"
    if tr.pd_shape is None and tr.skip:
        wire = "<omitted>"
    elif tr.pd_shape is not None:
        wire = f"{tr.pd_shape.prefill}p{tr.pd_shape.decode}d"
        if tk.new_bounds or tk.new_live:
            wire += " ← mut"
    else:
        wire = "-"
    return (f"{i:>4} {int(t_now):>5} {shape:>7} {phys:>7} "
            f"{verdicts:>10} {gates:>12} {prop:>16} "
            f"{hold:<28.28} {wire:<20}")


UP = sv.UP_COOLDOWN_S          # 900 s default
DOWN = sv.DOWN_COOLDOWN_S      # 1800 s
COMF = sv.COMFORT_SUSTAIN_S    # 1200 s


@dataclass
class Sim:
    name: str
    bounds: tuple
    live: tuple
    ticks: tuple


def _repeat(tick, n):
    return tuple(tick for _ in range(n))


# Comfy canonical reading: ttft 5 (<< 30×0.5 comfort line), otps 900
# (> 400/0.5 = 800 strict-comfort line). Anything ≤ 800 is GREY, which
# blocks R1c — a subtle SLO-headroom detail the scenarios below rely on.
COMFY = Tick(ttft=5.0, otps=900.0, request_count=1000)


SIMS = [
    Sim("cold_boot",
        bounds=(6, 24), live=(2, 1),
        ticks=(COMFY, Tick(ttft=30.0, otps=900.0, request_count=100,
                            expect="seed=live; bounds-fix (2,1)→(4,2)"))),

    Sim("traffic_both_comfy_noop",
        bounds=(6, 24), live=(4, 2),
        ticks=_repeat(COMFY, 4)),

    Sim("ttft_violated_grows_prefill",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            *_repeat(COMFY, int(UP / 60)),
            Tick(ttft=50.0, otps=900.0, request_count=1000,
                 expect="ttft x; cooldown just opened"),
            Tick(ttft=50.0, otps=900.0, request_count=1000,
                 expect="grow prefill"),
        )),

    Sim("otps_violated_grows_decode",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            *_repeat(COMFY, int(UP / 60)),
            Tick(ttft=5.0, otps=350.0, request_count=1000,
                 expect="otps x"),
            Tick(ttft=5.0, otps=350.0, request_count=1000,
                 expect="grow decode"),
        )),

    Sim("both_violated_both_grow",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            *_repeat(COMFY, int(UP / 60)),
            Tick(ttft=50.0, otps=350.0, request_count=1000,
                 expect="ttft x + otps x; both grow"),
        )),

    Sim("max_pin_holds",
        bounds=(6, 24), live=(16, 8),
        ticks=(
            *_repeat(COMFY, int(UP / 60)),
            Tick(ttft=35.0, otps=300.0, request_count=1000,
                 expect="E=24 = max; clamp"),
        )),

    Sim("comfort_shed_bigger_margin",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            Tick(ttft=50.0, otps=900.0, request_count=1000,
                 expect="violation so streak starts clean"),
            *_repeat(COMFY, int((DOWN + COMF) / 60) + 1),
            Tick(ttft=5.0, otps=900.0, request_count=1000,
                 expect="shed from ttft-comfier side"),
        )),

    Sim("rejection_fire_scales_both",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(rejection=0.3, rejection_count=10,
                 request_count=1000, ttft=5.0, otps=900.0,
                 expect="R1a: both roles scale (mult capped 1.2)"),
        )),

    Sim("rejection_fire_needs_evidence_floor",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(rejection=0.3, rejection_count=3,
                 request_count=1000, ttft=5.0, otps=900.0,
                 expect="3 < 5 rejects → no fire"),
        )),

    Sim("missing_physical_role_falls_back",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            *_repeat(COMFY, int(UP / 60)),
            Tick(ttft=5.0, otps=350.0, request_count=1000,
                 expect="grow decode"),
            Tick(physical_d=None, ttft=5.0, otps=900.0, request_count=1000,
                 expect="decode bodylog quiet; phys falls back"),
        )),

    Sim("bounds_shrinks_below_live_compresses",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(new_bounds=(2, 2), ttft=5.0, otps=900.0,
                 request_count=1000,
                 expect="CR max=2 → compress to 1p1d"),
        )),

    Sim("external_live_shape_drift_adopted",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(new_live=(5, 2), ttft=5.0, otps=900.0,
                 request_count=1000,
                 expect="live 5p2d inside window → adopt live"),
        )),

    Sim("no_traffic_comfort_accumulates",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            Tick(ttft=50.0, otps=900.0, request_count=1000,
                 expect="violation; comfort reset"),
            *_repeat(
                Tick(ttft_state="no_traffic",
                     otps_state="no_traffic",
                     request_count=0),
                int((DOWN + COMF) / 60) + 1),
            Tick(ttft_state="no_traffic",
                 otps_state="no_traffic",
                 request_count=0,
                 expect="streak alive; no_traffic counts as COMFY"),
        )),

    Sim("missing_slo_holds_and_resets_comfort",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            *_repeat(
                Tick(ttft_state="missing_series",
                     otps=900.0, request_count=1000),
                3),
            COMFY,
        )),

    Sim("per_slice_window_inexpressible_skips",
        bounds=(7, 7), live=(2, 1),
        ticks=(
            COMFY,
        )),

    Sim("scale_to_zero",
        bounds=(0, 0), live=(2, 1),
        ticks=(
            COMFY,
        )),

    Sim("r1c_then_hold_down_cooldown_rearms",
        bounds=(6, 24), live=(12, 6),
        ticks=(
            COMFY,
            *_repeat(COMFY, int((DOWN + COMF) / 60)),
            COMFY,
            COMFY,
        )),

    Sim("bounds_floor_acting_as_scale_up",
        bounds=(6, 24), live=(2, 1),
        ticks=(
            COMFY,
            COMFY,
        )),

    Sim("shed_frozen_by_ramp_up",
        bounds=(6, 24), live=(2, 1),
        ticks=(
            COMFY,
            Tick(ttft=50.0, otps=900.0, request_count=1000,
                 expect="ttft x; UP-gated"),
            # Advance clocks so both gates open on the SAME tick
            # (usually impossible in prod, but a useful probe).
            *[Tick(ttft=5.0, otps=900.0, request_count=1000)
              for _ in range(int(max(UP, DOWN, COMF) / 60) + 1)],
            Tick(ttft=5.0, otps=900.0, request_count=1000,
                 expect="both gates open; shed wins over ttft"),
        )),

    Sim("bounds_shrink_below_live_is_progressive",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(new_bounds=(4, 4), ttft=5.0, otps=900.0,
                 request_count=1000,
                 expect="CR max=4 → live E=6 → correct to (3,1)"),
            Tick(ttft=50.0, otps=900.0, request_count=1000,
                 expect="ttft x but cooldown re-armed on bounds-fix"),
        )),

    Sim("bounds_lifts_after_compress_recovers",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(new_bounds=(2, 2), ttft=5.0, otps=900.0,
                 request_count=1000,
                 expect="compress to 1p1d"),
            Tick(new_bounds=(6, 24), ttft=5.0, otps=900.0,
                 request_count=1000,
                 expect="bounds lifted → adopt live (4,2) again"),
        )),

    Sim("physical_drift_lag_uses_committed",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(physical_p=4, physical_d=2,
                 ttft=5.0, otps=900.0, request_count=1000,
                 expect="phys (4,2) == committed → settled"),
            Tick(physical_p=4, physical_d=3,
                 ttft=5.0, otps=900.0, request_count=1000,
                 expect="decode bodylog reads stale 3 → IN_DRAIN for d"),
            Tick(physical_p=4, physical_d=2,
                 ttft=5.0, otps=900.0, request_count=1000,
                 expect="decode aligns"),
        )),

    Sim("role_tied_margins_alternate",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            Tick(ttft=50.0, otps=900.0, request_count=1000,
                 expect="reset streak"),
            *[Tick(ttft=5.0, otps=900.0, request_count=1000)
              for _ in range(int((DOWN + COMF) / 60))],
            Tick(ttft=5.0, otps=900.0, request_count=1000,
                 expect="shed 1: margin pick"),
            Tick(ttft=5.0, otps=900.0, request_count=1000,
                 expect="shed 2: cooldown held"),
        )),

    Sim("violation_needs_request_floor",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            *_repeat(COMFY, int(UP / 60)),
            Tick(ttft=50.0, otps=900.0, request_count=5,
                 expect="5 < 20 requests → R1b skipped"),
        )),

    Sim("rejection_clean_slo_uses_strict_cap",
        bounds=(6, 24), live=(4, 2),
        ticks=(
            COMFY,
            Tick(rejection=0.6, rejection_count=20,
                 request_count=1000, ttft=5.0, otps=900.0,
                 expect="r=0.6 → mult 4.0 but clean SLO → cap 1.2"),
        )),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", help="run only these")
    ap.add_argument("--list", action="store_true", help="list and exit")
    args = ap.parse_args()

    if args.list:
        for s in SIMS:
            print(f"  {s.name}")
        return

    names = args.scenarios or [s.name for s in SIMS]
    by_name = {s.name: s for s in SIMS}
    for name in names:
        sim = by_name.get(name)
        if sim is None:
            print(f"unknown scenario: {name}", file=sys.stderr)
            continue
        try:
            run(sim.name, sim.bounds, sim.live, sim.ticks)
        except Exception as e:
            print(f"\n── {name}  FAILED: {e}")


if __name__ == "__main__":
    main()
