"""Build tracks/<map>/ from its occupancy map, then the evaluation layouts."""
from __future__ import annotations

import argparse
import signal
from typing import Any

import yaml

from evaluation.layouts.generate import generate_set
from ppo.layouts import LAYOUTS
from sim.track import compile_map
from trackgen.paths import TrackPaths
from trackgen.stages import centreline, mintime, raceline, zones

STAGES = ("centreline", "raceline", "mintime", "zones", "compile", "layouts")


def _stage_index(name: str) -> int:
    """Return the index of a stage name."""
    if name not in STAGES:
        raise SystemExit(f"unknown stage {name!r}; choose from {', '.join(STAGES)}")
    return STAGES.index(name)


def _stored(paths: TrackPaths) -> dict[str, Any]:
    """Return the track.yaml mapping already on disk, or an empty one."""
    if paths.track_yaml.exists():
        doc = yaml.safe_load(paths.track_yaml.read_text())
        if isinstance(doc, dict):
            return doc
    return {}


def _write_track_yaml(paths: TrackPaths, opt_time: float, vmax: float | None) -> None:
    """Write track.yaml: the minimum-time lap and the speed cap it was solved with."""
    text = f"opt_time: {float(f'{opt_time:.2f}')}\n"
    if vmax is not None:
        text += f"max_speed_mps: {float(vmax)}\n"
    paths.track_yaml.write_text(text)
    print(f"wrote {paths.track_yaml}")
    print(text.rstrip())


def run(
    map_name: str,
    direction: str,
    plot: bool,
    from_stage: str,
    zone_spec: zones.ZoneSpec = 2,
    vmax: float | None = None,
) -> None:
    """Run the track pipeline from from_stage onwards."""
    paths = TrackPaths(map_name)
    start = _stage_index(from_stage)
    stored = _stored(paths)
    stored_vmax = None if stored.get("max_speed_mps") is None else float(stored["max_speed_mps"])
    if vmax is None:
        vmax = stored_vmax
    elif start > _stage_index("mintime") and vmax != stored_vmax:
        raise SystemExit("--vmax changes opt_time; restart from the mintime stage")
    print("trackgen")
    for key, value in {"map": paths.yaml_path, "direction": direction, "zones": zone_spec,
                       "vmax": vmax, "plot": plot, "from": from_stage}.items():
        print(f"  {key}: {value}")

    if start <= _stage_index("centreline"):
        print("\n[centreline]")
        centreline.run(paths, direction, plot)
    if start <= _stage_index("raceline"):
        print("\n[raceline]")
        raceline.run(paths, direction, plot)
    if start <= _stage_index("mintime"):
        print("\n[mintime]")
        opt_time = mintime.run(paths, plot, vmax)
    elif "opt_time" in stored:
        opt_time = float(stored["opt_time"])
    else:
        raise SystemExit(f"missing opt_time in {paths.track_yaml}; restart from the mintime stage")
    if start <= _stage_index("zones"):
        print("\n[zones]")
        zones.run(paths, plot, zone_spec)

    print("\n[track.yaml]")
    _write_track_yaml(paths, opt_time, vmax)
    if start <= _stage_index("compile"):
        print("\n[compile]")
        print(f"wrote {compile_map(map_name)}")

    print("\n[layouts]")
    layouts_dir = LAYOUTS / map_name
    if layouts_dir.exists():
        print(f"kept {layouts_dir}")
    else:
        print(f"wrote {generate_set(map_name)}")


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the pipeline."""
    # A GUI plotting backend swallows Ctrl-C; keep the default handler so it aborts.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    parser = argparse.ArgumentParser(prog="./track", description="Build tracks/<map>/ from its occupancy map.")
    parser.add_argument("map", help="map name under tracks/ (reads tracks/<map>/<map>.yaml and its image)")
    parser.add_argument("--direction", choices=("cw", "ccw"), required=True,
                        help="travel direction of the car around the track")
    parser.add_argument("--zones", default="2", metavar="N|a:b,c:d",
                        help="number of obstacle zones to set with the dials, or explicit station ranges in metres")
    parser.add_argument("--vmax", type=float, default=None, metavar="MPS",
                        help="speed cap for the minimum-time lap, written to track.yaml "
                             "(default: the cap stored in track.yaml, else the vehicle's max_speed_mps)")
    parser.add_argument("--from", dest="from_stage", default="centreline", metavar="STAGE",
                        help="restart at this stage: " + ", ".join(STAGES))
    parser.add_argument("--no-plot", action="store_true", help="skip plots and Enter prompts")
    args = parser.parse_args(argv)
    run(
        map_name=args.map,
        direction=args.direction,
        plot=not args.no_plot,
        from_stage=args.from_stage,
        zone_spec=zones.parse_zone_spec(args.zones),
        vmax=args.vmax,
    )
