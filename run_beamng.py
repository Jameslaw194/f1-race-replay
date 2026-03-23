#!/usr/bin/env python3
"""BeamNG F1 Telemetry Replay – entry point.

Loads George Russell's (Mercedes, #63) race telemetry via FastF1 and
replays it inside BeamNG Drive on the blank ``smallgrid`` test map.
The in-game car follows the real position and speed data from the chosen
race weekend.

At the same time, the normal F1 Race Replay map and telemetry / insights
menu are launched in a background process so you can watch the replay
from both perspectives simultaneously.  Pass ``--no-map`` to skip that.

Requirements
------------
- BeamNG Drive installed (https://www.beamng.com)
- ``beamngpy``  –  pip install beamngpy
- ``fastf1``    –  already in requirements.txt

Usage examples
--------------
# BeamNG already running (default port 64256):
    python run_beamng.py --year 2023 --round 1

# Auto-launch BeamNG (provide the install directory):
    python run_beamng.py --year 2023 --round 1 --beamng-home "C:/BeamNG.drive"

# Skip the 2-D map window:
    python run_beamng.py --year 2023 --round 1 --no-map

Arguments
---------
--year          F1 season year (default: 2023)
--round         Race round number (default: 1)
--beamng-home   Path to BeamNG Drive install directory.  When provided,
                BeamNGpy will launch BeamNG automatically.  Omit if
                BeamNG is already running.
--port          TCP port BeamNG listens on (default: 64256)
--cache         FastF1 cache directory (default: fastf1_cache)
--no-map        Do not open the 2-D race map / telemetry window
"""

import argparse
import os
import subprocess
import sys
import tempfile
import uuid


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay George Russell's F1 telemetry in BeamNG Drive on a "
            "blank map (smallgrid), with the normal 2-D race map and "
            "insights menu opening alongside."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="F1 season year (default: 2023)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Race round number within the season (default: 1)",
    )
    parser.add_argument(
        "--beamng-home",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to the BeamNG Drive installation directory.  "
            "Required if BeamNG is not already running."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=64256,
        help="TCP port BeamNG listens on (default: 64256)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="fastf1_cache",
        metavar="DIR",
        help="FastF1 cache directory (default: fastf1_cache)",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        default=False,
        help="Skip launching the 2-D race map / telemetry window",
    )
    return parser.parse_args()


def _launch_map_replay(year: int, round_number: int, sync_file: str) -> subprocess.Popen:
    """Launch the normal F1 race map + insights menu in a separate process.

    Delegates to ``main.py --viewer`` with the same year/round so the full
    2-D visualisation starts alongside the BeamNG replay.  The map is held
    paused at frame 0 until the sync file is written (just before the BeamNG
    AI script starts), ensuring both views play in lock-step.  Returns the
    ``Popen`` handle; the process is deliberately left detached.
    """
    main_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "main.py")
    )
    cmd = [
        sys.executable,
        main_path,
        "--viewer",
        "--year", str(year),
        "--round", str(round_number),
        "--start-paused",
        "--sync-file", sync_file,
    ]
    print(
        f"Launching F1 race map (year={year}, round={round_number}) …\n"
        f"  Command: {' '.join(cmd)}\n"
        f"  Sync file: {sync_file}\n"
    )
    try:
        proc = subprocess.Popen(cmd)
        print(f"F1 race map process started (PID {proc.pid}).\n")
        return proc
    except Exception as exc:
        print(
            f"Warning: could not launch F1 race map: {exc}\n"
            "  The BeamNG replay will still continue.\n",
            file=sys.stderr,
        )
        return None


def main():
    args = _parse_args()

    # Check beamngpy is available before doing any work
    try:
        from src.beamng.beamng_replay import (
            load_driver_telemetry,
            BeamNGF1Replay,
            HAS_BEAMNGPY,
            HAS_FASTF1,
        )
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not HAS_FASTF1:
        print(
            "Error: fastf1 is not installed.\n"
            "  Install it with:  pip install fastf1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not HAS_BEAMNGPY:
        print(
            "Error: beamngpy is not installed.\n"
            "  Install it with:  pip install beamngpy",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Optionally launch the normal F1 race map + telemetry insights menu.
    #    The map is held paused at frame 0 until BeamNG is ready to start.
    sync_file = os.path.join(
        tempfile.gettempdir(), f"f1_beamng_sync_{uuid.uuid4().hex}"
    )
    map_proc = None
    if not args.no_map:
        map_proc = _launch_map_replay(args.year, args.round, sync_file)
    else:
        print("--no-map specified — skipping 2-D race map launch.\n")

    # 2. Load telemetry
    telemetry = load_driver_telemetry(
        year=args.year,
        round_number=args.round,
        cache_path=args.cache,
    )

    # 3. Run BeamNG replay (passes sync_file so it is written just before the
    #    AI script is submitted, triggering the map to start in sync)
    replay = BeamNGF1Replay(
        beamng_home=args.beamng_home,
        beamng_port=args.port,
    )
    replay.run(telemetry, sync_file=sync_file if not args.no_map else None)

    # 4. If the map window is still open when BeamNG finishes, leave it
    #    running — the user can close it manually.
    if map_proc is not None and map_proc.poll() is None:
        print(
            "\nBeamNG replay finished.  The F1 race map window is still open.\n"
            "Close it manually when you are done."
        )


if __name__ == "__main__":
    main()
