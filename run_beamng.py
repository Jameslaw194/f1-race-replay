#!/usr/bin/env python3
"""BeamNG F1 Telemetry Replay – entry point.

Loads George Russell's (Mercedes, #63) race telemetry via FastF1 and
replays it inside BeamNG Drive on the blank ``gridmap_v2`` test map.
The in-game car follows the real position, speed, gear, throttle and
brake data from the chosen race weekend.

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

# Different port:
    python run_beamng.py --year 2023 --round 5 --port 64257

Arguments
---------
--year          F1 season year (default: 2023)
--round         Race round number (default: 1)
--beamng-home   Path to BeamNG Drive install directory.  When provided,
                BeamNGpy will launch BeamNG automatically.  Omit if
                BeamNG is already running.
--port          TCP port BeamNG listens on (default: 64256)
--cache         FastF1 cache directory (default: fastf1_cache)
"""

import argparse
import sys


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay George Russell's F1 telemetry in BeamNG Drive on a "
            "blank map (gridmap_v2)."
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
    return parser.parse_args()


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

    # 1. Load telemetry
    telemetry = load_driver_telemetry(
        year=args.year,
        round_number=args.round,
        cache_path=args.cache,
    )

    # 2. Run BeamNG replay
    replay = BeamNGF1Replay(
        beamng_home=args.beamng_home,
        beamng_port=args.port,
    )
    replay.run(telemetry)


if __name__ == "__main__":
    main()
