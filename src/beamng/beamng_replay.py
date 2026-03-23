"""BeamNG integration for F1 telemetry replay.

Replays George Russell's (Mercedes) F1 race telemetry data inside BeamNG Drive
on the ``smallgrid`` test map.  The entire telemetry path is submitted to
BeamNG's built-in AI script system in one shot so the car *drives* smoothly
from point to point rather than being teleported every frame.

Dependencies
------------
- fastf1   (already in requirements.txt)
- beamngpy  (add with: pip install beamngpy)

BeamNG Drive must be installed and either already running (pass its TCP port)
or the path to its executable must be supplied so BeamNGpy can launch it.
"""

import logging
import math
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

def _configure_logging() -> None:
    """Attach a console handler to the f1_beamng logger if none exists yet.

    Called automatically when BeamNGF1Replay is first used.  A separate
    logger name keeps BeamNG chatter distinct from the rest of the app.
    """
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [BeamNG] %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    log.addHandler(handler)
    if log.level == logging.NOTSET:
        log.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Optional dependency guards – both give clear messages if missing.
# ---------------------------------------------------------------------------

try:
    import fastf1  # noqa: F401
    HAS_FASTF1 = True
except ImportError:
    HAS_FASTF1 = False

try:
    from beamngpy import BeamNGpy, Scenario, Vehicle  # noqa: F401
    HAS_BEAMNGPY = True
except ImportError:
    HAS_BEAMNGPY = False

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# George Russell – Mercedes
DRIVER_CODE = "RUS"
DRIVER_NAME = "George Russell"
TEAM_NAME = "Mercedes"
CAR_NUMBER = 63

# BeamNG scene
BEAMNG_MAP = "smallgrid"       # Blank, flat test environment
BEAMNG_CAR_MODEL = "FR26"      # F1-style car available in BeamNG

# Replay timing
REPLAY_FPS = 25                # Replay frame rate (matches the telemetry sampling rate)
FRAME_DT = 1.0 / REPLAY_FPS

# Car sits 0.3 m above the ground plane
GROUND_Z = 0.3


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

def _heading_to_quat(heading_rad: float) -> tuple:
    """Return a (qx, qy, qz, qw) quaternion for a yaw (Z-axis rotation).

    BeamNG uses a right-handed coordinate system where Z points up.
    A heading of 0 means the car faces the positive-X direction.
    """
    half = heading_rad * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _heading_from_delta(dx: float, dy: float) -> "float | None":
    """Return the heading angle in radians, or *None* if the delta is zero."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return math.atan2(dy, dx)


# ---------------------------------------------------------------------------
# Telemetry loader
# ---------------------------------------------------------------------------

def load_driver_telemetry(
    year: int,
    round_number: int,
    cache_path: str = "fastf1_cache",
) -> dict:
    """Load George Russell's complete race telemetry for the given event.

    Parameters
    ----------
    year:
        F1 season year (e.g. 2023).
    round_number:
        Round number within the season (e.g. 1 for Bahrain).
    cache_path:
        Directory used by FastF1's local cache.  Created if absent.

    Returns
    -------
    dict with keys:
        ``t``          – session timestamps (seconds, sorted, shape N)
        ``x``          – track-centred X coordinates (metres, shape N)
        ``y``          – track-centred Y coordinates (metres, shape N)
        ``speed``      – speed in km/h (shape N)
        ``gear``       – gear index 1–8 (shape N)
        ``throttle``   – throttle 0–100 % (shape N)
        ``brake``      – brake input 0–1 (shape N)
        ``event_name`` – human-readable event name string
        ``year``       – as supplied
        ``round``      – as supplied
    """
    if not HAS_FASTF1:
        raise ImportError(
            "fastf1 is not installed.  Run: pip install fastf1"
        )

    if not os.path.exists(cache_path):
        log.debug("Creating FastF1 cache directory: %s", cache_path)
        os.makedirs(cache_path)
    log.debug("FastF1 cache directory: %s", os.path.abspath(cache_path))
    fastf1.Cache.enable_cache(cache_path)

    log.info("Fetching F1 %d Round %d Race session from FastF1 …", year, round_number)
    print(f"Loading F1 {year} Round {round_number} Race session …")
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=True)

    event_name = session.event["EventName"]
    log.info("Session loaded: %s (year=%d, round=%d)", event_name, year, round_number)
    print(f"Session: {event_name}")
    print(
        f"Extracting telemetry for {DRIVER_NAME} ({DRIVER_CODE}) …"
    )

    driver_laps = session.laps.pick_drivers(DRIVER_CODE)
    if driver_laps.empty:
        log.error("No laps found for %s in %d Round %d", DRIVER_CODE, year, round_number)
        raise ValueError(
            f"No laps found for driver {DRIVER_CODE} in "
            f"{year} Round {round_number}."
        )

    total_laps = len(driver_laps)
    log.info("Found %d laps for %s — extracting telemetry …", total_laps, DRIVER_CODE)

    t_segs, x_segs, y_segs = [], [], []
    speed_segs, gear_segs, throttle_segs, brake_segs = [], [], [], []

    for _, lap in driver_laps.iterlaps():
        lap_num = getattr(lap, "LapNumber", "?")
        log.debug("Processing lap %s …", lap_num)
        try:
            tel = lap.get_telemetry()
        except Exception as exc:
            log.warning("Could not get telemetry for lap %s: %s", lap_num, exc)
            print(
                f"  Warning: could not get telemetry for lap "
                f"{lap_num}: {exc}"
            )
            continue

        if tel.empty:
            log.debug("Lap %s telemetry is empty — skipping.", lap_num)
            continue

        samples = len(tel)
        log.debug(
            "Lap %s: %d samples  speed=[%.0f, %.0f] km/h  gears=[%d, %d]",
            lap_num,
            samples,
            tel["Speed"].min(),
            tel["Speed"].max(),
            int(tel["nGear"].min()),
            int(tel["nGear"].max()),
        )

        t_segs.append(tel["SessionTime"].dt.total_seconds().to_numpy())
        x_segs.append(tel["X"].to_numpy())
        y_segs.append(tel["Y"].to_numpy())
        speed_segs.append(tel["Speed"].to_numpy())
        gear_segs.append(tel["nGear"].to_numpy())
        throttle_segs.append(tel["Throttle"].to_numpy())
        brake_segs.append(tel["Brake"].to_numpy().astype(float))

    if not t_segs:
        log.error("No valid telemetry segments found for %s.", DRIVER_CODE)
        raise ValueError(
            f"No valid telemetry segments found for {DRIVER_CODE}."
        )

    t = np.concatenate(t_segs)
    x = np.concatenate(x_segs)
    y = np.concatenate(y_segs)
    speed = np.concatenate(speed_segs)
    gear = np.concatenate(gear_segs)
    throttle = np.concatenate(throttle_segs)
    brake = np.concatenate(brake_segs)

    # Sort chronologically
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    speed, gear = speed[order], gear[order]
    throttle, brake = throttle[order], brake[order]

    log.debug("Merged %d segments into %d total samples.", len(t_segs), len(t))

    # Centre the track coordinates so the midpoint of the bounding box
    # maps to BeamNG's world origin (0, 0).
    x_centre = (x.max() + x.min()) / 2.0
    y_centre = (y.max() + y.min()) / 2.0
    x -= x_centre
    y -= y_centre

    duration = t[-1] - t[0]
    log.info(
        "Telemetry ready: %d samples  duration=%.1f s (%.1f min)",
        len(t), duration, duration / 60,
    )
    log.info(
        "Track extent (centred): X [%.0f, %.0f] m   Y [%.0f, %.0f] m",
        x.min(), x.max(), y.min(), y.max(),
    )
    log.info(
        "Speed range: %.0f – %.0f km/h   Gear range: %d – %d",
        speed.min(), speed.max(), int(gear.min()), int(gear.max()),
    )
    print(
        f"Loaded {len(t):,} telemetry samples spanning {duration:.1f} s "
        f"({duration / 60:.1f} min)."
    )
    print(
        f"Track extent after centring: "
        f"X [{x.min():.0f}, {x.max():.0f}] m  "
        f"Y [{y.min():.0f}, {y.max():.0f}] m"
    )

    return {
        "t": t,
        "x": x,
        "y": y,
        "speed": speed,
        "gear": gear,
        "throttle": throttle,
        "brake": brake,
        "event_name": event_name,
        "year": year,
        "round": round_number,
    }


# ---------------------------------------------------------------------------
# BeamNG replay controller
# ---------------------------------------------------------------------------

class BeamNGF1Replay:
    """Drives a BeamNG vehicle along George Russell's F1 telemetry path.

    The entire telemetry path is submitted to BeamNG's AI script system so
    the car physically drives from waypoint to waypoint rather than being
    teleported frame-by-frame.  BeamNG's physics engine handles steering,
    acceleration, and braking to follow the path at the recorded speeds,
    giving a smooth, natural-looking replay.

    Usage
    -----
    ::

        telemetry = load_driver_telemetry(2023, 1)
        replay = BeamNGF1Replay(beamng_home="/path/to/beamng")
        replay.run(telemetry)

    Parameters
    ----------
    beamng_home:
        Path to the BeamNG Drive installation directory.  Required when
        BeamNG is not already running.  Pass *None* to connect to an
        already-running instance.
    beamng_port:
        TCP port BeamNG listens on (default 64256).
    """

    def __init__(
        self,
        beamng_home: str = None,
        beamng_port: int = 64256,
    ) -> None:
        if not HAS_BEAMNGPY:
            raise ImportError(
                "beamngpy is not installed.  Run: pip install beamngpy"
            )
        _configure_logging()
        self.beamng_home = beamng_home
        self.beamng_port = beamng_port
        self.bng = None
        self.vehicle = None
        log.debug(
            "BeamNGF1Replay created — home=%s  port=%d",
            beamng_home or "(connect to running instance)",
            beamng_port,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scenario(self, start_x: float, start_y: float, start_heading: float) -> None:
        """Create the blank-map scenario and spawn the vehicle."""
        log.info(
            "Building scenario — map=%s  vehicle=%s  spawn=(%.2f, %.2f, %.2f)  heading=%.3f rad",
            BEAMNG_MAP, BEAMNG_CAR_MODEL, start_x, start_y, GROUND_Z, start_heading,
        )
        scenario = Scenario(
            BEAMNG_MAP,
            "F1_Telemetry_Replay",
            description=(
                f"{DRIVER_NAME} ({TEAM_NAME}) — F1 telemetry replay"
            ),
        )
        log.debug("Scenario object created: name=F1_Telemetry_Replay  map=%s", BEAMNG_MAP)

        self.vehicle = Vehicle(
            "george_russell",
            model=BEAMNG_CAR_MODEL,
        )
        log.debug("Vehicle object created: id=george_russell  model=%s", BEAMNG_CAR_MODEL)

        rot_quat = _heading_to_quat(start_heading)
        log.debug("Spawn rotation quaternion: %s", rot_quat)
        scenario.add_vehicle(
            self.vehicle,
            pos=(start_x, start_y, GROUND_Z),
            rot_quat=rot_quat,
        )
        log.debug("Vehicle added to scenario at pos=(%.2f, %.2f, %.2f)", start_x, start_y, GROUND_Z)

        log.info("Compiling scenario assets …")
        scenario.make(self.bng)
        log.debug("scenario.make() completed.")

        # Load and start – try the newer API first, fall back to the old one
        log.info("Loading and starting scenario in BeamNG …")
        try:
            self.bng.scenario.load(scenario)
            log.debug("bng.scenario.load() succeeded (newer API).")
            self.bng.scenario.start()
            log.debug("bng.scenario.start() succeeded.")
        except AttributeError:
            log.debug("Newer scenario API not found — falling back to legacy API.")
            self.bng.load_scenario(scenario)
            self.bng.start_scenario()
            log.debug("Legacy bng.load_scenario / bng.start_scenario succeeded.")

        log.info("Scenario active on %s.", BEAMNG_MAP)
        print(f"Scenario loaded on {BEAMNG_MAP}.  Starting replay …\n")

    def _build_ai_script(self, t_arr, x_arr, y_arr) -> list:
        """Convert telemetry arrays to a BeamNG AI waypoint script.

        Each entry contains the world-space position and the elapsed time
        (in seconds from the start of the replay).  BeamNG's AI will
        drive the car through each waypoint at the speed implied by the
        time delta between consecutive entries, producing smooth, natural
        vehicle movement with no teleporting.
        """
        t0 = float(t_arr[0])
        n = len(t_arr)
        script = [
            {
                "x": float(x_arr[i]),
                "y": float(y_arr[i]),
                "z": GROUND_Z,
                "t": float(t_arr[i]) - t0,
            }
            for i in range(n)
        ]
        log.debug(
            "AI script built: %d waypoints  t=[0.0, %.2f] s  "
            "x=[%.0f, %.0f] m  y=[%.0f, %.0f] m",
            n,
            script[-1]["t"],
            min(w["x"] for w in script),
            max(w["x"] for w in script),
            min(w["y"] for w in script),
            max(w["y"] for w in script),
        )
        log.debug(
            "First waypoint: x=%.2f  y=%.2f  z=%.2f  t=0.00",
            script[0]["x"], script[0]["y"], script[0]["z"],
        )
        log.debug(
            "Last waypoint:  x=%.2f  y=%.2f  z=%.2f  t=%.2f",
            script[-1]["x"], script[-1]["y"], script[-1]["z"], script[-1]["t"],
        )
        return script

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, telemetry: dict, sync_file: "str | None" = None) -> None:
        """Replay *telemetry* in BeamNG from start to finish.

        Submits the full telemetry path to BeamNG's AI script system once.
        BeamNG then drives the car smoothly along the path at the correct
        speed for each segment — no teleporting, no jerky jumps.

        Parameters
        ----------
        telemetry:
            Dict returned by :func:`load_driver_telemetry`.
        sync_file:
            Optional path to a file that should be created just before the
            AI script is submitted.  The companion 2-D map process watches
            for this file and unpauses itself the moment it appears, keeping
            both visualisations in lock-step from t=0.

        Press **Ctrl-C** at any time to stop early.
        """
        t = telemetry["t"]
        x = telemetry["x"]
        y = telemetry["y"]
        n = len(t)

        event_name = telemetry.get("event_name", "Unknown Event")
        duration = float(t[-1]) - float(t[0])

        log.info("=== BeamNG F1 Telemetry Replay starting ===")
        log.info("Driver  : %s (%s)  #%d", DRIVER_NAME, DRIVER_CODE, CAR_NUMBER)
        log.info("Team    : %s", TEAM_NAME)
        log.info("Event   : %s", event_name)
        log.info("Map     : %s", BEAMNG_MAP)
        log.info("Car     : %s", BEAMNG_CAR_MODEL)
        log.info("Samples : %d waypoints  duration=%.1f s (%.1f min)", n, duration, duration / 60)

        print(
            f"=== BeamNG F1 Telemetry Replay ===\n"
            f"Driver : {DRIVER_NAME} ({DRIVER_CODE})  #{CAR_NUMBER}\n"
            f"Team   : {TEAM_NAME}\n"
            f"Event  : {event_name}\n"
            f"Map    : {BEAMNG_MAP}\n"
            f"Car    : {BEAMNG_CAR_MODEL}\n"
            f"Frames : {n:,} waypoints  ({duration / 60:.1f} min)\n"
        )

        log.info(
            "Connecting to BeamNG at localhost:%d  (launch=%s) …",
            self.beamng_port,
            self.beamng_home is not None,
        )
        print(
            f"Connecting to BeamNG on localhost:{self.beamng_port} …\n"
            "  (Set --beamng-home to auto-launch BeamNG if not already "
            "running.)\n"
        )

        self.bng = BeamNGpy(
            "localhost",
            self.beamng_port,
            home=self.beamng_home,
        )
        log.debug("BeamNGpy object created: host=localhost  port=%d", self.beamng_port)

        launch = self.beamng_home is not None
        log.info("Opening BeamNG connection (launch=%s) …", launch)
        self.bng.open(launch=launch)
        log.info("BeamNG connection established.")

        try:
            # Compute initial heading from the first two telemetry points
            if n > 1:
                start_heading = _heading_from_delta(
                    float(x[1]) - float(x[0]),
                    float(y[1]) - float(y[0]),
                ) or 0.0
            else:
                start_heading = 0.0
            log.debug("Initial spawn heading: %.4f rad (%.1f°)", start_heading, math.degrees(start_heading))

            self._build_scenario(float(x[0]), float(y[0]), start_heading)

            log.info("Waiting 2 s for BeamNG physics to settle after spawn …")
            # Give BeamNG physics a moment to settle after spawning
            time.sleep(2.0)
            log.debug("Physics settle wait complete.")

            # Build the AI waypoint script from the full telemetry path
            log.info("Building AI waypoint script from %d telemetry samples …", n)
            print(f"Building AI waypoint script from {n:,} telemetry samples …")
            script = self._build_ai_script(t, x, y)
            log.info("AI script ready: %d waypoints spanning %.1f s.", len(script), script[-1]["t"])

            # Submit the script to BeamNG's AI — the car now drives itself
            # smoothly along the recorded F1 path.
            # Write the sync file FIRST so the 2-D map process unpauses at
            # exactly the same moment the BeamNG car begins to move.
            if sync_file:
                try:
                    with open(sync_file, "w") as _sf:
                        _sf.write("sync")
                    log.info("Sync file written → map replay will now start: %s", sync_file)
                except Exception as _sf_exc:
                    log.warning("Could not write sync file %s: %s", sync_file, _sf_exc)

            log.info("Submitting AI script to vehicle '%s' (cling=True) …", BEAMNG_CAR_MODEL)
            print(f"Submitting script to BeamNG AI — the car will drive the route.")
            self.vehicle.ai.set_script(script, cling=True)
            log.info("AI script accepted — vehicle is now following the F1 telemetry path.")

            # Wait for the replay to finish, printing live progress
            log.info("Replay in progress — total duration %.0f s (%.1f min).", duration, duration / 60)
            print(
                f"\nReplay running — press Ctrl-C to stop.\n"
                f"Total duration: {duration:.0f} s ({duration / 60:.1f} min)\n"
            )

            start_wall = time.monotonic()
            last_log_s = -10  # force a log line at t=0
            while True:
                elapsed = time.monotonic() - start_wall
                if elapsed >= duration:
                    break
                pct = min(elapsed / duration * 100.0, 100.0)
                remaining = max(duration - elapsed, 0.0)

                # Console progress bar (overwrites same line)
                print(
                    f"\r  Elapsed {elapsed:>6.0f} s / {duration:.0f} s  "
                    f"({pct:>5.1f} %)  remaining {remaining:>5.0f} s",
                    end="",
                    flush=True,
                )

                # Detailed log line every 30 s
                if elapsed - last_log_s >= 30:
                    log.info(
                        "Progress: %.0f / %.0f s  (%.1f %%)  remaining %.0f s",
                        elapsed, duration, pct, remaining,
                    )
                    last_log_s = elapsed

                time.sleep(1.0)

            print("\n")
            log.info("Replay completed normally after %.1f s.", duration)
            print("Replay finished.")

        except KeyboardInterrupt:
            log.warning("Replay interrupted by user (Ctrl-C).")
            print("\nReplay stopped by user (Ctrl-C).")
        except Exception as exc:
            log.exception("Unexpected error during replay: %s", exc)
            raise
        finally:
            log.info("Closing BeamNG connection …")
            if self.bng is not None:
                try:
                    self.bng.close()
                    log.info("BeamNG connection closed cleanly.")
                except Exception as close_exc:
                    log.warning("Error while closing BeamNG connection: %s", close_exc)

