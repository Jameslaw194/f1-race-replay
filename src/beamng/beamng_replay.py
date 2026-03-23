"""BeamNG integration for F1 telemetry replay.

Replays George Russell's (Mercedes) F1 race telemetry data inside BeamNG Drive
on a blank ``gridmap_v2`` test map.  The car is teleported to the correct
world-space position every frame and its control inputs (throttle, brake, gear)
are applied so the in-game representation tracks the real onboard data as
closely as possible.

Dependencies
------------
- fastf1   (already in requirements.txt)
- beamngpy  (add with: pip install beamngpy)

BeamNG Drive must be installed and either already running (pass its TCP port)
or the path to its executable must be supplied so BeamNGpy can launch it.
"""

import math
import os
import time

import numpy as np

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
BEAMNG_MAP = "gridmap_v2"      # Blank, flat test environment
BEAMNG_CAR_MODEL = "etk800"    # Closest vanilla Mercedes-style car in BeamNG
BEAMNG_CAR_COLOR = "Silver"    # Traditional Mercedes livery colour

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
        os.makedirs(cache_path)
    fastf1.Cache.enable_cache(cache_path)

    print(f"Loading F1 {year} Round {round_number} Race session …")
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=True)

    event_name = session.event["EventName"]
    print(f"Session: {event_name}")
    print(
        f"Extracting telemetry for {DRIVER_NAME} ({DRIVER_CODE}) …"
    )

    driver_laps = session.laps.pick_drivers(DRIVER_CODE)
    if driver_laps.empty:
        raise ValueError(
            f"No laps found for driver {DRIVER_CODE} in "
            f"{year} Round {round_number}."
        )

    t_segs, x_segs, y_segs = [], [], []
    speed_segs, gear_segs, throttle_segs, brake_segs = [], [], [], []

    for _, lap in driver_laps.iterlaps():
        try:
            tel = lap.get_telemetry()
        except Exception as exc:
            print(
                f"  Warning: could not get telemetry for lap "
                f"{lap.LapNumber}: {exc}"
            )
            continue

        if tel.empty:
            continue

        t_segs.append(tel["SessionTime"].dt.total_seconds().to_numpy())
        x_segs.append(tel["X"].to_numpy())
        y_segs.append(tel["Y"].to_numpy())
        speed_segs.append(tel["Speed"].to_numpy())
        gear_segs.append(tel["nGear"].to_numpy())
        throttle_segs.append(tel["Throttle"].to_numpy())
        brake_segs.append(tel["Brake"].to_numpy().astype(float))

    if not t_segs:
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

    # Centre the track coordinates so the midpoint of the bounding box
    # maps to BeamNG's world origin (0, 0).
    x_centre = (x.max() + x.min()) / 2.0
    y_centre = (y.max() + y.min()) / 2.0
    x -= x_centre
    y -= y_centre

    duration = t[-1] - t[0]
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
        self.beamng_home = beamng_home
        self.beamng_port = beamng_port
        self.bng = None
        self.vehicle = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scenario(self, start_x: float, start_y: float) -> None:
        """Create the blank-map scenario and spawn the vehicle."""
        scenario = Scenario(
            BEAMNG_MAP,
            "F1_Telemetry_Replay",
            description=(
                f"{DRIVER_NAME} ({TEAM_NAME}) — F1 telemetry replay"
            ),
        )

        self.vehicle = Vehicle(
            "george_russell",
            model=BEAMNG_CAR_MODEL,
            color=BEAMNG_CAR_COLOR,
        )

        scenario.add_vehicle(
            self.vehicle,
            pos=(start_x, start_y, GROUND_Z),
            rot_quat=(0.0, 0.0, 0.0, 1.0),
        )

        scenario.make(self.bng)

        # Load and start – try the newer API first, fall back to the old one
        try:
            self.bng.scenario.load(scenario)
            self.bng.scenario.start()
        except AttributeError:
            self.bng.load_scenario(scenario)
            self.bng.start_scenario()

        print("Scenario loaded on gridmap_v2.  Starting replay …\n")

    def _disable_ai(self) -> None:
        """Turn off the vehicle's AI so manual inputs take effect."""
        try:
            self.vehicle.ai.set_mode("disabled")
        except Exception:
            pass  # Older BeamNGpy versions don't expose ai.set_mode

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, telemetry: dict) -> None:
        """Replay *telemetry* in BeamNG from start to finish.

        Iterates over every sample in the telemetry dict, teleports the
        car to the correct (x, y, z) position, sets the heading from the
        direction of travel, and applies throttle / brake / gear inputs.

        Press **Ctrl-C** at any time to stop early.
        """
        t = telemetry["t"]
        x = telemetry["x"]
        y = telemetry["y"]
        speed = telemetry["speed"]
        gear = telemetry["gear"]
        throttle = telemetry["throttle"]
        brake = telemetry["brake"]
        n = len(t)

        event_name = telemetry.get("event_name", "Unknown Event")
        print(
            f"=== BeamNG F1 Telemetry Replay ===\n"
            f"Driver : {DRIVER_NAME} ({DRIVER_CODE})  #{CAR_NUMBER}\n"
            f"Team   : {TEAM_NAME}\n"
            f"Event  : {event_name}\n"
            f"Frames : {n:,} at {REPLAY_FPS} FPS "
            f"({(t[-1] - t[0]) / 60:.1f} min)\n"
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

        launch = self.beamng_home is not None
        self.bng.open(launch=launch)

        try:
            self._build_scenario(float(x[0]), float(y[0]))
            self._disable_ai()

            # Give BeamNG physics a moment to settle after spawning
            time.sleep(2.0)

            heading = 0.0          # running heading (radians)
            progress_interval = REPLAY_FPS * 10  # print every ~10 s

            print(
                f"Replay running — press Ctrl-C to stop.\n"
                f"{'Time':>8}  {'Speed':>10}  {'Gear':>5}  "
                f"{'Throttle':>10}  {'Pos (x, y)':>22}"
            )
            print("-" * 65)

            for i in range(n):
                frame_start = time.monotonic()

                pos_x = float(x[i])
                pos_y = float(y[i])

                # ---- heading from neighbouring samples -------------------
                if i + 1 < n:
                    h = _heading_from_delta(
                        float(x[i + 1]) - pos_x,
                        float(y[i + 1]) - pos_y,
                    )
                elif i > 0:
                    h = _heading_from_delta(
                        pos_x - float(x[i - 1]),
                        pos_y - float(y[i - 1]),
                    )
                else:
                    h = None

                if h is not None:
                    heading = h

                rot_quat = _heading_to_quat(heading)

                # ---- teleport to F1 position ----------------------------
                self.vehicle.teleport(
                    pos=(pos_x, pos_y, GROUND_Z),
                    rot_quat=rot_quat,
                    reset=False,
                )

                # ---- apply control inputs --------------------------------
                # Throttle in FastF1 is 0–100 %; BeamNG expects 0.0–1.0
                bng_throttle = float(np.clip(throttle[i] / 100.0, 0.0, 1.0))
                # Brake in FastF1 can be a boolean (0/1) or a percentage (0–100).
                # Normalise to [0, 1] for BeamNG by dividing if >1.
                raw_brake = float(brake[i])
                bng_brake = float(np.clip(raw_brake / 100.0 if raw_brake > 1.0 else raw_brake, 0.0, 1.0))
                # Gear: F1 uses 1–8; neutral/negative values default to 1
                bng_gear = max(1, int(round(float(gear[i]))))

                self.vehicle.control(
                    throttle=bng_throttle,
                    brake=bng_brake,
                    steering=0.0,
                    parkingbrake=0.0,
                    gear=bng_gear,
                )

                # ---- progress log every ~10 s ---------------------------
                if i % progress_interval == 0:
                    elapsed = t[i] - t[0]
                    print(
                        f"{elapsed:>7.1f}s  "
                        f"{float(speed[i]):>8.1f} km/h  "
                        f"{bng_gear:>5d}  "
                        f"{bng_throttle * 100:>8.1f} %  "
                        f"({pos_x:>8.1f}, {pos_y:>8.1f})"
                    )

                # ---- maintain replay cadence ----------------------------
                elapsed_wall = time.monotonic() - frame_start
                sleep_for = FRAME_DT - elapsed_wall
                if sleep_for > 0:
                    time.sleep(sleep_for)

            print("\nReplay finished.")

        except KeyboardInterrupt:
            print("\nReplay stopped by user (Ctrl-C).")
        finally:
            if self.bng is not None:
                try:
                    self.bng.close()
                except Exception:
                    pass
