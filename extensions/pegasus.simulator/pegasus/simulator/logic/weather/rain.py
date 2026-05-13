# pegasus.simulator.logic.weather.rain
"""
Visual-only rain that follows a drone and drifts with wind.

Uses UsdGeom.PointInstancer to render N elongated streaks. Positions
and orientation are updated in Python each step; no PhysX involved.
Cost is dominated by the per-frame USD array write — 3000 drops costs
well under 1 ms.
"""
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, Vt


def _quat_align_z(v: np.ndarray) -> Gf.Quath:
    """Quaternion rotating +Z onto direction v (unit-length not required)."""
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return Gf.Quath(1, 0, 0, 0)
    v = v / n
    dot = float(v[2])
    if dot > 0.99999:
        return Gf.Quath(1, 0, 0, 0)
    if dot < -0.99999:
        return Gf.Quath(0, 1, 0, 0)         # 180° flip
    axis = np.cross([0.0, 0.0, 1.0], v)
    axis /= (np.linalg.norm(axis) + 1e-12)
    half = np.arccos(np.clip(dot, -1.0, 1.0)) * 0.5
    s = np.sin(half)
    return Gf.Quath(float(np.cos(half)),
                    float(axis[0] * s),
                    float(axis[1] * s),
                    float(axis[2] * s))


class RainField:
    """
    Parameters
    ----------
    wind_field    : WindField — drives horizontal drift
    follow_path   : str | None — prim path whose translation centers the box
                    (e.g. "/World/CAP3"); None pins to world origin
    num_drops     : int  — instance count (1000=light, 3000=normal, 6000=storm)
    box           : (sx, sy, sz) — drop spawn volume in metres around follow point
    fall_speed    : float — terminal velocity in m/s (rain ≈ 8, drizzle ≈ 3)
    streak_length : float — cylinder length in metres
    streak_radius : float — cylinder radius in metres
    color         : (r, g, b)
    opacity       : float in [0, 1]
    """

    def __init__(self, stage, wind_field,
                 follow_path   = None,
                 num_drops     = 3000,
                 box           = (40.0, 40.0, 25.0),
                 fall_speed    = 8.0,
                 streak_length = 0.08,
                 streak_radius = 0.0025,
                 color         = (0.7, 0.82, 0.95),
                 opacity       = 0.55,
                 instancer_path= "/World/Rain"):
        self.stage       = stage
        self.wind        = wind_field
        self.follow_path = follow_path
        self.N           = int(num_drops)
        self.box         = np.asarray(box, dtype=np.float32)
        self.fall_speed  = float(fall_speed)
        self._center     = np.zeros(3, dtype=np.float32)
        self._last_q     = None

        self._build_instancer(instancer_path, streak_length, streak_radius,
                              color, opacity)
        self._init_positions()

    # ------------------------------------------------------------------
    def _build_instancer(self, path, length, radius, color, opacity):
        inst   = UsdGeom.PointInstancer.Define(self.stage, path)
        proto  = UsdGeom.Cylinder.Define(self.stage, f"{path}/Proto/Streak")
        proto.CreateAxisAttr("Z")
        proto.CreateHeightAttr(length)
        proto.CreateRadiusAttr(radius)
        proto.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        proto.CreateDisplayOpacityAttr([float(opacity)])
        # Hide the prototype itself; only the instancer's copies should draw.
        UsdGeom.Imageable(proto).MakeInvisible()

        inst.CreatePrototypesRel().AddTarget(proto.GetPath())
        self._inst = inst

    def _init_positions(self):
        rng = np.random.default_rng()
        self._pos = (rng.random((self.N, 3), dtype=np.float32) - 0.5) * self.box
        self._proto_idx = np.zeros(self.N, dtype=np.int32)
        self._push_all()

    def _push_all(self):
        # Compute orientation from current effective velocity
        v = self._effective_velocity()
        q = _quat_align_z(v)
        orients = Vt.QuathArray([q] * self.N)

        abs_pos = self._pos + self._center
        self._inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(abs_pos))
        self._inst.CreateProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(self._proto_idx))
        self._inst.CreateOrientationsAttr().Set(orients)
        self._last_q = q

    def _effective_velocity(self) -> np.ndarray:
        v = self.wind.vector().astype(np.float32).copy()
        v[2] -= self.fall_speed
        return v

    # ------------------------------------------------------------------
    def step(self, dt: float):
        if dt <= 0.0:
            return

        # Follow the drone
        if self.follow_path is not None:
            prim = self.stage.GetPrimAtPath(self.follow_path)
            if prim.IsValid():
                t = omni.usd.get_world_transform_matrix(prim).ExtractTranslation()
                self._center = np.array([t[0], t[1], t[2]], dtype=np.float32)

        # Advance positions
        v = self._effective_velocity()
        self._pos += v * dt

        # Wrap X/Y modulo box (drops leaving one side re-enter the other)
        half = self.box * 0.5
        for ax in (0, 1):
            self._pos[:, ax] = ((self._pos[:, ax] + half[ax]) % self.box[ax]) - half[ax]
        # Z: drops below floor respawn at top
        below = self._pos[:, 2] < -half[2]
        if np.any(below):
            self._pos[below, 2]    = half[2]
            # Re-randomize X/Y so respawned drops don't form vertical columns
            self._pos[below, 0:2]  = (np.random.rand(int(below.sum()), 2).astype(np.float32) - 0.5) * self.box[0:2]

        # Write to USD
        abs_pos = self._pos + self._center
        self._inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(abs_pos))

        # Orientation only changes if wind direction shifts noticeably
        q = _quat_align_z(v)
        if q != self._last_q:
            self._inst.GetOrientationsAttr().Set(Vt.QuathArray([q] * self.N))
            self._last_q = q

    def set_intensity(self, num_drops: int):
        """Re-size the instancer for storm vs drizzle without rebuilding."""
        if num_drops == self.N:
            return
        rng = np.random.default_rng()
        self._pos = np.resize(self._pos, (num_drops, 3))
        # Refill any new slots with random positions in the box
        if num_drops > self.N:
            extra = (rng.random((num_drops - self.N, 3), dtype=np.float32) - 0.5) * self.box
            self._pos[self.N:] = extra
        self.N = num_drops
        self._proto_idx = np.zeros(self.N, dtype=np.int32)
        self._push_all()