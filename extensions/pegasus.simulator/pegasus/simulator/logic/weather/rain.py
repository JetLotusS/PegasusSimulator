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
    follow_drone   : str | None — prim path whose translation centers the box
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
                 follow_drone   = None,
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
        self.follow_drone = follow_drone
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
        inst = UsdGeom.PointInstancer.Define(self.stage, path)

        UsdGeom.Imageable(inst.GetPrim()).MakeVisible()

        proto_scope = UsdGeom.Scope.Define(self.stage, f"{path}/Prototypes")
        UsdGeom.Imageable(proto_scope.GetPrim()).MakeInvisible()

        proto = UsdGeom.Cylinder.Define(self.stage, f"{path}/Prototypes/Streak")
        proto.CreateAxisAttr("Z")
        proto.CreateHeightAttr(length)
        proto.CreateRadiusAttr(radius)
        proto.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        proto.CreateDisplayOpacityAttr([float(opacity)])

        inst.CreatePrototypesRel().AddTarget(proto.GetPath())
        self._inst = inst

    def _init_positions(self):
        rng = np.random.default_rng()
        if self.follow_drone is not None:
            try:
                p = self.follow_drone.state.position
                self._center = np.array([p[0], p[1], p[2]], dtype=np.float32)
            except Exception:
                pass
        self._pos = ((rng.random((self.N, 3), dtype=np.float32) - 0.5) * self.box) + self._center
        self._proto_idx = np.zeros(self.N, dtype=np.int32)
        self._push_all()

    def _push_all(self):
        v = self._effective_velocity()
        q = _quat_align_z(v)
        orients = Vt.QuathArray([q] * self.N)
        self._inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(self._pos))
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

        # Track drone position (only used as wrap-around center, not applied to particles)
        if self.follow_drone is not None:
            try:
                p = self.follow_drone.state.position
                self._center = np.array([p[0], p[1], p[2]], dtype=np.float32)
            except Exception:
                pass

        # Advance positions in world space — pure physics, no drone coupling
        v = self._effective_velocity()
        self._pos += v * dt

        # Periodic wrap of X/Y around the current drone position
        half = self.box * 0.5
        rel  = self._pos - self._center
        for ax in (0, 1):
            out_pos = rel[:, ax] >  half[ax]
            out_neg = rel[:, ax] < -half[ax]
            self._pos[out_pos, ax] -= self.box[ax]
            self._pos[out_neg, ax] += self.box[ax]

        # Z: drops that fell out the bottom respawn at the top of the box,
        #    with fresh random X/Y around the drone
        below = (self._pos[:, 2] - self._center[2]) < -half[2]
        if np.any(below):
            n_b = int(below.sum())
            self._pos[below, 2]    = self._center[2] + half[2]
            self._pos[below, 0:2]  = (np.random.rand(n_b, 2).astype(np.float32) - 0.5) * self.box[0:2] + self._center[0:2]

        # Drops too far above (drone descended fast) — clamp to top
        above = (self._pos[:, 2] - self._center[2]) > half[2]
        if np.any(above):
            self._pos[above, 2] = self._center[2] + half[2]

        # Write world positions
        self._inst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(self._pos))

        # Orientation
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