# pegasus.simulator.logic.weather.wind_spray
"""
Apply WindField to PhysX particle batches.

The built-in PhysxForceFieldWindAPI does not affect PhysX particle
systems in current Isaac Sim — only rigid bodies. This helper fills
that gap by applying a drag-coupled wind force to each active particle
batch:  Δv = (1 - exp(-drag·dt)) · (v_wind - v_particle)

Particles asymptote toward wind velocity with time-constant 1/drag.
"""
import numpy as np
from pxr import UsdGeom, Vt


class WindOnSpray:
    def __init__(self, stage, wind_field, drag: float = 4.0):
        """
        Parameters
        ----------
        drag : float [1/s]
            Coupling strength between the air and a particle.
            Larger drag = particles converge to wind velocity faster.
            Reasonable range for water spray droplets: 2-8.
        """
        self.stage = stage
        self.wind  = wind_field
        self.drag  = float(drag)
        self._cache = {}    # path -> UsdGeom.Points

    def emission_bias(self, n: int) -> np.ndarray:
        """Wind velocity (N,3) to add to initial particle velocities at emission."""
        v = self.wind.vector().astype(np.float32)
        return np.tile(v, (n, 1))

    def step(self, dt: float, batch_paths):
        """Pull each live batch's velocity toward v_wind by one dt step."""
        if dt <= 0.0:
            return
        v_wind = self.wind.vector().astype(np.float32)
        alpha  = float(1.0 - np.exp(-self.drag * dt))   # ∈ (0,1)

        for path in batch_paths:
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            v_attr = UsdGeom.Points(prim).GetVelocitiesAttr()
            if not v_attr or not v_attr.HasValue():
                continue
            vel = v_attr.Get()
            if not vel:
                continue
            v_arr  = np.array(vel, dtype=np.float32)        # (N,3) — copy
            v_arr += alpha * (v_wind - v_arr)
            v_attr.Set(Vt.Vec3fArray.FromNumpy(v_arr))