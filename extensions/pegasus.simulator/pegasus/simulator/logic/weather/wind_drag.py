# weather/wind_drag.py
import numpy as np
from pegasus.simulator.logic.dynamics.drag import Drag

class WindDrag(Drag):
    """F = -k_lin·v_rel - k_quad·|v_rel|·v_rel,  v_rel = v_body - v_wind."""

    def __init__(self, wind_field,
                 linear_drag_coefficients=(0.50, 0.50, 0.75),
                 quadratic_drag_coefficients=(0.10, 0.10, 0.15)):
        super().__init__()
        self.wind = wind_field
        self.k1 = np.asarray(linear_drag_coefficients, dtype=np.float64)
        self.k2 = np.asarray(quadratic_drag_coefficients, dtype=np.float64)

    # Pegasus calls this every physics tick; signature matches LinearDrag
    def update(self, state, dt):
        v   = np.asarray(state.linear_velocity, dtype=np.float64)
        vw  = self.wind.vector(position=state.position)
        vr  = v - vw
        sp  = np.linalg.norm(vr)
        f   = -(self.k1 * vr + self.k2 * sp * vr)
        return [float(f[0]), float(f[1]), float(f[2])]