# weather/wind_field.py
import numpy as np
import math

def wind_from_compass(speed_mps: float,
                      bearing_deg: float,
                      sigma: float = 0.0,
                      convention: str = "toward") -> dict:
    """
    Build a WindField kwargs dict from speed + compass bearing.

    bearing_deg: 0=N, 90=E, 180=S, 270=W
    convention:
      "toward" → wind blows TOWARD this bearing (math convention)
      "from"   → wind comes FROM this bearing (meteorological convention,
                 matches what a weather report says)
    """
    rad = math.radians(bearing_deg)
    sign = 1.0 if convention == "toward" else -1.0
    vx = sign * speed_mps * math.sin(rad)   # east
    vy = sign * speed_mps * math.cos(rad)   # north
    return dict(mean_mps=(vx, vy, 0.0), sigma=sigma)

class WindField:
    """Mean wind + Dryden-style turbulence + 1-cosine gusts (ENU, m/s)."""

    def __init__(self, mean_mps=(0,0,0), sigma=0.0, length_scale=200.0, seed=None):
        self.mean   = np.asarray(mean_mps, dtype=np.float64)
        self.sigma  = float(sigma)            # 1σ turbulence intensity [m/s]
        self.L      = float(length_scale)     # turbulence length scale [m]
        self._rng   = np.random.default_rng(seed)
        self._turb  = np.zeros(3)
        self._gust  = np.zeros(3)
        self._gust_amp = np.zeros(3)
        self._gust_t = 0.0
        self._gust_T = 0.0

    def trigger_gust(self, peak_mps, duration_s):
        self._gust_amp = np.asarray(peak_mps, dtype=np.float64)
        self._gust_T   = float(duration_s)
        self._gust_t   = 0.0

    def step(self, dt, airspeed_hint=8.0):
        # Dryden-style first-order filter: τ = L / V_a
        if self.sigma > 0:
            tau   = self.L / max(airspeed_hint, 0.5)
            alpha = dt / (tau + dt)
            white = self._rng.standard_normal(3) * self.sigma * np.sqrt(2*tau/dt)
            self._turb = (1-alpha) * self._turb + alpha * white
        if self._gust_t < self._gust_T:
            phase = self._gust_t / self._gust_T
            self._gust = 0.5 * self._gust_amp * (1.0 - np.cos(2*np.pi*phase))
            self._gust_t += dt
        else:
            self._gust *= 0.0

    def vector(self, position=None):     # position reserved for spatial variation
        return self.mean + self._turb + self._gust


PRESETS = {
    "calm"  : dict(mean_mps=(0,0,0),    sigma=0.0),
    "weak"  : dict(mean_mps=(2,0,0),    sigma=0.3),
    "fresh" : dict(mean_mps=(6,1,0),    sigma=1.0),
    "strong": dict(mean_mps=(12,2,0),   sigma=2.5),
    "storm" : dict(mean_mps=(15,4,0),   sigma=4.0),
}