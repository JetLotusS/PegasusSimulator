from pathlib import Path
from pxr import Gf, Sdf, UsdLux

SKY_DIR = Path(__file__).parent.parent.parent / "assets" / "Skies"

def _hdr(name):
    p = SKY_DIR / name
    if p.exists():
        print(f"{str(p)} found")
    else:
        print(f"{str(p)} not found")
    return str(p) if p.exists() else None

SKY_PRESETS = {
    "clear"    : {"texture": _hdr("clear.hdr"),    "intensity": 1500.0, "color": (1.00, 1.00, 1.00), "exposure":  0.0},
    "cloudy"   : {"texture": _hdr("cloudy.hdr"),   "intensity": 700.0, "color": (0.95, 0.97, 1.00), "exposure":  0.0},
    "overcast" : {"texture": _hdr("overcast.hdr"), "intensity":  200.0, "color": (0.88, 0.91, 0.95), "exposure":  0.0},
    "stormy"   : {"texture": _hdr("stormy.hdr"),   "intensity":  100.0, "color": (0.65, 0.70, 0.82), "exposure": -1.0},
    "sunset"   : {"texture": _hdr("sunset.hdr"),   "intensity": 1200.0, "color": (1.00, 0.82, 0.65), "exposure":  0.0},
    "night"    : {"texture": None,                  "intensity":   80.0, "color": (0.20, 0.25, 0.40), "exposure":  0.0},
}

def apply_sky(stage, preset_name: str, dome_path: str = "/World/SkyDome"):
    cfg = SKY_PRESETS[preset_name]
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr().Set(float(cfg["intensity"]))
    dome.CreateColorAttr().Set(Gf.Vec3f(*cfg["color"]))
    dome.CreateExposureAttr().Set(float(cfg["exposure"]))
    if cfg["texture"]:
        dome.CreateTextureFileAttr().Set(Sdf.AssetPath(cfg["texture"]))
        dome.CreateTextureFormatAttr().Set("latlong")
    else:
        dome.CreateTextureFileAttr().Set(Sdf.AssetPath(""))
    return dome