"""Rolling-element bearing characteristic (defect) frequencies."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Explicit bearing geometric parameters (no silent defaults)."""

    n_balls: int
    ball_diameter: float  # mm (or consistent length unit with pitch_diameter)
    pitch_diameter: float  # mm
    contact_angle_deg: float = 0.0  # degrees

    def validate(self) -> None:
        if not isinstance(self.n_balls, (int,)) or isinstance(self.n_balls, bool) or self.n_balls < 1:
            raise ValueError("Number of rolling elements (N) must be a positive integer >= 1.")
        for label, val in (
            ("Ball diameter (d)", self.ball_diameter),
            ("Pitch diameter (D)", self.pitch_diameter),
        ):
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"{label} must be a finite positive number.")
        if self.ball_diameter >= self.pitch_diameter:
            raise ValueError("Ball diameter (d) must be smaller than pitch diameter (D).")
        if not math.isfinite(self.contact_angle_deg) or not (0.0 <= self.contact_angle_deg < 90.0):
            raise ValueError("Contact angle (phi) must be in [0, 90) degrees.")


# Explicit preset for SKF 6205-2RS (CWRU drive-end bearing) when user selects a preset.
CWRU_SKF_6205_2RS = BearingGeometry(
    n_balls=9,
    ball_diameter=7.94,
    pitch_diameter=39.04,
    contact_angle_deg=0.0,
)

# Explicit preset for the MFPT (Machinery Failure Prevention Technology) NICE roller bearing:
# roller diameter d = 0.235 in (5.969 mm), pitch diameter D = 1.245 in (31.623 mm),
# N = 8 rolling elements, contact angle phi = 0.0 deg (nominal shaft speed = 25 Hz = 1500 RPM).
MFPT_NICE_BEARING = BearingGeometry(
    n_balls=8,
    ball_diameter=5.969,
    pitch_diameter=31.623,
    contact_angle_deg=0.0,
)


def rpm_to_hz(rpm: float | None) -> float:
    """Convert shaft rotational speed in RPM to frequency in Hz."""
    if rpm is None:
        raise ValueError("Shaft speed (RPM) must be explicitly provided and cannot be assumed.")
    rpm_val = float(rpm)
    if not math.isfinite(rpm_val) or rpm_val <= 0:
        raise ValueError(f"Shaft speed (RPM) must be a finite positive value (got {rpm_val}).")
    return rpm_val / 60.0


def bearing_fault_frequencies(
    shaft_hz: float | None = None,
    geometry: BearingGeometry | None = None,
    *,
    rpm: float | None = None,
) -> dict[str, float]:
    """Calculate theoretical kinematic bearing defect frequencies (Hz).

    Exactly one of ``shaft_hz`` or ``rpm`` must be provided, along with an
    explicit :class:`BearingGeometry`. Missing bearing parameters are never
    silently assumed.

    Formulas (fr = shaft frequency in Hz, phi in radians):
    - FTF  = (fr / 2) * (1 - (d / D) * cos(phi))
    - BPFO = (N * fr / 2) * (1 - (d / D) * cos(phi))
    - BPFI = (N * fr / 2) * (1 + (d / D) * cos(phi))
    - BSF  = (D * fr / (2 * d)) * (1 - ((d / D) * cos(phi)) ** 2)
    """
    if shaft_hz is not None and rpm is not None:
        raise ValueError("Provide either shaft_hz or rpm, not both.")
    if rpm is not None:
        fr = rpm_to_hz(rpm)
    elif shaft_hz is not None:
        fr = float(shaft_hz)
        if not math.isfinite(fr) or fr <= 0:
            raise ValueError(f"Shaft frequency (Hz) must be a finite positive value (got {fr}).")
    else:
        raise ValueError("Shaft speed (rpm or shaft_hz) must be explicitly provided.")

    if geometry is None:
        raise ValueError("BearingGeometry must be explicitly provided and cannot be assumed.")
    geometry.validate()

    phi_rad = math.radians(geometry.contact_angle_deg)
    ratio = (geometry.ball_diameter / geometry.pitch_diameter) * math.cos(phi_rad)
    n = geometry.n_balls

    ftf = (fr / 2.0) * (1.0 - ratio)
    bpfo = (n * fr / 2.0) * (1.0 - ratio)
    bpfi = (n * fr / 2.0) * (1.0 + ratio)
    bsf = (geometry.pitch_diameter * fr / (2.0 * geometry.ball_diameter)) * (1.0 - ratio**2)

    return {
        "shaft": fr,
        "FTF": ftf,
        "BPFO": bpfo,
        "BPFI": bpfi,
        "BSF": bsf,
    }
