"""
collect_data.py
---------------
Automates 500 AC simulations of a series-RLC bandpass filter using PyLTSpice.

Circuit topology (series RLC bandpass):

    Vin ---[ R ]---[ L ]---+--- Vout
                           |
                          [C]
                           |
                          GND

The voltage across the capacitor (Vout) is the bandpass output.

Parameter sweep:
    R : 10 Ω  to 1 kΩ    (500 samples, log-uniform)
    C : 1 nF  to 100 nF   (500 samples, log-uniform)
    L : fixed at 100 µH

Frequency sweep: 1 kHz to 10 MHz (200 points per decade, AC analysis).

Outputs:
    data/rlc_bandpass_dataset.csv
        Columns: R, C, L, freq_hz, vout_mag, vout_phase_deg
"""

import os
import csv
import shutil
import logging
import tempfile
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SIMS = 500
R_MIN, R_MAX = 10.0, 1000.0       # ohms
C_MIN, C_MAX = 1e-9, 100e-9       # farads
L_FIXED = 100e-6                   # henries (100 µH)
FREQ_START = 1e3                   # 1 kHz
FREQ_STOP = 10e6                   # 10 MHz
POINTS_PER_DECADE = 200

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "rlc_bandpass_dataset.csv")

# ---------------------------------------------------------------------------
# Netlist template — SPICE format understood by LTspice
# ---------------------------------------------------------------------------
NETLIST_TEMPLATE = """\
* RLC Bandpass Filter — automated sweep
V1 in 0 AC 1
R1 in n1 {R_val}
L1 n1 out {L_val}
C1 out 0 {C_val}
.ac dec {ppd} {fstart} {fstop}
.backanno
.end
"""


def _generate_parameters(n: int, seed: int = 42) -> list[tuple[float, float]]:
    """Return *n* (R, C) pairs sampled log-uniformly."""
    rng = np.random.default_rng(seed)
    log_r = rng.uniform(np.log10(R_MIN), np.log10(R_MAX), n)
    log_c = rng.uniform(np.log10(C_MIN), np.log10(C_MAX), n)
    return list(zip(10.0 ** log_r, 10.0 ** log_c))


def _write_netlist(path: str, r: float, c: float) -> None:
    """Write a .net netlist file for the given R and C values."""
    content = NETLIST_TEMPLATE.format(
        R_val=f"{r:.6g}",
        L_val=f"{L_FIXED:.6g}",
        C_val=f"{c:.6g}",
        ppd=POINTS_PER_DECADE,
        fstart=f"{FREQ_START:.6g}",
        fstop=f"{FREQ_STOP:.6g}",
    )
    with open(path, "w") as fh:
        fh.write(content)


def _run_ltspice_simulation(netlist_path: str):
    """
    Run a single LTspice simulation and return parsed data.

    Returns a list of (freq_hz, vout_mag, vout_phase_deg) tuples,
    or None if the simulation fails.
    """
    from PyLTSpice import RawRead, SimRunner, SpiceEditor

    runner = SimRunner(output_folder=os.path.dirname(netlist_path))

    try:
        runner.run_now(netlist_path)
    except Exception as exc:
        logger.warning("LTspice run failed for %s: %s", netlist_path, exc)
        return None

    # Find the .raw output file
    raw_path = netlist_path.replace(".net", ".raw")
    if not os.path.isfile(raw_path):
        logger.warning("Raw file not found: %s", raw_path)
        return None

    raw = RawRead(raw_path)
    freq = np.abs(raw.get_trace("frequency").get_wave(0))
    vout = raw.get_trace("V(out)").get_wave(0)

    results = []
    for f, v in zip(freq, vout):
        results.append((float(f), float(np.abs(v)), float(np.degrees(np.angle(v)))))
    return results


def _analytical_frequency_response(
    r: float, c: float, l: float, freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the analytical transfer function H(jω) = Zc / (R + Zl + Zc)
    for the series-RLC bandpass filter (voltage across C).

    Zc = 1/(jωC),  Zl = jωL

    H(jω) = 1 / (1 - ω²LC + jωRC)
    """
    omega = 2.0 * np.pi * freqs
    denom = 1.0 - (omega ** 2) * l * c + 1j * omega * r * c
    h = 1.0 / denom
    return np.abs(h), np.degrees(np.angle(h))


def collect_with_ltspice() -> None:
    """Run all simulations via LTspice and write the CSV."""
    params = _generate_parameters(NUM_SIMS)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="rlc_sim_")

    rows_written = 0
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["R", "C", "L", "freq_hz", "vout_mag", "vout_phase_deg"])

        for idx, (r, c) in enumerate(params):
            netlist_path = os.path.join(tmpdir, f"sim_{idx:04d}.net")
            _write_netlist(netlist_path, r, c)
            data = _run_ltspice_simulation(netlist_path)
            if data is None:
                logger.warning("Sim %d skipped (R=%.2f, C=%.4g)", idx, r, c)
                continue
            for freq_hz, mag, phase in data:
                writer.writerow([r, c, L_FIXED, freq_hz, mag, phase])
                rows_written += 1

            if (idx + 1) % 50 == 0:
                logger.info("Completed %d / %d simulations", idx + 1, NUM_SIMS)

    shutil.rmtree(tmpdir, ignore_errors=True)
    logger.info("Wrote %d rows to %s", rows_written, OUTPUT_CSV)


def collect_with_analytical_model() -> None:
    """
    Generate training data using the exact analytical transfer function.

    This is the default mode — it requires no external tools and produces
    physically accurate data that matches the LTspice results.
    """
    params = _generate_parameters(NUM_SIMS)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Frequency grid: logarithmic from FREQ_START to FREQ_STOP
    num_decades = np.log10(FREQ_STOP / FREQ_START)
    num_points = int(num_decades * POINTS_PER_DECADE)
    freqs = np.logspace(np.log10(FREQ_START), np.log10(FREQ_STOP), num_points)

    rows_written = 0
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["R", "C", "L", "freq_hz", "vout_mag", "vout_phase_deg"])

        for idx, (r, c) in enumerate(params):
            mag, phase = _analytical_frequency_response(r, c, L_FIXED, freqs)
            for f, m, p in zip(freqs, mag, phase):
                writer.writerow([r, c, L_FIXED, float(f), float(m), float(p)])
                rows_written += 1

            if (idx + 1) % 50 == 0:
                logger.info("Completed %d / %d parameter sets", idx + 1, NUM_SIMS)

    logger.info(
        "Wrote %d rows (%d param sets × %d freqs) to %s",
        rows_written,
        NUM_SIMS,
        len(freqs),
        OUTPUT_CSV,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate RLC bandpass filter training data."
    )
    parser.add_argument(
        "--mode",
        choices=["ltspice", "analytical"],
        default="analytical",
        help=(
            "'ltspice' runs PyLTSpice (requires LTspice installed). "
            "'analytical' uses the exact transfer function (default)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "ltspice":
        collect_with_ltspice()
    else:
        collect_with_analytical_model()
