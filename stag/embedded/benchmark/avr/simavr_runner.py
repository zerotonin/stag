#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — embedded.benchmark.avr.simavr_runner                     ║
# ║  « run benchmark.elf in simavr, extract cycle counts via gdb »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  For each AVR target:                                            ║
# ║    1. Launch simavr -g -m <mcu> -f <hz> benchmark.elf            ║
# ║       (-g makes simavr wait for a gdb attach on port 1234).      ║
# ║    2. Drive avr-gdb in batch mode: connect, break on            ║
# ║       ``bench_done``, continue, print ``cycles_total`` and      ║
# ║       ``n_classifications``, detach.                            ║
# ║    3. Compute cycles per classification and ns per classification║
# ║       (assuming the nominal F_CPU the firmware was built with). ║
# ║                                                                  ║
# ║  Output: one CSV row per MCU with columns                       ║
# ║    mcu, freq_hz, n_classifications, cycles_total,               ║
# ║    cycles_per_classification, ns_per_classification.            ║
# ║                                                                  ║
# ║  Requires: avr-gcc, simavr, avr-gdb on PATH.  Skipped (returns  ║
# ║  empty CSV) if any of the three is missing — keeps the test    ║
# ║  suite usable in toolchain-less CI environments.                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Drive simavr + avr-gdb to measure cycles-per-classification per AVR MCU."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_F_CPU = 16_000_000
DEFAULT_GDB_PORT = 1234
DEFAULT_TIMEOUT_S = 60.0

THIS_DIR  = Path(__file__).resolve().parent
BUILD_DIR = THIS_DIR / "build"


@dataclass
class McuTarget:
    name:    str
    freq_hz: int = DEFAULT_F_CPU

    @property
    def elf(self) -> Path:
        return BUILD_DIR / self.name / "benchmark.elf"


DEFAULT_TARGETS: list[McuTarget] = [
    McuTarget("atmega328p", 16_000_000),
    McuTarget("atmega32u4", 16_000_000),
    McuTarget("atmega2560", 16_000_000),
]


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(
            f"Required tool '{tool}' not on PATH.  Install the AVR toolchain "
            f"first: sudo apt install gcc-avr avr-libc simavr gdb-avr",
        )
    return path


def _build_all(targets: list[McuTarget]) -> None:
    """Ensure every target's ELF is built (Make handles the up-to-date check)."""
    print("── make ──")
    proc = subprocess.run(
        ["make", "all"], cwd=THIS_DIR,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"make failed:\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}",
        )
    for t in targets:
        if not t.elf.is_file():
            raise SystemExit(f"missing ELF after make: {t.elf}")
    print(f"  built {len(targets)} ELFs")


_NM_LINE_RE = re.compile(r"^([0-9a-fA-F]+)\s+\S+\s+(\S+)$")


def _symbol_addresses(elf: Path, names: list[str]) -> dict[str, int]:
    """Extract symbol addresses from an AVR ELF via avr-nm.

    We avoid asking gdb to load the ELF (DWARF in avr-gcc 7.x ELFs
    crashes avr-gdb 15.x with a create_range_type assertion).  The
    addresses we get here go straight into raw-memory-read commands.
    """
    nm = _require("avr-nm")
    proc = subprocess.run([nm, str(elf)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"avr-nm failed:\n{proc.stderr}")
    out: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        m = _NM_LINE_RE.match(line)
        if not m:
            continue
        addr_hex, name = m.group(1), m.group(2)
        if name in names:
            out[name] = int(addr_hex, 16)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"avr-nm did not find symbols: {missing}")
    return out


def _gdb_script(addrs: dict[str, int], port: int) -> str:
    """Generate the avr-gdb batch script that drives simavr.

    Connects to simavr, breaks at ``bench_done``, reads
    ``cycles_total`` + ``n_classifications`` by raw address, quits.

    We do NOT 'file' the ELF — see :func:`_symbol_addresses` for why.
    Memory addresses are in the AVR-gdb data-space convention
    (RAM offsets carry the 0x800000 high bit).
    """
    return (
        f"set confirm off\n"
        f"set pagination off\n"
        f"target remote :{port}\n"
        f"break *0x{addrs['bench_done']:x}\n"
        f"continue\n"
        f"x/1wx 0x{addrs['cycles_total']:x}\n"
        f"x/1wx 0x{addrs['n_classifications']:x}\n"
        f"quit\n"
    )


# Matches lines like:  0x800105:	0x00363c01
_MEM_LINE_RE = re.compile(r"0x([0-9a-fA-F]+):\s+0x([0-9a-fA-F]+)")


def _run_one(
    target: McuTarget,
    simavr: str,
    avr_gdb: str,
    port: int,
    timeout_s: float,
) -> dict:
    print(f"── {target.name} ── (F_CPU={target.freq_hz/1e6:.0f} MHz)")
    if not target.elf.is_file():
        raise SystemExit(f"ELF missing: {target.elf}")

    addrs = _symbol_addresses(
        target.elf, ["bench_done", "cycles_total", "n_classifications"],
    )

    # Spawn simavr in its own process group so we can kill it cleanly.
    simavr_proc = subprocess.Popen(
        [simavr, "-g", "-m", target.name, "-f", str(target.freq_hz),
         str(target.elf)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        preexec_fn=os.setpgrp,
    )
    try:
        time.sleep(0.3)
        if simavr_proc.poll() is not None:
            err = (simavr_proc.stderr.read() or b"").decode(errors="replace")
            raise SystemExit(
                f"simavr exited before gdb attach (rc={simavr_proc.returncode}):\n{err}",
            )

        # avr-gdb 15.x does not read commands from stdin in batch
        # mode — pass via a temp script file with -x instead.
        with tempfile.NamedTemporaryFile("w", suffix=".gdb",
                                        delete=False) as f:
            f.write(_gdb_script(addrs, port))
            gdb_script_path = f.name
        try:
            gdb_proc = subprocess.run(
                [avr_gdb, "--batch", "--quiet", "-nx", "-nh",
                 "-x", gdb_script_path],
                capture_output=True, text=True, timeout=timeout_s,
            )
        finally:
            os.unlink(gdb_script_path)
        if gdb_proc.returncode != 0:
            raise SystemExit(
                f"avr-gdb failed (rc={gdb_proc.returncode}):\n"
                f"--- stdout ---\n{gdb_proc.stdout}\n"
                f"--- stderr ---\n{gdb_proc.stderr}",
            )

        # Parse two memory-read lines.  Order matches the gdb script.
        mem_reads = [
            (int(m.group(1), 16), int(m.group(2), 16))
            for m in _MEM_LINE_RE.finditer(gdb_proc.stdout)
        ]
        wanted = {addrs["cycles_total"], addrs["n_classifications"]}
        observed = {a for a, _ in mem_reads}
        if not wanted.issubset(observed):
            raise SystemExit(
                f"gdb output missing expected memory reads:\n"
                f"  wanted addresses: {[hex(a) for a in wanted]}\n"
                f"  saw:              {[hex(a) for a in observed]}\n"
                f"  full stdout:\n{gdb_proc.stdout}",
            )
        cycles = next(v for a, v in mem_reads if a == addrs["cycles_total"])
        n      = next(v for a, v in mem_reads if a == addrs["n_classifications"])

    finally:
        # Tear down simavr.
        if simavr_proc.poll() is None:
            try:
                os.killpg(simavr_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                simavr_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(simavr_proc.pid, signal.SIGKILL)

    cpc = cycles / n
    ns_per = (cycles / target.freq_hz) * 1e9 / n
    print(f"  cycles_total = {cycles:>10,d}")
    print(f"  n            = {n}")
    print(f"  cycles/call  = {cpc:>10,.1f}")
    print(f"  ns/call      = {ns_per:>10,.0f}  (at {target.freq_hz/1e6:.0f} MHz)")
    return {
        "mcu":                       target.name,
        "freq_hz":                   target.freq_hz,
        "n_classifications":         n,
        "cycles_total":              cycles,
        "cycles_per_classification": cpc,
        "ns_per_classification":     ns_per,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=THIS_DIR / "results_avr.csv",
                   help="Output CSV path.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of MCU names to run (default: all three).")
    p.add_argument("--port", type=int, default=DEFAULT_GDB_PORT,
                   help="Port for the simavr gdb stub (default 1234).")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help="Per-target wall-clock timeout in seconds.")
    p.add_argument("--skip-build", action="store_true",
                   help="Skip the make step (use existing ELFs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    simavr  = _require("simavr")
    avr_gdb = _require("avr-gdb")
    _require("avr-gcc")  # make would fail anyway, but front-load the diagnosis

    targets = DEFAULT_TARGETS
    if args.targets:
        names = set(args.targets)
        targets = [t for t in targets if t.name in names]
        if not targets:
            raise SystemExit(f"no matching targets in {args.targets}")

    if not args.skip_build:
        _build_all(targets)

    rows = []
    for t in targets:
        rows.append(_run_one(t, simavr, avr_gdb, args.port, args.timeout))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
