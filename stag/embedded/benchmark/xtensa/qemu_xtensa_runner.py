#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — embedded.benchmark.xtensa.qemu_xtensa_runner              ║
# ║  « run benchmark.elf in Espressif's QEMU fork, parse UART »      ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The Espressif xtensa-esp-elf toolchain ships gcc + binutils but ║
# ║  no gdb — and gdb-multiarch's register layout does not match     ║
# ║  Espressif QEMU's `g` packet for Xtensa.  So instead of driving  ║
# ║  the run with a gdb attach, the firmware self-prints its result ║
# ║  to UART0 (TX FIFO at 0x3FF40000) which QEMU's `-serial file:`   ║
# ║  redirects to a host-side text file we regex.                    ║
# ║                                                                  ║
# ║  Two QEMU flags are critical:                                    ║
# ║    -machine esp32       : actual ESP32 board model with the      ║
# ║                          right peripheral set + memory map       ║
# ║    -icount shift=auto  : makes CCOUNT advance deterministically  ║
# ║                          across runs (without it, results vary    ║
# ║                          by ~3× between runs based on host load) ║
# ║                                                                  ║
# ║  No gdb in the loop — keeps the runner ~150 lines and fast.      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Drive QEMU + parse UART output for ESP32 (Xtensa LX6) benchmark."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


THIS_DIR  = Path(__file__).resolve().parent
QEMU_DEFAULT_PATH = Path.home() / "opt" / "qemu" / "bin" / "qemu-system-xtensa"
TIMEOUT_S = 30.0


@dataclass
class McuTarget:
    name:    str       # subdir under stag/embedded/benchmark/xtensa/<name>/
    freq_hz: int       # silicon nominal clock for ns/call conversion

    @property
    def target_dir(self) -> Path:
        return THIS_DIR / self.name

    @property
    def elf(self) -> Path:
        return self.target_dir / "build" / self.name / "benchmark.elf"


# ESP32 (Xtensa LX6) — single core, 240 MHz default clock when PLL
# is up (we don't bother configuring the PLL in firmware; the cycle
# count we measure is virtual-CCOUNT under QEMU icount=auto, which
# tracks the host's resolution and is independent of the CPU clock
# parameter — the 240 MHz figure is only used for ns/call reporting).
DEFAULT_TARGETS: list[McuTarget] = [
    McuTarget("esp32", 240_000_000),
]


# The firmware emits a single line of the form
#   STAG_RESULT cycles=0xXXXXXXXX n=0xXX
# after the benchmark loop completes.  We regex it out of the
# captured UART file.
_RESULT_RE = re.compile(
    r"STAG_RESULT\s+cycles=0x([0-9a-fA-F]+)\s+n=0x([0-9a-fA-F]+)",
)


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(
            f"Required tool '{tool}' not on PATH.  Install QEMU "
            f"(Espressif fork) and the xtensa-esp-elf toolchain.",
        )
    return path


def _build_all(targets: list[McuTarget]) -> None:
    print("── make ──")
    for t in targets:
        proc = subprocess.run(
            ["make", t.name],
            cwd=t.target_dir, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(
                f"make failed for {t.name}:\n--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}",
            )
        if not t.elf.is_file():
            raise SystemExit(f"missing ELF after make: {t.elf}")
    print(f"  built {len(targets)} ELFs")


def _run_one(target: McuTarget, qemu_path: Path, timeout_s: float) -> dict:
    print(f"── {target.name} ── (F_CPU={target.freq_hz/1e6:.0f} MHz)")
    if not target.elf.is_file():
        raise SystemExit(f"ELF missing: {target.elf}")

    uart_log = Path(tempfile.gettempdir()) / f"qemu_uart_{target.name}.txt"
    uart_log.write_text("")

    qemu_proc = subprocess.Popen(
        [
            str(qemu_path),
            "-machine", "esp32",
            "-cpu",     "esp32",
            "-m",       "4M",
            "-display", "none",
            "-kernel",  str(target.elf),
            "-serial",  f"file:{uart_log}",
            "-icount",  "shift=auto",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + timeout_s
        match = None
        while time.time() < deadline:
            if qemu_proc.poll() is not None:
                # QEMU exited unexpectedly — read whatever was captured
                break
            text = uart_log.read_text()
            match = _RESULT_RE.search(text)
            if match is not None:
                break
            time.sleep(0.1)
        if match is None:
            text = uart_log.read_text()
            raise SystemExit(
                f"QEMU did not emit STAG_RESULT line within {timeout_s:.0f} s.\n"
                f"UART buffer ({len(text)} bytes):\n{text!r}",
            )
    finally:
        qemu_proc.terminate()
        try:
            qemu_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            qemu_proc.kill()

    cycles = int(match.group(1), 16)
    n      = int(match.group(2), 16)
    if cycles <= 0 or n <= 0:
        raise SystemExit(f"Invalid measurement: cycles={cycles}, n={n}.")

    cpc = cycles / n
    ns_per = (cycles / target.freq_hz) * 1e9 / n
    print(f"  cycles_total = {cycles:>12,d}")
    print(f"  n            = {n}")
    print(f"  cycles/call  = {cpc:>12,.1f}")
    print(f"  ns/call      = {ns_per:>12,.0f}   (at {target.freq_hz/1e6:.0f} MHz)")
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
    p.add_argument("--out", type=Path, default=THIS_DIR / "results_xtensa.csv")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--qemu", type=Path, default=QEMU_DEFAULT_PATH,
                   help="QEMU launcher path (Espressif fork).")
    p.add_argument("--timeout", type=float, default=TIMEOUT_S)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.qemu.is_file():
        raise SystemExit(
            f"QEMU launcher not found at {args.qemu}.  Pass --qemu "
            f"explicitly or install Espressif's QEMU fork there.",
        )
    _require("xtensa-esp32-elf-gcc")

    targets = DEFAULT_TARGETS
    if args.targets:
        names = set(args.targets)
        targets = [t for t in targets if t.name in names]
        if not targets:
            raise SystemExit(f"no matching targets in {args.targets}")

    if not args.skip_build:
        _build_all(targets)

    rows = [_run_one(t, args.qemu, args.timeout) for t in targets]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
