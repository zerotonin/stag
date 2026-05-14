#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — embedded.benchmark.msp430.mspsim_runner                  ║
# ║  « run benchmark.elf in mspdebug --sim, extract cycle count »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  mspdebug ships a built-in software MSP430 simulator with a      ║
# ║  configurable Timer_A peripheral (`simio add timer`).  We wire   ║
# ║  that to the G2553's Timer_A0 register block (base 0x160) and    ║
# ║  let the firmware self-time the classifier — same pattern as     ║
# ║  the AVR Timer1 + simavr runner.                                 ║
# ║                                                                  ║
# ║  After the firmware calls return/exit, mspdebug's `setbreak      ║
# ║  exit` fires.  We read the cycles_total + n_classifications      ║
# ║  globals via `md` (memory dump) and parse the hex output.        ║
# ║                                                                  ║
# ║  Target: MSP430G2553 (TI LaunchPad value-line silicon, 16 MHz,   ║
# ║  16 KB flash, 512 B SRAM, no hardware multiplier).  Not F5529 —  ║
# ║  see the benchmark_msp430.c header for the toolchain-emulator    ║
# ║  compatibility reasoning.                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Drive mspdebug --sim to measure cycles-per-call on MSP430G2553."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


THIS_DIR  = Path(__file__).resolve().parent
BUILD_DIR = THIS_DIR / "build"

# G2553 Timer_A0 register block; the simio timer model needs the
# base address so its registers land at the correct sysbus offsets.
G2553_TIMER_A0_BASE = 0x160
G2553_TIMER_A0_IV   = 0x12e

TI_TOOLCHAIN_BIN    = Path.home() / "ti" / "msp430-gcc" / "bin"
DEFAULT_NM          = TI_TOOLCHAIN_BIN / "msp430-elf-nm"


@dataclass
class McuTarget:
    name:    str       # subdir under stag/embedded/benchmark/msp430/build/
    freq_hz: int       # silicon nominal clock for ns/call conversion
    symbols: tuple[str, ...] = field(
        default_factory=lambda: ("cycles_total", "n_classifications"),
    )

    @property
    def elf(self) -> Path:
        return BUILD_DIR / self.name / "benchmark.elf"


# MSP430G2553: TI LaunchPad value-line, 16 MHz max via DCO+XT2.
DEFAULT_TARGETS: list[McuTarget] = [
    McuTarget("msp430g2553", 16_000_000),
]


_NM_LINE_RE = re.compile(r"^([0-9a-fA-F]+)\s+\S+\s+(\S+)$")
_MD_LINE_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+([0-9a-f ]{11,})", re.MULTILINE)


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(
            f"Required tool '{tool}' not on PATH.  Install mspdebug "
            f"(`sudo apt install mspdebug`) and the TI MSP430-GCC.",
        )
    return path


def _symbol_addresses(elf: Path, names: tuple[str, ...]) -> dict[str, int]:
    proc = subprocess.run(
        [str(DEFAULT_NM), str(elf)], capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"msp430-elf-nm failed:\n{proc.stderr}")
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
        raise SystemExit(f"msp430-elf-nm did not find symbols: {missing}")
    return out


def _build_all(targets: list[McuTarget]) -> None:
    print("── make ──")
    proc = subprocess.run(
        ["make", *[t.name for t in targets]],
        cwd=THIS_DIR, capture_output=True, text=True,
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


def _mspdebug_script(addrs: dict[str, int]) -> str:
    """Return the stdin script for `mspdebug -q sim`.

    Installs a Timer_A0 peripheral at the G2553's register block, loads
    the ELF, runs to the `exit` symbol breakpoint, and dumps the two
    result globals as raw 4-byte little-endian hex via `md`.
    """
    cyc_addr = addrs["cycles_total"]
    n_addr   = addrs["n_classifications"]
    return "\n".join([
        f"simio add timer ta0 3",
        f"simio config ta0 base 0x{G2553_TIMER_A0_BASE:x}",
        f"simio config ta0 iv 0x{G2553_TIMER_A0_IV:x}",
        f"prog {addrs['__elf']}",
        f"setbreak exit",
        f"run",
        f"md 0x{cyc_addr:x} 4",
        f"md 0x{n_addr:x} 2",
        f"exit",
        "",  # trailing newline
    ])


def _parse_uint(md_output: str, addr: int, width_bytes: int) -> int:
    """Pluck a little-endian unsigned int from mspdebug's `md` dump.

    mspdebug prints lines like::

        00204: 13 e9 1e 00                                     |....            |

    where the bytes after the colon are the raw memory contents in
    little-endian order.  We grep the line whose address matches,
    parse the requested number of bytes, and assemble the integer.
    """
    target = f"{addr:05x}"
    for m in _MD_LINE_RE.finditer(md_output):
        if m.group(1).lower() != target:
            continue
        byte_str = m.group(2).strip().split()
        if len(byte_str) < width_bytes:
            continue
        val = 0
        for i in range(width_bytes):
            val |= int(byte_str[i], 16) << (8 * i)
        return val
    raise SystemExit(
        f"Failed to parse memory at 0x{addr:x} from mspdebug output:\n"
        f"{md_output}",
    )


def _run_one(target: McuTarget, mspdebug_path: str) -> dict:
    print(f"── {target.name} ── (F_CPU={target.freq_hz/1e6:.0f} MHz)")
    if not target.elf.is_file():
        raise SystemExit(f"ELF missing: {target.elf}")
    addrs = _symbol_addresses(target.elf, target.symbols)
    addrs["__elf"] = str(target.elf)

    script = _mspdebug_script(addrs)
    proc = subprocess.run(
        [mspdebug_path, "-q", "sim"],
        input=script, capture_output=True, text=True, timeout=60.0,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"mspdebug failed (rc={proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}",
        )

    md_output = proc.stdout
    cycles = _parse_uint(md_output, addrs["cycles_total"],      width_bytes=4)
    n      = _parse_uint(md_output, addrs["n_classifications"], width_bytes=2)

    if cycles <= 0 or n <= 0:
        raise SystemExit(
            f"Invalid measurement: cycles={cycles}, n={n}.\n"
            f"mspdebug output:\n{md_output}",
        )

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
    p.add_argument("--out", type=Path, default=THIS_DIR / "results_msp430.csv")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--mspdebug", type=str, default="mspdebug")
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mspdebug_path = _require(args.mspdebug)
    if not DEFAULT_NM.is_file():
        raise SystemExit(
            f"msp430-elf-nm not found at {DEFAULT_NM}.  "
            f"Install the TI MSP430-GCC toolchain.",
        )

    targets = DEFAULT_TARGETS
    if args.targets:
        names = set(args.targets)
        targets = [t for t in targets if t.name in names]
        if not targets:
            raise SystemExit(f"no matching targets in {args.targets}")

    if not args.skip_build:
        _build_all(targets)

    rows = [_run_one(t, mspdebug_path) for t in targets]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
