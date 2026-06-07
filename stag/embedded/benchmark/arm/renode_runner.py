#!/usr/bin/env python
# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — embedded.benchmark.arm.renode_runner                     ║
# ║  « run benchmark.elf in Renode, extract cycle counts via gdb »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Mirrors the AVR runner's design: emulator launches with a       ║
# ║  gdb stub, runner attaches with gdb in batch mode, sets a       ║
# ║  breakpoint on bench_done, continues, reads the result          ║
# ║  globals via raw memory addresses (extracted with                ║
# ║  arm-none-eabi-nm), tells the emulator to exit.                  ║
# ║                                                                  ║
# ║  Renode 1.15.3's `machine StartGdbServer` defaults to port 3333.║
# ║  We use gdb-multiarch on Ubuntu 24.04 since the standalone       ║
# ║  arm-none-eabi-gdb is not packaged on noble.                     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Drive Renode + gdb-multiarch to measure cycles-per-call on ARM MCUs."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_GDB_PORT       = 3333
DEFAULT_TIMEOUT_S      = 90.0
RENODE_DEFAULT_PATH    = Path.home() / "opt" / "renode_1.15.3_portable" / "renode"

THIS_DIR  = Path(__file__).resolve().parent
BUILD_DIR = THIS_DIR / "build"


@dataclass
class McuTarget:
    name:    str       # subdir under stag/embedded/benchmark/arm/<name>/
    freq_hz: int       # nominal default clock as Renode emulates it
    nm_tool: str = "arm-none-eabi-nm"
    # symbols whose value addresses we read.  Same for every Cortex-M
    # target since the firmware is shared.
    symbols: tuple[str, ...] = field(
        default_factory=lambda: (
            "bench_start", "bench_done", "n_classifications",
        ),
    )

    @property
    def elf(self) -> Path:
        return BUILD_DIR / self.name / "benchmark.elf"

    @property
    def resc(self) -> Path:
        return THIS_DIR / self.name / "run.resc"


# Nominal CPU clocks per silicon datasheet (used only to convert
# cycles → ns for reporting; Renode's ExecutedInstructions counter
# is clock-independent).
DEFAULT_TARGETS: list[McuTarget] = [
    McuTarget("samd21",         48_000_000),   # Cortex-M0+ (SAMD21G18A)
    McuTarget("nrf52840",       64_000_000),   # Cortex-M4F
    McuTarget("rp2040",        133_000_000),   # Cortex-M0+ (Raspberry Pi Pico)
    McuTarget("stm32f4_disco", 168_000_000),   # Cortex-M4F (STM32F407)
    McuTarget("imxrt1064",     600_000_000),   # Cortex-M7
]


_NM_LINE_RE = re.compile(r"^([0-9a-fA-F]+)\s+\S+\s+(\S+)$")
_MEM_LINE_RE = re.compile(r"0x([0-9a-fA-F]+):\s+0x([0-9a-fA-F]+)")


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise SystemExit(
            f"Required tool '{tool}' not on PATH.  "
            f"Install the ARM toolchain + Renode + gdb-multiarch.",
        )
    return path


def _symbol_addresses(elf: Path, nm_tool: str, names: tuple[str, ...]) -> dict[str, int]:
    proc = subprocess.run([nm_tool, str(elf)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{nm_tool} failed:\n{proc.stderr}")
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
        raise SystemExit(f"{nm_tool} did not find symbols: {missing}")
    return out


def _build_all(targets: list[McuTarget]) -> None:
    print("── make ──")
    target_names = " ".join(t.name for t in targets)
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


def _gdb_script(addrs: dict[str, int], port: int) -> str:
    """Connect to Renode's gdb stub, break twice on bench_done, read
    `cpu ExecutedInstructions` at each.  The firmware loops the
    benchmark in an outer infinite loop so two consecutive hits of
    bench_done bracket exactly one complete N-classification run.

    Renode 1.15.3's FPB emulation only honours one of the two hbreaks
    reliably for this CPU model — using a single breakpoint at
    bench_done sidesteps the issue.

    Renode's `monitor cpu ExecutedInstructions` prints a 0x-prefixed
    hex value (e.g. ``0x000000000B7E3CE5``).
    """
    return (
        "set confirm off\n"
        "set pagination off\n"
        "set architecture arm\n"
        f"target remote :{port}\n"
        f"hbreak *0x{addrs['bench_done']:x}\n"
        "continue\n"
        "printf \"STAG_BENCH_AT_DONE_1\\n\"\n"
        "monitor cpu ExecutedInstructions\n"
        "continue\n"
        "printf \"STAG_BENCH_AT_DONE_2\\n\"\n"
        "monitor cpu ExecutedInstructions\n"
        f"x/1wx 0x{addrs['n_classifications']:x}\n"
        "monitor quit\n"
        "quit\n"
    )


# Renode's `monitor cpu ExecutedInstructions` output comes back via
# the gdb remote `qRcmd` reply channel, which gdb forwards to its
# stderr — not stdout.  Capture every standalone 0x-prefixed hex
# line from stderr in order: hits 1 and 2 are our two readings.
_HEX_LINE_RE = re.compile(r"^\s*0x([0-9a-fA-F]+)\s*$", re.MULTILINE)


def _run_one(
    target: McuTarget,
    renode_path: Path,
    gdb_tool: str,
    port: int,
    timeout_s: float,
) -> dict:
    print(f"── {target.name} ── (F_CPU={target.freq_hz/1e6:.0f} MHz)")
    if not target.elf.is_file():
        raise SystemExit(f"ELF missing: {target.elf}")
    if not target.resc.is_file():
        raise SystemExit(f"Renode script missing: {target.resc}")

    addrs = _symbol_addresses(target.elf, target.nm_tool, target.symbols)

    # Renode 1.15.3's `-e` runs AFTER the positional script — so a
    # ``$bin=@...`` injection never gets read by the .resc.  Render
    # the .resc template into a temp file with absolute paths baked
    # in for $bin (the ELF) and $repl (the per-target platform file
    # when the target ships its own — used for chips Renode does
    # not bundle a .repl for, e.g. SAMD21 / RP2040).
    resc_template = target.resc.read_text()
    rendered_resc = resc_template.replace("$bin", f"@{target.elf}")
    custom_repl = THIS_DIR / target.name / f"{target.name}.repl"
    if custom_repl.is_file():
        rendered_resc = rendered_resc.replace("$repl", f"@{custom_repl}")
    with tempfile.NamedTemporaryFile("w", suffix=".resc",
                                    delete=False) as f:
        f.write(rendered_resc)
        resc_path = f.name

    # Redirect Renode's stdout/stderr to a log file (avoids deadlocks
    # on the PIPE).  Keep stdin attached to a PIPE we never close —
    # otherwise Renode exits the moment the .resc script ends, even
    # though the CPU is still emulating.
    renode_log = Path(tempfile.gettempdir()) / f"renode_{target.name}.log"
    renode_log_fh = open(renode_log, "w")
    renode_proc = subprocess.Popen(
        [str(renode_path), "--disable-xwt", "--console", resc_path],
        stdin=subprocess.PIPE,
        stdout=renode_log_fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,
    )
    try:
        # Poll Renode's log file for the gdb-server-ready line.
        deadline = time.time() + 30.0
        gdb_ready_re = re.compile(
            rf"GDB server.*started on port :{port}\b",
        )
        while True:
            if renode_proc.poll() is not None:
                log_text = renode_log.read_text() if renode_log.is_file() else ""
                raise SystemExit(
                    f"Renode exited before gdb stub opened "
                    f"(rc={renode_proc.returncode}):\n{log_text}",
                )
            if renode_log.is_file():
                text = renode_log.read_text()
                if gdb_ready_re.search(text):
                    break
            if time.time() > deadline:
                log_text = renode_log.read_text() if renode_log.is_file() else ""
                raise SystemExit(
                    f"Renode gdb stub never opened on port {port} "
                    f"after 30 s.  Log:\n{log_text}",
                )
            time.sleep(0.2)
        # Give the listener a beat to fully accept connections.
        time.sleep(0.3)

        with tempfile.NamedTemporaryFile("w", suffix=".gdb", delete=False) as f:
            f.write(_gdb_script(addrs, port))
            gdb_script_path = f.name
        try:
            gdb_proc = subprocess.run(
                [gdb_tool, "--batch", "--quiet", "-nx", "-nh",
                 "-x", gdb_script_path],
                capture_output=True, text=True, timeout=timeout_s,
            )
        finally:
            os.unlink(gdb_script_path)

        if gdb_proc.returncode != 0:
            raise SystemExit(
                f"gdb failed (rc={gdb_proc.returncode}):\n"
                f"--- stdout ---\n{gdb_proc.stdout}\n"
                f"--- stderr ---\n{gdb_proc.stderr}",
            )

        # The `monitor cpu ExecutedInstructions` output comes back via
        # the qRcmd reply, which gdb routes to its stderr (NOT stdout).
        # We unconditionally pull every standalone 0x-line from stderr
        # and take the first two — that is our two readings in order.
        hex_readings = _HEX_LINE_RE.findall(gdb_proc.stderr)
        if len(hex_readings) < 2:
            raise SystemExit(
                f"expected 2 hex readings on gdb stderr, got "
                f"{len(hex_readings)}.\n"
                f"--- stderr ---\n{gdb_proc.stderr}\n"
                f"--- stdout ---\n{gdb_proc.stdout}",
            )
        cycles_at_start = int(hex_readings[0], 16)
        cycles_at_done  = int(hex_readings[1], 16)
        cycles = cycles_at_done - cycles_at_start
        if cycles <= 0:
            raise SystemExit(
                f"non-positive cycle delta: {hex_readings[0]} → "
                f"{hex_readings[1]} (Δ={cycles})",
            )

        mem_reads = [
            (int(m.group(1), 16), int(m.group(2), 16))
            for m in _MEM_LINE_RE.finditer(gdb_proc.stdout)
        ]
        observed = {a for a, _ in mem_reads}
        if addrs["n_classifications"] not in observed:
            raise SystemExit(
                f"gdb output missing n_classifications memory read:\n"
                f"  wanted: {hex(addrs['n_classifications'])}\n"
                f"  saw:    {[hex(a) for a in observed]}\n"
                f"  full stdout:\n{gdb_proc.stdout}",
            )
        n = next(v for a, v in mem_reads if a == addrs["n_classifications"])

    finally:
        if renode_proc.poll() is None:
            try:
                os.killpg(renode_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                renode_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(renode_proc.pid, signal.SIGKILL)
        renode_log_fh.close()
        try:
            os.unlink(resc_path)
        except OSError:
            pass

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
    p.add_argument("--out", type=Path, default=THIS_DIR / "results_arm.csv")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of MCU names to run.")
    p.add_argument("--renode", type=Path, default=RENODE_DEFAULT_PATH,
                   help="Renode launcher path.")
    p.add_argument("--gdb", type=str, default="gdb-multiarch",
                   help="Multi-arch gdb tool (default gdb-multiarch).")
    p.add_argument("--port", type=int, default=DEFAULT_GDB_PORT)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gdb_tool = _require(args.gdb)
    _require("arm-none-eabi-gcc")
    _require("arm-none-eabi-nm")
    if not args.renode.is_file():
        raise SystemExit(
            f"Renode launcher not found at {args.renode}.  Pass --renode "
            f"explicitly or install the portable bundle there.",
        )

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
        rows.append(_run_one(t, args.renode, gdb_tool, args.port, args.timeout))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
