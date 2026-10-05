// Licensed under the Apache-2.0 license

//! PLDM firmware-update benchmark instrumentation for the update agent (UA).
//!
//! Enabled by setting `PLDM_BENCH=1` on the emulator process. When enabled,
//! the UA records both wall-clock time and emulated MCU clock ticks at each
//! milestone of the firmware-update lifecycle, plus per-chunk download
//! statistics, and prints a single summary once, at the very end of the run
//! (from an exit hook: run by the emulator right before the firmware
//! terminates it, or by the test after the run on FPGA).
//!
//! Timing probes deliberately start *after* the UA's fixed start-up delay
//! (see `PldmDaemon::event_loop`), so the reported PLDM phases measure only
//! the protocol work performed once the device is fully initialized.

use caliptra_mcu_emulator_state::{get_emulator_state, register_exit_hook, EmulatorState};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Environment variable that enables benchmarking.
pub const BENCH_ENV: &str = "PLDM_BENCH";

/// Marker printed at the start of the summary.
const BENCH_SUMMARY_HEADER: &str = "===== PLDM firmware update benchmark =====";

/// Milestones observed by the update agent, in lifecycle order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Milestone {
    /// Discovery begins (after the UA's fixed start-up delay).
    DiscoveryStart,
    /// QueryDeviceIdentifiers sent: firmware-update negotiation begins.
    UpdateStart,
    /// First RequestFirmwareData received from the device.
    DownloadStart,
    /// TransferComplete received from the device.
    TransferComplete,
    /// VerifyComplete received from the device.
    VerifyComplete,
    /// ApplyComplete received from the device.
    ApplyComplete,
    /// ActivateFirmware response received.
    ActivateResponse,
    /// The firmware terminated the emulator (v2 firmware booted and exited).
    EmulatorExit,
}

/// Phases reported, as (name, start milestone, end milestone, description).
const PHASES: &[(&str, Milestone, Milestone, &str)] = &[
    (
        "discovery",
        Milestone::DiscoveryStart,
        Milestone::UpdateStart,
        "PLDM base discovery (includes any wait for device boot)",
    ),
    (
        "negotiation",
        Milestone::UpdateStart,
        Milestone::DownloadStart,
        "QueryDeviceIdentifiers .. UpdateComponent",
    ),
    (
        "download",
        Milestone::DownloadStart,
        Milestone::TransferComplete,
        "RequestFirmwareData loop .. TransferComplete",
    ),
    (
        "verify",
        Milestone::TransferComplete,
        Milestone::VerifyComplete,
        "Device-side image verification",
    ),
    (
        "apply",
        Milestone::VerifyComplete,
        Milestone::ApplyComplete,
        "Staging -> inactive partition copy",
    ),
    (
        "activate",
        Milestone::ApplyComplete,
        Milestone::ActivateResponse,
        "ActivateFirmware request/response",
    ),
    (
        "reset_to_exit",
        Milestone::ActivateResponse,
        Milestone::EmulatorExit,
        "Activation, hitless reset, v2 boot",
    ),
    (
        "pldm_total",
        Milestone::UpdateStart,
        Milestone::ActivateResponse,
        "Firmware-update PLDM work on an initialized device",
    ),
];

#[derive(Clone, Copy)]
struct Stamp {
    wall: Instant,
    ticks: u64,
}

struct Recorder {
    state: Arc<EmulatorState>,
    milestones: BTreeMap<Milestone, Stamp>,
    component_size: Option<u64>,
    chunks: u64,
    requested_bytes: u64,
    max_chunk: u32,
    last_chunk: Option<Stamp>,
}

static RECORDER: Mutex<Option<Recorder>> = Mutex::new(None);

fn recorder() -> std::sync::MutexGuard<'static, Option<Recorder>> {
    RECORDER.lock().unwrap_or_else(|e| e.into_inner())
}

fn with_recorder(f: impl FnOnce(&mut Recorder)) {
    if let Some(rec) = recorder().as_mut() {
        f(rec);
    }
}

impl Recorder {
    fn now(&self) -> Stamp {
        Stamp {
            wall: Instant::now(),
            ticks: self.state.ticks.load(Ordering::Relaxed),
        }
    }

    fn delta(&self, start: Milestone, end: Milestone) -> Option<(u64, f64)> {
        let s = self.milestones.get(&start)?;
        let e = self.milestones.get(&end)?;
        Some(span(s, e))
    }
}

fn span(s: &Stamp, e: &Stamp) -> (u64, f64) {
    (
        e.ticks.saturating_sub(s.ticks),
        e.wall.saturating_duration_since(s.wall).as_secs_f64(),
    )
}

/// Enable benchmarking if `PLDM_BENCH` is set. Must be called from a thread
/// with per-instance emulator state (e.g. from `PldmDaemon::run`).
/// Subsequent calls are no-ops.
pub fn init() {
    if std::env::var_os(BENCH_ENV).is_none() {
        return;
    }
    let Some(state) = get_emulator_state() else {
        log::warn!("{BENCH_ENV} set but no emulator state; benchmarking disabled");
        return;
    };
    let mut guard = recorder();
    if guard.is_some() {
        return;
    }
    *guard = Some(Recorder {
        state,
        milestones: BTreeMap::new(),
        component_size: None,
        chunks: 0,
        requested_bytes: 0,
        max_chunk: 0,
        last_chunk: None,
    });
    drop(guard);
    register_exit_hook(finish);
}

/// Record the first occurrence of `milestone`.
pub fn mark(milestone: Milestone) {
    with_recorder(|rec| {
        let stamp = rec.now();
        rec.milestones.entry(milestone).or_insert(stamp);
    });
}

/// Record the size of the component being downloaded.
pub fn set_component_size(size: u32) {
    with_recorder(|rec| {
        rec.component_size.get_or_insert(size as u64);
    });
}

/// Record one served RequestFirmwareData chunk of `len` bytes.
pub fn record_chunk(len: u32) {
    with_recorder(|rec| {
        let stamp = rec.now();
        rec.milestones
            .entry(Milestone::DownloadStart)
            .or_insert(stamp);
        rec.chunks += 1;
        rec.requested_bytes += len as u64;
        rec.max_chunk = rec.max_chunk.max(len);
        rec.last_chunk = Some(stamp);
    });
}

/// `num / den`, or 0 when `den` is 0.
fn ratio(num: f64, den: f64) -> f64 {
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

fn throughput_row(out: &mut String, name: &str, bytes: u64, ticks: u64, secs: f64) {
    let (b, t) = (bytes as f64, ticks as f64);
    let per_kilotick = ratio(b * 1000.0, t);
    let ticks_per_byte = ratio(t, b);
    let kib_s = ratio(b / 1024.0, secs);
    let _ = writeln!(
        out,
        "  {name:<14} {bytes:>10} {ticks:>14} {per_kilotick:>14.3} {ticks_per_byte:>12.1} {kib_s:>10.2}"
    );
}

fn summary(rec: &Recorder, exit_code: i32) -> String {
    let mut out = String::new();
    let payload = rec.component_size.unwrap_or(rec.requested_bytes);
    let _ = writeln!(out, "{BENCH_SUMMARY_HEADER}");
    let _ = writeln!(
        out,
        "exit code: {exit_code}   payload: {payload} B   chunks: {}   requested: {} B   max chunk: {} B",
        rec.chunks, rec.requested_bytes, rec.max_chunk
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {:<14} {:>14} {:>10}  description",
        "phase", "MCU ticks", "wall (s)"
    );
    for (name, start, end, desc) in PHASES {
        match rec.delta(*start, *end) {
            Some((ticks, secs)) => {
                let _ = writeln!(out, "  {name:<14} {ticks:>14} {secs:>10.3}  {desc}");
            }
            None => {
                let _ = writeln!(out, "  {name:<14} {:>14} {:>10}  {desc}", "n/a", "n/a");
            }
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {:<14} {:>10} {:>14} {:>14} {:>12} {:>10}",
        "throughput", "bytes", "MCU ticks", "B/kilotick", "ticks/B", "KiB/s"
    );
    if let Some((t, s)) = rec.delta(Milestone::DownloadStart, Milestone::TransferComplete) {
        throughput_row(&mut out, "download", payload, t, s);
    }
    // Steady-state chunk loop: first chunk served -> last chunk served.
    if let (Some(first), Some(last)) = (
        rec.milestones.get(&Milestone::DownloadStart),
        rec.last_chunk.as_ref(),
    ) {
        let (t, s) = span(first, last);
        throughput_row(&mut out, "chunk_loop", rec.requested_bytes, t, s);
        if rec.chunks > 1 {
            let _ = writeln!(
                out,
                "  avg MCU ticks per chunk: {:.0}",
                t as f64 / (rec.chunks - 1) as f64
            );
        }
    }
    if let Some((t, s)) = rec.delta(Milestone::UpdateStart, Milestone::ActivateResponse) {
        throughput_row(&mut out, "pldm_total", payload, t, s);
    }
    let _ = writeln!(out, "==========================================");
    out
}

/// Print the summary. Registered as an emulator exit hook.
fn finish(exit_code: i32) {
    let Some(mut rec) = recorder().take() else {
        return;
    };
    let stamp = rec.now();
    rec.milestones
        .entry(Milestone::EmulatorExit)
        .or_insert(stamp);
    println!("{}", summary(&rec, exit_code));
}
