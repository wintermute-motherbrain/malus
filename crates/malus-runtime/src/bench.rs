// M30 — warm per-step median timer (ADR-0038). Dormant unless the CLI calls
// bench_enable() (--bench); the builtins compile to calls that return
// immediately in normal runs. bench_step_end flushes GPU work inside the
// timed region to match bench/nanogpt_pytorch.py's torch.mps.synchronize()
// methodology — the two medians are only comparable if both serialize the
// step.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const WARMUP_STEPS: usize = 3;

static BENCH_ENABLED: AtomicBool = AtomicBool::new(false);
static STEP_START: Mutex<Option<Instant>> = Mutex::new(None);
static STEP_TIMES: Mutex<Vec<Duration>> = Mutex::new(Vec::new());

pub fn bench_enable() {
    BENCH_ENABLED.store(true, Ordering::SeqCst);
}

pub extern "C" fn bench_step_begin() {
    if !BENCH_ENABLED.load(Ordering::SeqCst) {
        return;
    }
    *STEP_START.lock().unwrap() = Some(Instant::now());
}

pub extern "C" fn bench_step_end() {
    if !BENCH_ENABLED.load(Ordering::SeqCst) {
        return;
    }
    crate::gpu_barrier();
    let start = STEP_START.lock().unwrap().take();
    if let Some(start) = start {
        bench_record(start.elapsed());
    }
}

pub(crate) fn bench_record(elapsed: Duration) {
    STEP_TIMES.lock().unwrap().push(elapsed);
}

pub struct BenchReport {
    pub warm_steps: usize,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
}

pub fn bench_report() -> Option<BenchReport> {
    let times = STEP_TIMES.lock().unwrap();
    if times.len() <= WARMUP_STEPS {
        return None;
    }
    let mut warm: Vec<Duration> = times[WARMUP_STEPS..].to_vec();
    warm.sort();
    let n = warm.len();
    let median = if n % 2 == 1 {
        warm[n / 2]
    } else {
        (warm[n / 2 - 1] + warm[n / 2]) / 2
    };
    Some(BenchReport {
        warm_steps: n,
        median,
        min: warm[0],
        max: warm[n - 1],
    })
}

pub fn bench_reset() {
    BENCH_ENABLED.store(false, Ordering::SeqCst);
    *STEP_START.lock().unwrap() = None;
    STEP_TIMES.lock().unwrap().clear();
}
