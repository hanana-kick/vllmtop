use std::collections::VecDeque;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    pub timestamp: DateTime<Utc>,
    pub running: u32,
    pub waiting: u32,
    pub gpu_kv_cache: f32,
    pub cpu_kv_cache: Option<f32>,
    pub prefix_hit_rate: Option<f32>,
    pub prompt_throughput: f32,
    pub generation_throughput: f32,
    // speculative
    pub spec_acceptance: Option<f32>, // 0-100
    pub spec_mean_accepted: Option<f32>,
    pub spec_accepted_tps: Option<f32>,
    pub spec_draft_tps: Option<f32>,
    // PLE
    pub ple_ms_per_op: Option<f32>,
    pub ple_gather_ms: Option<f32>,
    pub ple_rows: Option<u64>,
    pub ple_mib: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct History {
    pub gen_tps: VecDeque<f64>,
    pub prompt_tps: VecDeque<f64>,
    pub spec_acceptance: VecDeque<f64>,
    pub ple_ms: VecDeque<f64>,
    pub kv_cache: VecDeque<f64>,
    pub prefix_hit: VecDeque<f64>,
    pub max_len: usize,
}

impl Default for History {
    fn default() -> Self {
        Self {
            gen_tps: VecDeque::new(),
            prompt_tps: VecDeque::new(),
            spec_acceptance: VecDeque::new(),
            ple_ms: VecDeque::new(),
            kv_cache: VecDeque::new(),
            prefix_hit: VecDeque::new(),
            max_len: 120,
        }
    }
}

impl History {
    pub fn push(&mut self, m: &MetricsSnapshot) {
        Self::push_val(&mut self.gen_tps, m.generation_throughput as f64, self.max_len);
        Self::push_val(&mut self.prompt_tps, m.prompt_throughput as f64, self.max_len);
        Self::push_val(&mut self.kv_cache, m.gpu_kv_cache as f64, self.max_len);
        if let Some(v) = m.prefix_hit_rate {
            Self::push_val(&mut self.prefix_hit, v as f64, self.max_len);
        }
        if let Some(v) = m.spec_acceptance {
            Self::push_val(&mut self.spec_acceptance, v as f64, self.max_len);
        }
        if let Some(v) = m.ple_ms_per_op {
            Self::push_val(&mut self.ple_ms, v as f64, self.max_len);
        }
    }

    fn push_val(d: &mut VecDeque<f64>, v: f64, max: usize) {
        d.push_back(v);
        if d.len() > max {
            d.pop_front();
        }
    }

    pub fn clear(&mut self) {
        self.gen_tps.clear();
        self.prompt_tps.clear();
        self.spec_acceptance.clear();
        self.ple_ms.clear();
        self.kv_cache.clear();
        self.prefix_hit.clear();
    }

    pub fn sparkline_data(deque: &VecDeque<f64>) -> Vec<u64> {
        // scale to 0-100 for sparkline; ratatui sparkline uses relative max
        // We'll multiply by 10 and cast
        deque.iter().map(|v| (*v * 10.0).round() as u64).collect()
    }
}

#[derive(Debug, Clone)]
pub struct MetricState {
    pub current: MetricsSnapshot,
    pub history: History,
    pub start_time: Option<DateTime<Utc>>,
    pub log_lines: VecDeque<String>,
    pub max_log_lines: usize,
    pub paused: bool,
}

impl Default for MetricState {
    fn default() -> Self {
        Self {
            current: MetricsSnapshot {
                timestamp: Utc::now(),
                ..Default::default()
            },
            history: History::default(),
            start_time: None,
            log_lines: VecDeque::new(),
            max_log_lines: 500,
            paused: false,
        }
    }
}

impl MetricState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_start_time(&mut self, t: DateTime<Utc>) {
        self.start_time = Some(t);
    }

    pub fn uptime(&self) -> Option<chrono::Duration> {
        self.start_time.map(|s| Utc::now() - s)
    }

    pub fn uptime_str(&self) -> String {
        if let Some(d) = self.uptime() {
            let secs = d.num_seconds().max(0);
            let h = secs / 3600;
            let m = (secs % 3600) / 60;
            let s = secs % 60;
            format!("{:02}:{:02}:{:02}", h, m, s)
        } else {
            "--:--:--".to_string()
        }
    }

    pub fn apply_update(&mut self, upd: ParsedUpdate) {
        if self.paused {
            return;
        }
        let mut m = self.current.clone();
        m.timestamp = Utc::now();
        if let Some(v) = upd.running { m.running = v; }
        if let Some(v) = upd.waiting { m.waiting = v; }
        if let Some(v) = upd.gpu_kv_cache { m.gpu_kv_cache = v; }
        if let Some(v) = upd.cpu_kv_cache { m.cpu_kv_cache = Some(v); }
        if let Some(v) = upd.prefix_hit_rate { m.prefix_hit_rate = Some(v); }
        if let Some(v) = upd.prompt_throughput { m.prompt_throughput = v; }
        if let Some(v) = upd.generation_throughput { m.generation_throughput = v; }
        if let Some(v) = upd.spec_acceptance { m.spec_acceptance = Some(v); }
        if let Some(v) = upd.spec_mean_accepted { m.spec_mean_accepted = Some(v); }
        if let Some(v) = upd.spec_accepted_tps { m.spec_accepted_tps = Some(v); }
        if let Some(v) = upd.spec_draft_tps { m.spec_draft_tps = Some(v); }
        if let Some(v) = upd.ple_ms_per_op { m.ple_ms_per_op = Some(v); }
        if let Some(v) = upd.ple_gather_ms { m.ple_gather_ms = Some(v); }
        if let Some(v) = upd.ple_rows { m.ple_rows = Some(v); }
        if let Some(v) = upd.ple_mib { m.ple_mib = Some(v); }
        self.current = m.clone();
        self.history.push(&m);
    }

    pub fn push_log(&mut self, line: String) {
        self.log_lines.push_back(line);
        if self.log_lines.len() > self.max_log_lines {
            self.log_lines.pop_front();
        }
    }

    pub fn reset_history(&mut self) {
        self.history.clear();
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }
}

#[derive(Debug, Clone, Default)]
pub struct ParsedUpdate {
    pub running: Option<u32>,
    pub waiting: Option<u32>,
    pub gpu_kv_cache: Option<f32>,
    pub cpu_kv_cache: Option<f32>,
    pub prefix_hit_rate: Option<f32>,
    pub prompt_throughput: Option<f32>,
    pub generation_throughput: Option<f32>,
    pub spec_acceptance: Option<f32>,
    pub spec_mean_accepted: Option<f32>,
    pub spec_accepted_tps: Option<f32>,
    pub spec_draft_tps: Option<f32>,
    pub ple_ms_per_op: Option<f32>,
    pub ple_gather_ms: Option<f32>,
    pub ple_rows: Option<u64>,
    pub ple_mib: Option<f32>,
}
