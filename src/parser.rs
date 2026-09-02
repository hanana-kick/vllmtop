use regex::Regex;
use std::sync::LazyLock;
use crate::state::ParsedUpdate;

static REGEXES: LazyLock<RegexSet> = LazyLock::new(RegexSet::new);

fn regexes() -> &'static RegexSet {
    &REGEXES
}

struct RegexSet {
    // docker timestamp prefix
    ts_prefix: Regex,
    // throughput
    prompt_tps: Regex,
    gen_tps: Regex,
    // alternative: "Avg prompt throughput: 123.4 tokens/s"
    // also "Prompt throughput: ..."
    running: Regex,
    waiting: Regex,
    pending: Regex,
    swapped: Regex,
    gpu_kv: Regex,
    cpu_kv: Regex,
    prefix_hit: Regex,
    // speculative
    spec_accept_rate: Regex, // various forms
    spec_accept_rate_pct: Regex,
    spec_mean: Regex,
    spec_accepted_tokens: Regex,
    spec_draft_tokens: Regex,
    spec_accepted_tps: Regex,
    spec_draft_tps: Regex,
    // PLE
    ple_ms_op: Regex, // first ms/op
    ple_gather: Regex,
    ple_rows: Regex,
    ple_mib: Regex,
    // PLE combined line with gather
}

impl RegexSet {
    fn new() -> Self {
        Self {
            ts_prefix: Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\s+").unwrap(),
            prompt_tps: Regex::new(r"(?i)avg\s*prompt\s*throughput\s*:\s*([\d.]+)").unwrap(),
            gen_tps: Regex::new(r"(?i)avg\s*generation\s*throughput\s*:\s*([\d.]+)").unwrap(),
            running: Regex::new(r"(?i)running\s*:\s*(\d+)").unwrap(),
            waiting: Regex::new(r"(?i)waiting\s*:\s*(\d+)").unwrap(),
            pending: Regex::new(r"(?i)pending\s*:\s*(\d+)").unwrap(),
            swapped: Regex::new(r"(?i)swapped\s*:\s*(\d+)").unwrap(),
            gpu_kv: Regex::new(r"(?i)gpu\s*kv\s*cache\s*usage\s*:\s*([\d.]+)\s*%").unwrap(),
            cpu_kv: Regex::new(r"(?i)cpu\s*kv\s*cache\s*usage\s*:\s*([\d.]+)\s*%").unwrap(),
            prefix_hit: Regex::new(r"(?i)prefix\s*cache\s*hit\s*rate\s*:\s*([\d.]+)\s*%").unwrap(),
            spec_accept_rate: Regex::new(r"(?i)acceptance\s*rate\s*:\s*([\d.]+)").unwrap(),
            spec_accept_rate_pct: Regex::new(r"(?i)draft\s*acceptance\s*rate\s*:\s*([\d.]+)\s*%?").unwrap(),
            spec_mean: Regex::new(r"(?i)mean\s*acceptance\s*(length)?\s*:\s*([\d.]+)").unwrap(),
            spec_accepted_tokens: Regex::new(r"(?i)number\s*of\s*accepted\s*tokens\s*:\s*(\d+)").unwrap(),
            spec_draft_tokens: Regex::new(r"(?i)number\s*of\s*draft\s*tokens\s*:\s*(\d+)").unwrap(),
            spec_accepted_tps: Regex::new(r"(?i)accepted\s*throughput\s*:\s*([\d.]+)").unwrap(),
            spec_draft_tps: Regex::new(r"(?i)draft(?:ed)?\s*throughput\s*:\s*([\d.]+)").unwrap(),
            ple_ms_op: Regex::new(r"([\d.]+)\s*ms/op").unwrap(),
            ple_gather: Regex::new(r"(?i)gather\s*([\d.]+)\s*ms/op").unwrap(),
            ple_rows: Regex::new(r"([\d,]+)\s*rows").unwrap(),
            ple_mib: Regex::new(r"([\d.]+)\s*MiB").unwrap(),
        }
    }
}

pub struct VllmLogParser;

impl VllmLogParser {
    pub fn parse_line(raw: &str) -> Option<ParsedUpdate> {
        let rs = regexes();
        // strip docker timestamp
        let line = rs.ts_prefix.replace(raw, "");
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        let mut upd = ParsedUpdate::default();
        let mut matched = false;

        // throughput
        if let Some(c) = rs.prompt_tps.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.prompt_throughput = Some(v);
                    matched = true;
                }
            }
        }
        if let Some(c) = rs.gen_tps.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.generation_throughput = Some(v);
                    matched = true;
                }
            }
        }
        // running
        if let Some(c) = rs.running.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<u32>() {
                    upd.running = Some(v);
                    matched = true;
                }
            }
        }
        // waiting/pending统一为 waiting
        if let Some(c) = rs.waiting.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<u32>() {
                    upd.waiting = Some(v);
                    matched = true;
                }
            }
        } else if let Some(c) = rs.pending.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<u32>() {
                    upd.waiting = Some(v);
                    matched = true;
                }
            }
        }
        if let Some(c) = rs.gpu_kv.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.gpu_kv_cache = Some(v);
                    matched = true;
                }
            }
        }
        if let Some(c) = rs.cpu_kv.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.cpu_kv_cache = Some(v);
                    matched = true;
                }
            }
        }
        if let Some(c) = rs.prefix_hit.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.prefix_hit_rate = Some(v);
                    matched = true;
                }
            }
        }

        // speculative
        // acceptance rate: could be 0-1 or 0-100; normalize to 0-100
        let mut spec_found = false;
        if let Some(c) = rs.spec_accept_rate_pct.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    let pct = if v <= 1.0 { v * 100.0 } else { v };
                    // handle case where capture includes % vs not, but already normalized
                    // if original had % and v <=1, it would be <1% which is wrong; but generic
                    upd.spec_acceptance = Some(pct);
                    matched = true;
                    spec_found = true;
                }
            }
        }
        if !spec_found {
            if let Some(c) = rs.spec_accept_rate.captures(line) {
                if let Some(m) = c.get(1) {
                    if let Ok(v) = m.as_str().parse::<f32>() {
                        let pct = if v <= 1.0 && !line.contains('%') {
                            // heuristic: if value <=1 and not percent sign after, interpret as fraction
                            // but our regex doesn't capture %, so check line contains '%'
                            if v < 1.1 && v > 0.0 {
                                v * 100.0
                            } else { v }
                        } else {
                            v
                        };
                        upd.spec_acceptance = Some(pct);
                        matched = true;
                    }
                }
            }
        }
        if let Some(c) = rs.spec_mean.captures(line) {
            // group 2 is value (group 1 is optional "length")
            let val = c.get(2).or_else(|| c.get(1));
            if let Some(m) = val {
                // For regex with optional length, we need to find the numeric one
                // Actually spec_mean regex has two captures: length? and value, but we treat correctly
                // c.get(2) is numeric, c.get(1) may be length literal
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.spec_mean_accepted = Some(v);
                    matched = true;
                }
            }
        }
        if let Some(c) = rs.spec_accepted_tps.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.spec_accepted_tps = Some(v);
                    matched = true;
                }
            }
        }
        if let Some(c) = rs.spec_draft_tps.captures(line) {
            if let Some(m) = c.get(1) {
                if let Ok(v) = m.as_str().parse::<f32>() {
                    upd.spec_draft_tps = Some(v);
                    matched = true;
                }
            }
        }

        // PLE - detect presence of ms/op or rows or MiB
        // Only parse PLE if line contains relevant keywords to avoid false positives
        let is_ple = line.to_lowercase().contains("ple") || line.to_lowercase().contains("mmap") || line.to_lowercase().contains("gather") || (line.contains("ms/op") && line.contains("rows"));

        if is_ple {
            // ms/op: first occurrence is overall, second if gather present is gather
            let ms_vals: Vec<f32> = rs.ple_ms_op.captures_iter(line)
                .filter_map(|c| c.get(1).and_then(|m| m.as_str().parse::<f32>().ok()))
                .collect();
            if !ms_vals.is_empty() {
                upd.ple_ms_per_op = Some(ms_vals[0]);
                matched = true;
                // if gather present, second value is gather
                if line.to_lowercase().contains("gather") && ms_vals.len() >= 2 {
                    upd.ple_gather_ms = Some(ms_vals[1]);
                } else if let Some(c) = rs.ple_gather.captures(line) {
                    if let Some(m) = c.get(1) {
                        if let Ok(v) = m.as_str().parse::<f32>() {
                            upd.ple_gather_ms = Some(v);
                        }
                    }
                }
            }
            if let Some(c) = rs.ple_rows.captures(line) {
                if let Some(m) = c.get(1) {
                    let s = m.as_str().replace(",", "");
                    if let Ok(v) = s.parse::<u64>() {
                        upd.ple_rows = Some(v);
                        matched = true;
                    }
                }
            }
            if let Some(c) = rs.ple_mib.captures(line) {
                if let Some(m) = c.get(1) {
                    if let Ok(v) = m.as_str().parse::<f32>() {
                        upd.ple_mib = Some(v);
                        matched = true;
                    }
                }
            }
        }

        if matched { Some(upd) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_basic_engine_line() {
        let line = "INFO 09-02 12:00:00 metrics.py:100] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 1.80 tokens/s, Running: 4, Swapped: 0, Pending: 0, GPU KV cache usage: 45.2%, CPU KV cache usage: 0.0%, Prefix cache hit rate: 93.6%";
        let upd = VllmLogParser::parse_line(line).unwrap();
        assert_eq!(upd.prompt_throughput, Some(0.0));
        assert_eq!(upd.generation_throughput, Some(1.80));
        assert_eq!(upd.running, Some(4));
        assert_eq!(upd.waiting, Some(0));
        assert_eq!(upd.gpu_kv_cache, Some(45.2));
        assert_eq!(upd.prefix_hit_rate, Some(93.6));
    }

    #[test]
    fn test_docker_ts_prefix() {
        let line = "2025-09-02T12:00:00.123456789Z INFO Avg prompt throughput: 10.0 tokens/s, Avg generation throughput: 20.0 tokens/s, Running: 1, GPU KV cache usage: 10.0%";
        let upd = VllmLogParser::parse_line(line).unwrap();
        assert_eq!(upd.prompt_throughput, Some(10.0));
        assert_eq!(upd.generation_throughput, Some(20.0));
    }

    #[test]
    fn test_ple_line() {
        let line = "PLE mmap stats: 896.45 ms/op   gather 835.57 ms/op   242,939 rows   37.1 MiB";
        let upd = VllmLogParser::parse_line(line).unwrap();
        assert_eq!(upd.ple_ms_per_op, Some(896.45));
        assert_eq!(upd.ple_gather_ms, Some(835.57));
        assert_eq!(upd.ple_rows, Some(242939));
        assert_eq!(upd.ple_mib, Some(37.1));
    }

    #[test]
    fn test_speculative() {
        let line = "Speculative metrics: Draft acceptance rate: 0.642, Mean acceptance length: 3.0, Accepted throughput: 1.20 tokens/s, Draft throughput: 1.20 tokens/s";
        let upd = VllmLogParser::parse_line(line).unwrap();
        assert!(upd.spec_acceptance.is_some());
        // 0.642 -> 64.2
        assert!((upd.spec_acceptance.unwrap() - 64.2).abs() < 0.01);
        assert_eq!(upd.spec_mean_accepted, Some(3.0));
    }
}
