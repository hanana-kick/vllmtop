use crate::docker::{ContainerInfo, DockerClient};
use crate::state::MetricState;
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEventKind};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Sparkline, Table, TableState, Wrap},
    Frame,
};
use tokio::sync::mpsc;

#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Select,
    Dashboard,
    Logs,
}

pub struct App {
    pub mode: Mode,
    pub socket_path: String,
    pub tail: usize,
    pub containers: Vec<ContainerInfo>,
    pub selected_idx: usize,
    pub table_state: TableState,
    pub current_container: Option<ContainerInfo>,
    pub state: MetricState,
    pub should_quit: bool,
    pub show_help: bool,
    pub log_scroll: usize,
    pub log_tx: Option<mpsc::Sender<String>>,
    pub log_task_handle: Option<tokio::task::JoinHandle<()>>,
    pub pending_rx: Option<mpsc::Receiver<String>>,
    pub docker: DockerClient,
    pub status_msg: Option<String>,
    pub needs_refresh: bool,
    pub needs_fetch_start: bool,
}
impl App {
    pub fn new(socket_path: String, tail: usize) -> Self {
        let docker = DockerClient::new(socket_path.clone());
        Self {
            mode: Mode::Select,
            socket_path,
            tail,
            containers: Vec::new(),
            selected_idx: 0,
            table_state: TableState::default(),
            current_container: None,
            state: MetricState::new(),
            should_quit: false,
            show_help: false,
            log_scroll: 0,
            log_tx: None,
            log_task_handle: None,
            pending_rx: None,
            docker,
            status_msg: None,
            needs_refresh: false,
            needs_fetch_start: false,
        }
    }
    pub async fn refresh_containers(&mut self) -> Result<()> {
        match self.docker.list_containers().await {
            Ok(list) => {
                self.containers = list;
                if self.selected_idx >= self.containers.len() && !self.containers.is_empty() {
                    self.selected_idx = 0;
                }
                self.table_state.select(Some(self.selected_idx));
                self.status_msg = None;
            }
            Err(e) => {
                self.status_msg = Some(format!("docker error: {e}"));
            }
        }
        Ok(())
    }

    pub fn set_container(&mut self, info: ContainerInfo) {
        if let Some(h) = self.log_task_handle.take() {
            h.abort();
        }
        self.current_container = Some(info.clone());
        self.state = MetricState::new();
        self.state.set_start_time(chrono::Utc::now());
        self.mode = Mode::Dashboard;
        self.status_msg = None;
        self.needs_fetch_start = true;
        self.start_log_stream(info.id);
    }

    pub async fn fetch_start_time(&mut self) {
        if let Some(c) = self.current_container.clone() {
            if let Ok((started, _tty, _img)) = self.docker.inspect_container(&c.id).await {
                if let Some(st) = started {
                    self.state.set_start_time(st);
                }
            }
        }
    }

    fn start_log_stream(&mut self, id: String) {
        let (tx, rx) = mpsc::channel::<String>(1000);
        self.log_tx = Some(tx.clone());
        let socket = self.socket_path.clone();
        let tail = self.tail;
        let handle = tokio::spawn(async move {
            let client = DockerClient::new(socket);
            let _ = client.logs_follow(&id, tail, tx).await;
        });
        self.log_task_handle = Some(handle);
        self.pending_rx = Some(rx);
    }

    pub fn poll_logs(&mut self) {
        if let Some(rx) = &mut self.pending_rx {
            while let Ok(line) = rx.try_recv() {
                self.state.push_log(line.clone());
                if let Some(upd) = crate::parser::VllmLogParser::parse_line(&line) {
                    self.state.apply_update(upd);
                }
            }
        }
    }
    pub fn on_key(&mut self, key: crossterm::event::KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        // Ctrl+C always quits
        if key.code == KeyCode::Char('c') && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
            self.should_quit = true;
            return;
        }
        match self.mode {
            Mode::Select => match key.code {
                KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                KeyCode::Up | KeyCode::Char('k') => {
                    if !self.containers.is_empty() {
                        if self.selected_idx > 0 {
                            self.selected_idx -= 1;
                        } else {
                            self.selected_idx = self.containers.len() - 1;
                        }
                        self.table_state.select(Some(self.selected_idx));
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if !self.containers.is_empty() {
                        self.selected_idx = (self.selected_idx + 1) % self.containers.len();
                        self.table_state.select(Some(self.selected_idx));
                    }
                }
                KeyCode::Enter => {
                    if let Some(c) = self.containers.get(self.selected_idx).cloned() {
                        self.set_container(c);
                    }
                }
                KeyCode::Char('r') => {
                    self.needs_refresh = true;
                }
                _ => {}
            },
            Mode::Dashboard => match key.code {
                KeyCode::Char('q') => self.should_quit = true,
                KeyCode::Char('c') => {
                    self.mode = Mode::Select;
                    if let Some(h) = self.log_task_handle.take() {
                        h.abort();
                    }
                    self.pending_rx = None;
                    self.needs_refresh = true;
                }
                KeyCode::Char('l') => {
                    self.mode = Mode::Logs;
                    self.log_scroll = 0;
                }
                KeyCode::Char('p') => {
                    self.state.toggle_pause();
                }
                KeyCode::Char('r') => {
                    self.state.reset_history();
                }
                _ => {}
            },
            Mode::Logs => match key.code {
                KeyCode::Char('q') | KeyCode::Esc | KeyCode::Char('l') => {
                    self.mode = Mode::Dashboard;
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.log_scroll > 0 {
                        self.log_scroll -= 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.log_scroll += 1;
                }
                KeyCode::Char('c') => {
                    self.mode = Mode::Select;
                    if let Some(h) = self.log_task_handle.take() {
                        h.abort();
                    }
                    self.pending_rx = None;
                    self.needs_refresh = true;
                }
                KeyCode::Char('p') => self.state.toggle_pause(),
                KeyCode::Char('r') => self.state.reset_history(),
                _ => {}
            },
        }
    }

    pub fn draw(&mut self, f: &mut Frame) {
        match self.mode {
            Mode::Select => self.draw_select(f),
            Mode::Dashboard => self.draw_dashboard(f),
            Mode::Logs => self.draw_logs(f),
        }
    }

    fn draw_select(&mut self, f: &mut Frame) {
        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5), Constraint::Length(3)])
            .split(area);

        let title = Paragraph::new(Line::from(vec![
            Span::styled("vllmtop ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(format!("{} containers", self.containers.len())),
            Span::raw("  —  Select Docker container"),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" vllmtop "))
        .wrap(Wrap { trim: false });
        f.render_widget(title, chunks[0]);

        if self.containers.is_empty() {
            let msg = if let Some(m) = &self.status_msg {
                format!("No containers found.\n{m}\n\nCheck socket: {}\nPress r to refresh, q to quit", self.socket_path)
            } else {
                format!("No running containers.\nSocket: {}\nPress r to refresh, q to quit", self.socket_path)
            };
            let p = Paragraph::new(msg)
                .block(Block::default().borders(Borders::ALL).title(" Containers "))
                .wrap(Wrap { trim: false });
            f.render_widget(p, chunks[1]);
        } else {
            let header = Row::new(vec!["NAME", "IMAGE", "STATUS", "ID"])
                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                .height(1);
            let rows: Vec<Row> = self.containers.iter().map(|c| {
                let short_id = if c.id.len() > 12 { &c.id[..12] } else { &c.id };
                Row::new(vec![
                    c.name.clone(),
                    truncate(&c.image, 40),
                    truncate(&c.status, 30),
                    short_id.to_string(),
                ])
                .height(1)
            }).collect();
            let widths = [
                Constraint::Length(22),
                Constraint::Min(20),
                Constraint::Length(28),
                Constraint::Length(14),
            ];
            let table = Table::new(rows, widths)
                .header(header)
                .block(Block::default().borders(Borders::ALL).title(" Containers (↑/↓ select, Enter open) "))
                .row_highlight_style(Style::default().bg(Color::DarkGray).fg(Color::White).add_modifier(Modifier::BOLD))
                .highlight_symbol("▶ ");
            f.render_stateful_widget(table, chunks[1], &mut self.table_state);
        }

        let footer = Paragraph::new(Line::from(vec![
            Span::styled("↑/↓", Style::default().fg(Color::Cyan)),
            Span::raw(" select  "),
            Span::styled("Enter", Style::default().fg(Color::Cyan)),
            Span::raw(" open  "),
            Span::styled("r", Style::default().fg(Color::Cyan)),
            Span::raw(" refresh  "),
            Span::styled("q", Style::default().fg(Color::Cyan)),
            Span::raw(" quit"),
        ]))
        .block(Block::default().borders(Borders::ALL));
        f.render_widget(footer, chunks[2]);
    }

    fn draw_dashboard(&self, f: &mut Frame) {
        let area = f.area();
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);

        // Header bar
        let container_name = self.current_container.as_ref().map(|c| c.name.clone()).unwrap_or("-".to_string());
        let paused_str = if self.state.paused { " ⏸ PAUSED " } else { "" };
        let header_line = Line::from(vec![
            Span::styled(format!(" RUNNING {:>3} ", self.state.current.running), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(format!(" WAITING {:>3} ", self.state.current.waiting)),
            Span::raw(format!(" uptime {} ", self.state.uptime_str())),
            Span::styled(paused_str, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw(format!("  container: {} ", container_name)),
        ]);
        let header = Paragraph::new(header_line)
            .block(Block::default().borders(Borders::ALL).title(format!(" vllmtop ─ {} ", container_name)))
            .style(Style::default().bg(Color::Black));
        f.render_widget(header, main_chunks[0]);

        // Middle grid
        let mid = main_chunks[1];
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),
                Constraint::Length(6),
                Constraint::Length(6),
                Constraint::Length(6),
            ])
            .split(mid);

        // Row1: Throughput | KV Cache
        let row1_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);
        self.draw_throughput(f, row1_cols[0]);
        self.draw_kv(f, row1_cols[1]);

        // Row2: Prefix Cache | Requests
        let row2_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);
        self.draw_prefix(f, row2_cols[0]);
        self.draw_requests(f, row2_cols[1]);

        // Row3: Speculative Decoding
        self.draw_spec(f, rows[2]);

        // Row4: PLE
        self.draw_ple(f, rows[3]);

        // Footer help
        let help = Paragraph::new(Line::from(vec![
            Span::styled("c", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" container  "),
            Span::styled("l", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" logs  "),
            Span::styled("p", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" pause  "),
            Span::styled("r", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" reset history  "),
            Span::styled("q", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)), Span::raw(" quit"),
        ]))
        .style(Style::default().fg(Color::DarkGray));
        f.render_widget(help, main_chunks[2]);
    }

    fn draw_throughput(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Throughput ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 2 { return; }
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(2), Constraint::Length(1), Constraint::Length(1)])
            .split(inner);
        let gen = Paragraph::new(Line::from(vec![
            Span::raw("Generation "),
            Span::styled(format!("{:>6.2} tok/s", self.state.current.generation_throughput), Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]));
        f.render_widget(gen, chunks[0]);
        let data = crate::state::History::sparkline_data(&self.state.history.gen_tps);
        let spark = Sparkline::default()
            .data(&data)
            .style(Style::default().fg(Color::Green));
        f.render_widget(spark, chunks[1]);
        let prompt = Paragraph::new(Line::from(vec![
            Span::raw("Prompt     "),
            Span::styled(format!("{:>6.2} tok/s", self.state.current.prompt_throughput), Style::default().fg(Color::Cyan)),
        ]));
        f.render_widget(prompt, chunks[2]);
        // prompt sparkline small
        let pdata = crate::state::History::sparkline_data(&self.state.history.prompt_tps);
        let pspark = Sparkline::default().data(&pdata).style(Style::default().fg(Color::Cyan));
        // Use remaining line if height allows; we already used 4 rows, last is maybe empty
        if chunks.len() > 3 {
            f.render_widget(pspark, Rect { height: 1, ..chunks[3] });
        }
    }

    fn draw_kv(&self, f: &mut Frame, area: Rect) {
        let pct = self.state.current.gpu_kv_cache.clamp(0.0, 100.0) as u16;
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" KV Cache "))
            .gauge_style(Style::default().fg(if pct > 85 { Color::Red } else if pct > 70 { Color::Yellow } else { Color::Green }))
            .percent(pct)
            .label(format!("{:.1}%", self.state.current.gpu_kv_cache));
        f.render_widget(gauge, area);
        // sparkline inside? We could overlay but keep simple: add small sparkline below gauge via inner
        // Instead, we render sparkline in the gauge area's bottom line
        // For simplicity, not adding extra
    }

    fn draw_prefix(&self, f: &mut Frame, area: Rect) {
        let hit = self.state.current.prefix_hit_rate.unwrap_or(0.0);
        let pct = hit.clamp(0.0,100.0) as u16;
        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Prefix Cache "))
            .gauge_style(Style::default().fg(Color::Blue))
            .percent(pct)
            .label(format!("Hit rate {:.1}%", hit));
        f.render_widget(gauge, area);
    }

    fn draw_requests(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Requests ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        let txt = format!("Running {:>4}\nWaiting {:>4}", self.state.current.running, self.state.current.waiting);
        let p = Paragraph::new(txt).style(Style::default().fg(Color::White));
        f.render_widget(p, inner);
    }

    fn draw_spec(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" Speculative Decoding ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height == 0 { return; }
        let cur = &self.state.current;
        let acceptance = cur.spec_acceptance.map(|v| format!("{:.1}%", v)).unwrap_or("--".to_string());
        let mean = cur.spec_mean_accepted.map(|v| format!("{:.2}", v)).unwrap_or("--".to_string());
        let accepted = cur.spec_accepted_tps.map(|v| format!("{:.2}", v)).unwrap_or("--".to_string());
        let draft = cur.spec_draft_tps.map(|v| format!("{:.2} tok/s", v)).unwrap_or("--".to_string());
        let line1 = Line::from(vec![
            Span::raw("Acceptance "), Span::styled(acceptance, Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::raw(" │ Mean "), Span::styled(mean, Style::default().fg(Color::Yellow)),
            Span::raw(" │ Accepted "), Span::styled(accepted, Style::default().fg(Color::Cyan)),
            Span::raw(" │ Draft "), Span::styled(draft, Style::default().fg(Color::Magenta)),
        ]);
        let chunks = Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(1), Constraint::Min(1)]).split(inner);
        f.render_widget(Paragraph::new(line1), chunks[0]);
        if chunks.len() > 1 {
            let data = crate::state::History::sparkline_data(&self.state.history.spec_acceptance);
            let spark = Sparkline::default().data(&data).style(Style::default().fg(Color::Yellow));
            f.render_widget(spark, chunks[1]);
        }
    }

    fn draw_ple(&self, f: &mut Frame, area: Rect) {
        let block = Block::default().borders(Borders::ALL).title(" PLE mmap ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height == 0 { return; }
        let cur = &self.state.current;
        let ms = cur.ple_ms_per_op.map(|v| format!("{:.2} ms/op", v)).unwrap_or("--".to_string());
        let gather = cur.ple_gather_ms.map(|v| format!("gather {:.2} ms/op", v)).unwrap_or("".to_string());
        let rows = cur.ple_rows.map(|v| format!("{} rows", v)).unwrap_or("--".to_string());
        let mib = cur.ple_mib.map(|v| format!("{:.1} MiB", v)).unwrap_or("--".to_string());
        let line1 = Line::from(vec![
            Span::styled(ms, Style::default().fg(Color::Green)),
            Span::raw("   "),
            Span::styled(gather, Style::default().fg(Color::Yellow)),
            Span::raw("   "),
            Span::styled(rows, Style::default().fg(Color::Cyan)),
            Span::raw("   "),
            Span::styled(mib, Style::default().fg(Color::Magenta)),
        ]);
        let chunks = Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(1), Constraint::Min(1)]).split(inner);
        f.render_widget(Paragraph::new(line1), chunks[0]);
        if chunks.len() > 1 {
            let data = crate::state::History::sparkline_data(&self.state.history.ple_ms);
            let spark = Sparkline::default().data(&data).style(Style::default().fg(Color::Green));
            f.render_widget(spark, chunks[1]);
        }
    }

    fn draw_logs(&self, f: &mut Frame) {
        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5), Constraint::Length(1)])
            .split(area);
        let name = self.current_container.as_ref().map(|c| c.name.clone()).unwrap_or("-".to_string());
        let header = Paragraph::new(Line::from(vec![
            Span::styled(format!("Logs ─ {}", name), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw(format!("  ({} lines) ", self.state.log_lines.len())),
            Span::raw(if self.state.paused { "⏸ paused" } else { "" }),
        ]))
        .block(Block::default().borders(Borders::ALL).title(" vllmtop logs "));
        f.render_widget(header, chunks[0]);

        // logs content
        let total = self.state.log_lines.len();
        // calculate window
        let visible_h = chunks[1].height as usize;
        let max_scroll = total.saturating_sub(visible_h);
        let scroll = self.log_scroll.min(max_scroll);
        let start = total.saturating_sub(visible_h).saturating_sub(scroll);
        // Actually we want scroll 0 = bottom, increasing = scroll up
        // So start = total - visible - scroll
        let end = (start + visible_h).min(total);
        let lines: Vec<Line> = self.state.log_lines.iter().skip(start).take(end - start).map(|l| {
            // colorize by level
            let style = if l.contains("ERROR") || l.contains("error") {
                Style::default().fg(Color::Red)
            } else if l.contains("WARN") {
                Style::default().fg(Color::Yellow)
            } else if l.contains("INFO") {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default().fg(Color::White)
            };
            Line::styled(l.clone(), style)
        }).collect();
        let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL)).wrap(Wrap { trim: false });
        f.render_widget(para, chunks[1]);

        let footer = Paragraph::new(Line::from(vec![
            Span::styled("↑/↓", Style::default().fg(Color::Cyan)), Span::raw(" scroll  "),
            Span::styled("l/Esc", Style::default().fg(Color::Cyan)), Span::raw(" back  "),
            Span::styled("q", Style::default().fg(Color::Cyan)), Span::raw(" quit"),
        ]));
        f.render_widget(footer, chunks[2]);
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}…", &s[..max-1]) }
}
