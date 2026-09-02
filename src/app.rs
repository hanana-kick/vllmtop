use crate::docker::{ContainerInfo, DockerClient};
use crate::state::MetricState;
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEventKind};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Row, Table, TableState, Wrap},
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
                // Extract docker timestamp if present (e.g. 2026-09-02T21:50:36.123456789Z)
                if let Some(first) = line.split_whitespace().next() {
                    if first.contains('T') && first.ends_with('Z') {
                        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(first) {
                            self.state.set_last_log_time(dt.with_timezone(&chrono::Utc));
                        }
                    }
                }
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
                    if self.log_scroll < 10000 {
                        self.log_scroll += 1;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.log_scroll > 0 {
                        self.log_scroll -= 1;
                    }
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
        let area = f.area();
        if area.width < 50 || area.height < 12 {
            let p = Paragraph::new("Terminal too small — resize to at least 80x24")
                .block(block_default(" vllmtop "))
                .wrap(Wrap { trim: false });
            f.render_widget(p, area);
            return;
        }
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
            Span::styled("vllmtop ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{} containers", self.containers.len())),
            Span::raw("  —  Select Docker container"),
        ]))
        .block(block_default(" vllmtop "))
        .wrap(Wrap { trim: false });
        f.render_widget(title, chunks[0]);

        if self.containers.is_empty() {
            let msg = if let Some(m) = &self.status_msg {
                format!("No containers found.\n{m}\n\nCheck socket: {}\nPress r to refresh, q to quit", self.socket_path)
            } else {
                format!("No running containers.\nSocket: {}\nPress r to refresh, q to quit", self.socket_path)
            };
            let p = Paragraph::new(msg)
                .block(block_default(" Containers "))
                .wrap(Wrap { trim: false });
            f.render_widget(p, chunks[1]);
        } else {
            let header = Row::new(vec!["NAME", "IMAGE", "STATUS", "ID"])
                .style(Style::default().add_modifier(Modifier::BOLD))
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
                .block(block_default(" Containers (↑/↓ select, Enter open) "))
                .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
                .highlight_symbol("▶ ");
            f.render_stateful_widget(table, chunks[1], &mut self.table_state);
        }

        let footer = Paragraph::new(Line::from(vec![
            Span::styled("↑/↓", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" select  "),
            Span::styled("Enter", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" open  "),
            Span::styled("r", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" refresh  "),
            Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" quit"),
        ]))
        .block(block_default(""));
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

        let container_name = self.current_container.as_ref().map(|c| c.name.clone()).unwrap_or("-".to_string());
        let paused_str = if self.state.paused { " ⏸ PAUSED " } else { "" };
        let header_line = Line::from(vec![
            Span::styled(format!(" RUNNING {:>3} ", self.state.current.running), Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(" WAITING {:>3} ", self.state.current.waiting)),
            Span::raw(format!(" uptime {} ", self.state.uptime_str())),
            Span::raw(format!(" last log {} ", self.state.last_log_str())),
            Span::styled(paused_str, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(" container: {} ", truncate(&container_name, 18))),
        ]);
        let header_title = format!(" vllmtop ─ {} ", container_name);
        let header = Paragraph::new(header_line).block(block_default(&header_title));
        f.render_widget(header, main_chunks[0]);

        let mid = main_chunks[1];
        // Numbers-only layout: compact, no graphs
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(3),
                Constraint::Length(3),
            ])
            .split(mid);

        let row1_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);
        self.draw_throughput(f, row1_cols[0]);
        self.draw_kv(f, row1_cols[1]);

        let row2_cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[1]);
        self.draw_prefix(f, row2_cols[0]);
        self.draw_requests(f, row2_cols[1]);

        self.draw_spec(f, rows[2]);
        self.draw_ple(f, rows[3]);

        let help = Paragraph::new(Line::from(vec![
            Span::styled("c", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" container  "),
            Span::styled("l", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" logs  "),
            Span::styled("p", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" pause  "),
            Span::styled("r", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" reset history  "),
            Span::styled("q", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" quit"),
        ]));
        f.render_widget(help, main_chunks[2]);
    }

    fn draw_throughput(&self, f: &mut Frame, area: Rect) {
        let block = block_default(" Throughput ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 2 || inner.width < 10 { return; }
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw("Generation "),
                Span::styled(format!("{:>6.2} tok/s", self.state.current.generation_throughput), Style::default().add_modifier(Modifier::BOLD)),
            ])),
            Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 },
        );
        f.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw("Prompt     "),
                Span::styled(format!("{:>6.2} tok/s", self.state.current.prompt_throughput), Style::default().add_modifier(Modifier::BOLD)),
            ])),
            Rect { x: inner.x, y: inner.y + 1, width: inner.width, height: 1 },
        );
    }

    fn draw_kv(&self, f: &mut Frame, area: Rect) {
        let pct = self.state.current.gpu_kv_cache.clamp(0.0, 100.0);
        let block = block_default(" KV Cache ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 1 || inner.width < 10 { return; }
        f.render_widget(Paragraph::new(format!("{:.1}%", pct)), Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 });
        if inner.height >= 2 {
            let bar_width = inner.width as usize;
            let filled = ((pct / 100.0) * bar_width as f32).round() as usize;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(bar_width.saturating_sub(filled)));
            f.render_widget(Paragraph::new(bar), Rect { x: inner.x, y: inner.y + 1, width: inner.width, height: 1 });
        }
    }

    fn draw_prefix(&self, f: &mut Frame, area: Rect) {
        let hit = self.state.current.prefix_hit_rate.unwrap_or(0.0).clamp(0.0, 100.0);
        let block = block_default(" Prefix Cache ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 1 || inner.width < 10 { return; }
        let label = if self.state.current.prefix_hit_rate.is_some() {
            format!("Hit rate {:.1}%", hit)
        } else {
            "Hit rate --".to_string()
        };
        f.render_widget(Paragraph::new(label), Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 });
        if inner.height >= 2 {
            let bar_width = inner.width as usize;
            let filled = ((hit / 100.0) * bar_width as f32).round() as usize;
            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(bar_width.saturating_sub(filled)));
            f.render_widget(Paragraph::new(bar), Rect { x: inner.x, y: inner.y + 1, width: inner.width, height: 1 });
        }
    }

    fn draw_requests(&self, f: &mut Frame, area: Rect) {
        let block = block_default(" Requests ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 2 { return; }
        f.render_widget(Paragraph::new(format!("Running {:>4}", self.state.current.running)), Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 });
        f.render_widget(Paragraph::new(format!("Waiting {:>4}", self.state.current.waiting)), Rect { x: inner.x, y: inner.y + 1, width: inner.width, height: 1 });
    }

    fn draw_spec(&self, f: &mut Frame, area: Rect) {
        let block = block_default(" Speculative Decoding ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 1 || inner.width < 20 { return; }
        let cur = &self.state.current;
        let acceptance = cur.spec_acceptance.map(|v| format!("{:.1}%", v)).unwrap_or("--".to_string());
        let mean = cur.spec_mean_accepted.map(|v| format!("{:.2}", v)).unwrap_or("--".to_string());
        let accepted = cur.spec_accepted_tps.map(|v| format!("{:.2}", v)).unwrap_or("--".to_string());
        let draft = cur.spec_draft_tps.map(|v| format!("{:.2} tok/s", v)).unwrap_or("--".to_string());
        let line1 = Line::from(vec![
            Span::raw("Acceptance "), Span::styled(acceptance, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" │ Mean "), Span::styled(mean, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" │ Accepted "), Span::styled(accepted, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(" │ Draft "), Span::styled(draft, Style::default().add_modifier(Modifier::BOLD)),
        ]);
        f.render_widget(Paragraph::new(line1).wrap(Wrap { trim: false }), Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 });
    }

    fn draw_ple(&self, f: &mut Frame, area: Rect) {
        let block = block_default(" PLE mmap ");
        let inner = block.inner(area);
        f.render_widget(block, area);
        if inner.height < 1 || inner.width < 20 { return; }
        let cur = &self.state.current;
        let ms = cur.ple_ms_per_op.map(|v| format!("{:.2} ms/op", v)).unwrap_or("--".to_string());
        let gather = cur.ple_gather_ms.map(|v| format!("gather {:.2} ms/op", v)).unwrap_or("".to_string());
        let rows = cur.ple_rows.map(|v| format!("{} rows", v)).unwrap_or("--".to_string());
        let mib = cur.ple_mib.map(|v| format!("{:.1} MiB", v)).unwrap_or("--".to_string());
        let mut parts: Vec<Span> = vec![Span::styled(ms, Style::default().add_modifier(Modifier::BOLD))];
        if !gather.is_empty() {
            parts.push(Span::raw("   "));
            parts.push(Span::styled(gather, Style::default()));
        }
        parts.push(Span::raw("   "));
        parts.push(Span::styled(rows, Style::default()));
        parts.push(Span::raw("   "));
        parts.push(Span::styled(mib, Style::default()));
        let line1 = Line::from(parts);
        f.render_widget(Paragraph::new(line1).wrap(Wrap { trim: false }), Rect { x: inner.x, y: inner.y, width: inner.width, height: 1 });
    }

    fn draw_logs(&self, f: &mut Frame) {
        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(3), Constraint::Min(5), Constraint::Length(1)])
            .split(area);
        let name = self.current_container.as_ref().map(|c| c.name.clone()).unwrap_or("-".to_string());
        let header = Paragraph::new(Line::from(vec![
            Span::styled(format!("Logs ─ {}", truncate(&name, 30)), Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("  ({} lines) ", self.state.log_lines.len())),
            Span::raw(if self.state.paused { "⏸ paused" } else { "" }),
        ]))
        .block(block_default(" vllmtop logs "));
        f.render_widget(header, chunks[0]);

        let total = self.state.log_lines.len();
        let visible_h = chunks[1].height as usize;
        let max_scroll = total.saturating_sub(visible_h);
        let scroll = self.log_scroll.min(max_scroll);
        let start = total.saturating_sub(visible_h).saturating_sub(scroll);
        let end = (start + visible_h).min(total);
        let lines: Vec<Line> = self.state.log_lines.iter().skip(start).take(end - start).map(|l| {
            let style = if l.contains("ERROR") || l.contains("error") {
                Style::default().add_modifier(Modifier::BOLD)
            } else if l.contains("WARN") {
                Style::default().add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            let truncated = truncate(l, chunks[1].width as usize);
            Line::styled(truncated, style)
        }).collect();
        let para = Paragraph::new(lines).block(block_default("")).wrap(Wrap { trim: false });
        f.render_widget(para, chunks[1]);

        let footer = Paragraph::new(Line::from(vec![
            Span::styled("↑/↓", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" scroll  "),
            Span::styled("l/Esc", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" back  "),
            Span::styled("q", Style::default().add_modifier(Modifier::BOLD)), Span::raw(" quit"),
        ]));
        f.render_widget(footer, chunks[2]);
    }
}

fn block_default(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(title)
}

fn truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_string()
    } else if max <= 1 {
        "…".to_string()
    } else {
        let truncated: String = chars[..max-1].iter().collect();
        format!("{}…", truncated)
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    fn make_app() -> App {
        let mut app = App::new("/var/run/docker.sock".to_string(), 100);
        app.state.current.running = 4;
        app.state.current.waiting = 0;
        app.state.current.generation_throughput = 49.70;
        app.state.current.prompt_throughput = 0.0;
        app.state.current.gpu_kv_cache = 55.0;
        app.state.current.prefix_hit_rate = Some(91.8);
        app.state.current.spec_acceptance = Some(67.2);
        app.state.current.spec_mean_accepted = Some(2.34);
        app.state.current.spec_accepted_tps = Some(28.49);
        app.state.current.spec_draft_tps = Some(42.38);
        app.state.current.ple_ms_per_op = Some(243.93);
        app.state.current.ple_gather_ms = Some(233.98);
        app.state.current.ple_rows = Some(339238);
        app.state.current.ple_mib = Some(51.8);
        app.current_container = Some(ContainerInfo {
            id: "abc".to_string(),
            name: "qwen38-flash".to_string(),
            image: "test".to_string(),
            status: "Up 21 hours".to_string(),
            state: "running".to_string(),
            created: 0,
            started_at: None,
            ports: "".to_string(),
        });
        app.mode = Mode::Dashboard;
        app
    }

    #[test]
    fn test_render_dashboard_80x24() {
        let mut app = make_app();
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let content: String = terminal.backend().buffer().content.iter().map(|c| c.symbol()).collect();
        assert!(content.contains("RUNNING"), "should contain RUNNING");
        assert!(content.contains("qwen38-flash"), "should contain container name");
        assert!(content.contains("55.0%"), "kv cache label");
        // No sparkline block characters should dominate
        assert!(!content.contains("▇▇▇▇"), "should not contain sparkline artifacts");
    }

    #[test]
    fn test_render_dashboard_large() {
        let mut app = make_app();
        let backend = TestBackend::new(160, 30);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let content: String = terminal.backend().buffer().content.iter().map(|c| c.symbol()).collect();
        assert!(content.contains("Generation"));
        assert!(content.contains("55.0%"));
    }

    #[test]
    fn test_render_small() {
        let mut app = make_app();
        let backend = TestBackend::new(30, 8);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| app.draw(f)).unwrap();
        let content: String = terminal.backend().buffer().content.iter().map(|c| c.symbol()).collect();
        assert!(content.contains("too small"), "should show too small message");
    }
}
