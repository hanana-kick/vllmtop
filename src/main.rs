mod app;
mod docker;
mod parser;
mod state;

use app::{App, Mode};
use clap::Parser;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use std::io;
use tokio::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "vllmtop", version, about = "btop-style TUI for vLLM via Docker logs", long_about = None)]
struct Cli {
    /// Container name or ID to attach directly (skip selector)
    container: Option<String>,

    /// Docker socket path
    #[arg(long, default_value = "/var/run/docker.sock")]
    socket: String,

    /// Number of past log lines to fetch on start
    #[arg(long, default_value_t = 100)]
    tail: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // If directly requested container, verify it exists; else go to selector
    let mut app = App::new(cli.socket.clone(), cli.tail);

    // Try to fetch containers initially
    app.refresh_containers().await?;

    if let Some(target) = cli.container.clone() {
        // Find by id prefix or name
        let found = app.containers.iter().find(|c| {
            c.id.starts_with(&target) || c.name == target || c.id == target
        }).cloned();
        if let Some(info) = found {
            app.set_container(info);
            app.fetch_start_time().await;
        } else {
            eprintln!("container '{}' not found, showing selector", target);
            // keep selector mode, but already have list
            app.mode = Mode::Select;
        }
    } else {
        // if no containers, still show selector with error
        if app.containers.is_empty() {
            // keep message
        }
    }
    // Setup terminal — if no TTY, fallback to plain listing
    if let Err(e) = enable_raw_mode() {
        eprintln!("no TTY ({}), showing container list:", e);
        for c in &app.containers {
            println!("{:<22} {:<40} {}", c.name, c.image, c.status);
        }
        if app.containers.is_empty() {
            if let Some(msg) = app.status_msg {
                eprintln!("{}", msg);
            }
        }
        return Ok(());
    }
    let mut stdout = io::stdout();
    if let Err(e) = execute!(stdout, EnterAlternateScreen) {
        let _ = disable_raw_mode();
        eprintln!("failed to enter alternate screen ({}), fallback list:", e);
        for c in &app.containers {
            println!("{:<22} {:<40} {}", c.name, c.image, c.status);
        }
        return Ok(());
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    // Main loop with tokio intervals for tick and log polling
    let tick_rate = Duration::from_millis(200);
    let mut last_tick = std::time::Instant::now();
    // For async refresh of containers when in select mode? We'll handle periodically
    let mut last_refresh = std::time::Instant::now();

    loop {
        // poll logs
        app.poll_logs();

        // handle pending fetch
        if app.needs_fetch_start {
            app.needs_fetch_start = false;
            app.fetch_start_time().await;
        }
        if app.needs_refresh {
            app.needs_refresh = false;
            let _ = app.refresh_containers().await;
            last_refresh = std::time::Instant::now();
        }

        terminal.draw(|f| app.draw(f))?;

        // event polling with timeout
        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or(Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let crossterm::event::Event::Key(key) = crossterm::event::read()? {
                app.on_key(key);
                if app.should_quit {
                    break;
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = std::time::Instant::now();
        }

        // periodic refresh when in select mode every 2 seconds
        if app.mode == Mode::Select && last_refresh.elapsed() > Duration::from_secs(2) {
            let _ = app.refresh_containers().await;
            last_refresh = std::time::Instant::now();
        }

        if app.should_quit {
            break;
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Abort log task
    if let Some(h) = app.log_task_handle.take() {
        h.abort();
    }

    Ok(())
}
