use anyhow::{Context, Result};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use http_body_util::{BodyExt, Empty};
use hyper::Request;
use hyper::client::conn::http1::handshake;
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use tokio::net::UnixStream;

#[derive(Debug, Clone)]
pub struct ContainerInfo {
    pub id: String,
    pub name: String,
    pub image: String,
    pub status: String,
    pub state: String,
    pub created: i64,
    pub started_at: Option<DateTime<Utc>>,
    pub ports: String,
}

#[derive(Debug, Deserialize)]
struct ContainerJson {
    Id: String,
    Names: Vec<String>,
    Image: String,
    Status: String,
    State: String,
    Created: i64,
    Ports: Option<Vec<PortMapping>>,
    // Labels etc ignored
}

#[derive(Debug, Deserialize)]
struct PortMapping {
    PublicPort: Option<u16>,
    PrivatePort: u16,
    Type: String,
}

#[derive(Debug, Deserialize)]
struct InspectJson {
    Id: String,
    Created: String,
    State: InspectState,
    Config: InspectConfig,
}

#[derive(Debug, Deserialize)]
struct InspectState {
    Status: String,
    StartedAt: String,
}

#[derive(Debug, Deserialize)]
struct InspectConfig {
    Tty: bool,
    Image: String,
}

#[derive(Clone)]
pub struct DockerClient {
    pub socket_path: String,
}

impl DockerClient {
    pub fn new(socket_path: String) -> Self {
        Self { socket_path }
    }

    async fn http_get(&self, path_and_query: &str) -> Result<(hyper::StatusCode, Bytes)> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .with_context(|| format!("connect docker socket {}", self.socket_path))?;
        let io = TokioIo::new(stream);
        let (mut sender, conn) = handshake(io).await.context("handshake")?;
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                eprintln!("connection error: {e}");
            }
        });
        let req = Request::builder()
            .uri(path_and_query)
            .method("GET")
            .header("Host", "localhost")
            .header("Connection", "close")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res = sender.send_request(req).await.context("send_request")?;
        let status = res.status();
        let body = res.collect().await.context("collect")?.to_bytes();
        Ok((status, body))
    }

    pub async fn list_containers(&self) -> Result<Vec<ContainerInfo>> {
        let (status, body) = self.http_get("/containers/json").await?;
        if !status.is_success() {
            anyhow::bail!("GET /containers/json failed: {} {}", status, String::from_utf8_lossy(&body));
        }
        let containers: Vec<ContainerJson> = serde_json::from_slice(&body).context("parse containers json")?;
        let mut out = Vec::new();
        for c in containers {
            let name = c.Names.first().map(|n| n.trim_start_matches('/').to_string()).unwrap_or(c.Id[..12].to_string());
            let ports = c.Ports.as_ref().map(|ps| {
                ps.iter().map(|p| {
                    if let Some(pubp) = p.PublicPort {
                        format!("{}:{}/{}", pubp, p.PrivatePort, p.Type)
                    } else {
                        format!("{}/{}", p.PrivatePort, p.Type)
                    }
                }).collect::<Vec<_>>().join(", ")
            }).unwrap_or_default();
            out.push(ContainerInfo {
                id: c.Id,
                name,
                image: c.Image,
                status: c.Status,
                state: c.State,
                created: c.Created,
                started_at: None,
                ports,
            });
        }
        // sort by name
        out.sort_by(|a,b| a.name.cmp(&b.name));
        Ok(out)
    }

    pub async fn inspect_container(&self, id: &str) -> Result<(Option<DateTime<Utc>>, bool, String)> {
        let (status, body) = self.http_get(&format!("/containers/{}/json", id)).await?;
        if !status.is_success() {
            anyhow::bail!("inspect failed: {} {}", status, String::from_utf8_lossy(&body));
        }
        let info: InspectJson = serde_json::from_slice(&body).context("parse inspect")?;
        // StartedAt is like "2025-09-02T12:00:00.123456789Z" or "0001-01-01T00:00:00Z" if not started
        let started = if info.State.StartedAt.starts_with("0001-") {
            None
        } else {
            DateTime::parse_from_rfc3339(&info.State.StartedAt).ok().map(|dt| dt.with_timezone(&Utc))
        };
        Ok((started, info.Config.Tty, info.Config.Image))
    }

    /// Stream logs for container, sending each line to tx.
    /// Tail is number of lines to fetch initially.
    /// This function runs until connection closes or tx closed.
    pub async fn logs_follow(&self, id: &str, tail: usize, tx: tokio::sync::mpsc::Sender<String>) -> Result<()> {
        let path = format!(
            "/containers/{}/logs?follow=true&stdout=true&stderr=true&tail={}&timestamps=true",
            id, tail
        );
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .with_context(|| format!("connect docker socket {}", self.socket_path))?;
        let io = TokioIo::new(stream);
        let (mut sender, conn) = handshake(io).await.context("handshake logs")?;
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                // connection closed is expected on switch
                eprintln!("logs conn error: {e}");
            }
        });
        let req = Request::builder()
            .uri(path)
            .method("GET")
            .header("Host", "localhost")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res = sender.send_request(req).await.context("send logs request")?;
        if !res.status().is_success() {
            let status = res.status();
            let body = res.collect().await.map(|b| b.to_bytes()).unwrap_or_default();
            anyhow::bail!("logs request failed: {} {}", status, String::from_utf8_lossy(&body));
        }

        let mut body = res.into_body();
        // Buffer for multiplex decoding
        let mut buf: Vec<u8> = Vec::with_capacity(8192);
        // For raw mode detection, if we never see multiplex header, treat as raw
        // We'll attempt multiplex parsing loop; if header invalid, fallback to raw line split
        let mut is_multiplex: Option<bool> = None;

        while let Some(frame_result) = body.frame().await {
            let frame = frame_result.context("frame")?;
            let Some(data) = frame.data_ref() else { continue; };
            // Append to buffer
            buf.extend_from_slice(data);

            // If we haven't decided multiplex vs raw, try to detect
            if is_multiplex.is_none() && buf.len() >= 8 {
                let st = buf[0];
                let is_mplex = (st == 1 || st == 2) && buf[1]==0 && buf[2]==0 && buf[3]==0;
                // Additional check: size plausible (less than 2MB)
                let size = u32::from_be_bytes([buf[4],buf[5],buf[6],buf[7]]) as usize;
                if is_mplex && size < 2*1024*1024 && size != 0 {
                    is_multiplex = Some(true);
                } else {
                    // Check if buffer contains newline and no valid header => raw
                    // For TTY containers, first byte is not 1/2, and content is ascii
                    // We'll decide raw if first byte not 1/2
                    if st != 1 && st != 2 {
                        is_multiplex = Some(false);
                    } else {
                        // header claims multiplex but size huge => treat as raw (unlikely)
                        // peek further bytes: if header looks invalid, go raw
                        is_multiplex = Some(false);
                    }
                }
            }

            // Now decode according to mode
            if is_multiplex == Some(true) {
                // decode multiplex frames from buf
                let mut offset = 0usize;
                loop {
                    if buf.len() - offset < 8 {
                        break;
                    }
                    let st = buf[offset];
                    if st != 1 && st != 2 {
                        // Corrupt frame, switch to raw for remainder
                        // treat remaining as raw
                        is_multiplex = Some(false);
                        break;
                    }
                    // expect 3 zero bytes
                    if buf[offset+1]!=0 || buf[offset+2]!=0 || buf[offset+3]!=0 {
                        is_multiplex = Some(false);
                        break;
                    }
                    let size = u32::from_be_bytes([buf[offset+4],buf[offset+5],buf[offset+6],buf[offset+7]]) as usize;
                    if buf.len() - offset < 8 + size {
                        break; // need more data
                    }
                    let payload = &buf[offset+8..offset+8+size];
                    // split payload into lines
                    for line in payload.split(|&b| b == b'\n') {
                        if line.is_empty() { continue; }
                        let s = String::from_utf8_lossy(line).to_string();
                        // Could have timestamp prefix still; we keep it, parser strips
                        let _ = tx.send(s).await;
                        if tx.is_closed() { return Ok(()); }
                    }
                    offset += 8 + size;
                }
                if offset > 0 {
                    buf.drain(0..offset);
                }
            } else if is_multiplex == Some(false) {
                // raw mode: split by newline
                // Find newlines and emit
                while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                    let line_bytes = buf[..pos].to_vec();
                    buf.drain(0..pos+1);
                    if line_bytes.is_empty() { continue; }
                    let s = String::from_utf8_lossy(&line_bytes).to_string();
                    let _ = tx.send(s).await;
                    if tx.is_closed() { return Ok(()); }
                }
                // leave incomplete line in buf for next chunk
                // prevent unbounded growth: if buf > 1MB without newline, flush as line
                if buf.len() > 1024*1024 {
                    let s = String::from_utf8_lossy(&buf).to_string();
                    buf.clear();
                    let _ = tx.send(s).await;
                }
            } else {
                // not yet decided, keep buffering until 8 bytes
                if buf.len() > 8192 {
                    // force raw
                    is_multiplex = Some(false);
                }
            }
        }
        // Flush remaining buffer as final line if any
        if !buf.is_empty() {
            let s = String::from_utf8_lossy(&buf).to_string();
            if !s.trim().is_empty() {
                let _ = tx.send(s).await;
            }
        }
        Ok(())
    }
}
