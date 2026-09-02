# vllmtop

btop-style TUI for vLLM — parses Docker logs in real time to show throughput, KV cache, and request status at a glance

> Single binary, works with only the Docker socket

**Language:** [한국어](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [中文](README.zh.md)

---

### Installation

#### One-line Install (GitHub) — Recommended

The simplest way. Fetches the install script directly from GitHub, auto-detects architecture (x86_64 / aarch64) and downloads the latest release.

```bash
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash
```

With `wget`:

```bash
wget -qO- https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash
```

Specific version:

```bash
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | VLLMTOP_VERSION=v0.1.0 bash
```

Options:

```bash
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash -s -- --user
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash -s -- --system
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash -s -- --prefix /opt/bin
```

If installed to `~/.local/bin`, add it to your PATH if needed:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### Direct Download from Releases

If you prefer manual download or are offline, get the asset from [Releases](https://github.com/hanana-kick/vllmtop/releases):

```bash
# Direct binary
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64
chmod +x vllmtop-linux-x86_64
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop

# Or tar.gz (includes install.sh)
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64.tar.gz
tar xzf vllmtop-linux-x86_64.tar.gz
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop
# or via included script
./install.sh
./install.sh --user
./install.sh --system
```

For ARM64 servers, use `vllmtop-linux-aarch64`.

Manual install:

```bash
sudo install -m 755 ./vllmtop /usr/local/bin/vllmtop
# or user-local
install -Dm755 ./vllmtop ~/.local/bin/vllmtop
```

#### Permissions

```bash
ls -l /var/run/docker.sock
# srw-rw---- root docker ...
docker ps   # if this works, vllmtop will work
```

If `docker ps` fails with permission denied, add yourself to the docker group or use `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
# or
sudo vllmtop
```

#### Uninstall

```bash
./uninstall.sh              # removes from both /usr/local/bin and ~/.local/bin
./uninstall.sh --user       # only ~/.local/bin
./uninstall.sh --system     # only /usr/local/bin
# manual
sudo rm -f /usr/local/bin/vllmtop
rm -f ~/.local/bin/vllmtop
# even if installed via one-line
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/uninstall.sh | bash
```

### Usage

#### Interactive — Container Selector

```bash
vllmtop
```

Shows all running containers.

```
vllmtop

Select Docker container

  NAME                    IMAGE
> qwen38                  provsalt/qwen3.8-flash...
  glm53                   glm53-vllm:latest

↑/↓ select   Enter open   q quit
```

Press `Enter` on the container you want.

#### Direct Attach — Skip Selector

```bash
vllmtop qwen38
vllmtop glm53
vllmtop a1b2c3d4e5f6   # container ID prefix also works
```

#### Options

```bash
vllmtop --help
vllmtop --version
vllmtop --socket /var/run/docker.sock
vllmtop qwen38 --tail 500
vllmtop --socket /other/docker.sock qwen38 --tail 200
```

| Option | Description | Default |
|--------|-------------|---------|
| `CONTAINER` | Container name/ID to attach directly | none (shows selector) |
| `--socket PATH` | Docker socket path | `/var/run/docker.sock` |
| `--tail N` | Past log lines to fetch on start | `100` |

#### Dashboard

```
┌ vllmtop ─ qwen38 ────────────────────────────────────────────────┐
│ RUNNING 4  WAITING 0  uptime 01:42:31  last log 01:42:30  container: qwen38  │
├ Throughput ──────────────────────┬ KV Cache ─────────────────────┤
│ Generation     1.80 tok/s        │ 45.2%                         │
│ Prompt         0.00 tok/s        │ █████████░░░░░░░░░░           │
├ Prefix Cache ────────────────────┼ Requests ─────────────────────┤
│ Hit rate       93.6%             │ Running       4               │
│ ██████████████████░              │ Waiting       0               │
├ Speculative Decoding ────────────────────────────────────────────┤
│ Acceptance 100% │ Mean 3.00 │ Accepted 1.20 │ Draft 1.20 tok/s │
├ PLE mmap ────────────────────────────────────────────────────────┤
│ 896.45 ms/op   gather 835.57 ms/op   242,939 rows   37.1 MiB   │
└─────────────────────────────────────────────────────────────────┘
 c container   l logs   p pause   r reset history   q quit
```

- **Header**: RUNNING/WAITING, uptime, last log time, container
- **Throughput**: Generation / Prompt tokens/s
- **KV Cache**: GPU KV cache usage (bar)
- **Prefix Cache**: Hit rate (bar)
- **Requests**: Running / Waiting
- **Speculative Decoding**: Acceptance rate, mean, throughput
- **PLE mmap**: Op time, gather time, rows, size

#### Keys

| Key | Action |
|-----|--------|
| `↑` `↓` / `k` `j` | Navigate list (selector), scroll logs (log view) |
| `Enter` | Open container |
| `c` | Switch container (back to selector) |
| `l` | Toggle logs / back to dashboard |
| `p` | Pause / resume |
| `r` | Reset history (refresh list in selector) |
| `q` / `Esc` | Quit (or back to dashboard from logs) |
| `Ctrl+C` | Quit |
