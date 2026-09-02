# vllmtop

类 btop 的 vLLM 监控 TUI — 实时解析 Docker 日志，直观展示吞吐、KV 缓存和请求状态

> 单一二进制文件，仅需 Docker 套接字即可运行

**Language:** [한국어](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [中文](README.zh.md)

---

### 安装

#### 1) 从 Releases 下载（推荐）

```bash
# 下载最新二进制
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64
chmod +x vllmtop-linux-x86_64
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop

# 或通过 tar.gz
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64.tar.gz
tar xzf vllmtop-linux-x86_64.tar.gz
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop
```

ARM64 服务器请使用 `vllmtop-linux-aarch64`。

#### 2) 使用安装脚本

`tar.gz` 中包含 `install.sh`。

```bash
tar xzf vllmtop-linux-x86_64.tar.gz
./install.sh              # 到 /usr/local/bin（不可写则到 ~/.local/bin）
./install.sh --user       # 到 ~/.local/bin
./install.sh --system     # 到 /usr/local/bin
./install.sh --prefix /opt/bin  # 自定义路径
```

若安装到 `~/.local/bin`，请按需添加到 PATH：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 3) 手动安装

```bash
sudo install -m 755 ./vllmtop /usr/local/bin/vllmtop
# 或用户目录
install -Dm755 ./vllmtop ~/.local/bin/vllmtop
```

#### 权限

```bash
ls -l /var/run/docker.sock
# srw-rw---- root docker ...
docker ps   # 此命令可用，则 vllmtop 即可用
```

若 `docker ps` 提示权限不足，请加入 docker 组或使用 `sudo`：

```bash
sudo usermod -aG docker $USER
newgrp docker
# 或
sudo vllmtop
```

#### 卸载

```bash
./uninstall.sh              # 从 /usr/local/bin 和 ~/.local/bin 同时移除
./uninstall.sh --user       # 仅 ~/.local/bin
./uninstall.sh --system     # 仅 /usr/local/bin
# 手动
sudo rm -f /usr/local/bin/vllmtop
rm -f ~/.local/bin/vllmtop
```

### 使用方法

#### 交互式 — 容器选择

```bash
vllmtop
```

显示所有运行中的容器。

```
vllmtop

Select Docker container

  NAME                    IMAGE
> qwen38                  provsalt/qwen3.8-flash...
  glm53                   glm53-vllm:latest

↑/↓ 选择   Enter 打开   q 退出
```

在目标容器上按 `Enter`。

#### 直接连接 — 跳过选择界面

```bash
vllmtop qwen38
vllmtop glm53
vllmtop a1b2c3d4e5f6   # 容器 ID 前缀亦可
```

#### 选项

```bash
vllmtop --help
vllmtop --version
vllmtop --socket /var/run/docker.sock
vllmtop qwen38 --tail 500
vllmtop --socket /other/docker.sock qwen38 --tail 200
```

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `CONTAINER` | 直接连接的容器名/ID | 无（显示选择界面） |
| `--socket PATH` | Docker 套接字路径 | `/var/run/docker.sock` |
| `--tail N` | 启动时读取的历史日志行数 | `100` |

#### 仪表盘

```
┌ vllmtop ─ qwen38 ────────────────────────────────────────────────┐
│ RUNNING 4     WAITING 0                         uptime 01:42:31  │
├ Throughput ──────────────────────┬ KV Cache ─────────────────────┤
│ Generation     1.80 tok/s        │ 45.2%                         │
│ Prompt         0.00 tok/s        │ █████████░░░░░░░░░░           │
├ Prefix Cache ────────────────────┼ Requests ─────────────────────┤
│ Hit rate       93.6%             │ Running       4               │
├ Speculative Decoding ────────────────────────────────────────────┤
│ Acceptance 100% │ Mean 3.00 │ Accepted 1.20 │ Draft 1.20 tok/s │
├ PLE mmap ────────────────────────────────────────────────────────┤
│ 896.45 ms/op   gather 835.57 ms/op   242,939 rows   37.1 MiB   │
└─────────────────────────────────────────────────────────────────┘
 c container   l logs   p pause   r reset history   q quit
```

- **Throughput**: Generation / Prompt 每秒 token 数与火花线
- **KV Cache**: GPU KV 缓存使用率
- **Prefix Cache**: 命中率
- **Requests**: 运行中 / 等待中
- **Speculative Decoding**: 接受率、平均接受长度、吞吐
- **PLE mmap**: 单次操作耗时、gather 耗时、行数、大小

#### 按键

| 按键 | 功能 |
|------|------|
| `↑` `↓` / `k` `j` | 列表移动（选择界面）、日志滚动（日志界面） |
| `Enter` | 打开容器 |
| `c` | 切换容器（返回选择界面） |
| `l` | 切换日志 / 返回仪表盘 |
| `p` | 暂停 / 恢复 |
| `r` | 重置历史（选择界面为刷新列表） |
| `q` / `Esc` | 退出（日志界面则返回仪表盘） |
| `Ctrl+C` | 退出 |
