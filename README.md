# vllmtop

btop 스타일 vLLM 모니터링 TUI — Docker 로그를 실시간으로 파싱해 처리량, KV 캐시, 요청 상태를 한눈에 표시

> 단일 바이너리, Docker 소켓만으로 동작

---

## 🇰🇷 한국어

### 설치

#### 1) 릴리즈에서 다운로드 (권장)

```bash
# 최신 릴리즈 바이너리 다운로드
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64
chmod +x vllmtop-linux-x86_64
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop

# 또는 tar.gz 로 다운로드
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64.tar.gz
tar xzf vllmtop-linux-x86_64.tar.gz
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop
```

ARM64 서버인 경우 `vllmtop-linux-aarch64` 를 사용하세요.

#### 2) 설치 스크립트 사용

tar.gz 를 풀면 `install.sh` 가 포함되어 있습니다.

```bash
tar xzf vllmtop-linux-x86_64.tar.gz
./install.sh              # /usr/local/bin 에 설치 (권한 없으면 ~/.local/bin)
./install.sh --user       # ~/.local/bin 에 설치
./install.sh --system     # /usr/local/bin 에 설치
./install.sh --prefix /opt/bin  # 원하는 경로에 설치
```

설치 후 `~/.local/bin` 에 설치한 경우 PATH 추가가 필요할 수 있습니다.

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 3) 수동 설치

```bash
sudo install -m 755 ./vllmtop /usr/local/bin/vllmtop
# 또는 사용자 경로
install -Dm755 ./vllmtop ~/.local/bin/vllmtop
```

#### 권한

```bash
ls -l /var/run/docker.sock
# srw-rw---- root docker ...
docker ps   # 이 명령이 동작하면 vllmtop 도 동작합니다
```

`docker ps` 가 권한 오류라면 Docker 그룹에 추가하거나 `sudo` 로 실행하세요.

```bash
sudo usermod -aG docker $USER
newgrp docker
# 또는
sudo vllmtop
```

#### 삭제

```bash
./uninstall.sh              # /usr/local/bin 과 ~/.local/bin 에서 자동 제거
./uninstall.sh --user       # ~/.local/bin 만
./uninstall.sh --system     # /usr/local/bin 만
# 수동
sudo rm -f /usr/local/bin/vllmtop
rm -f ~/.local/bin/vllmtop
```

### 사용법

#### 기본 실행 — 컨테이너 선택

```bash
vllmtop
```

실행 중인 Docker 컨테이너 목록이 표시됩니다.

```
vllmtop

Select Docker container

  NAME                    IMAGE
> qwen38                  provsalt/qwen3.8-flash...
  glm53                   glm53-vllm:latest

↑/↓ 선택   Enter 열기   q 종료
```

원하는 컨테이너에서 `Enter`.

#### 바로 연결 — 선택 화면 건너뛰기

```bash
vllmtop qwen38
vllmtop glm53
vllmtop a1b2c3d4e5f6   # 컨테이너 ID 일부도 가능
```

#### 옵션

```bash
vllmtop --help
vllmtop --version
vllmtop --socket /var/run/docker.sock
vllmtop qwen38 --tail 500          # 시작 시 가져올 과거 로그 줄 수
vllmtop --socket /other/docker.sock qwen38 --tail 200
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `CONTAINER` | 바로 연결할 컨테이너 이름/ID | 없음 (선택 화면) |
| `--socket PATH` | Docker 소켓 경로 | `/var/run/docker.sock` |
| `--tail N` | 시작 시 읽어올 과거 로그 줄 수 | `100` |

#### 대시보드

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

- **Throughput**: Generation / Prompt 토큰 처리량과 스파크라인
- **KV Cache**: GPU KV 캐시 사용률
- **Prefix Cache**: 히트율
- **Requests**: Running / Waiting 요청 수
- **Speculative Decoding**: 채택률, 평균 채택 길이, 처리량
- **PLE mmap**: 연산 시간, gather 시간, 행 수, 용량

#### 키

| 키 | 동작 |
|----|------|
| `↑` `↓` / `k` `j` | 컨테이너 목록 이동 (선택 화면), 로그 스크롤 (로그 화면) |
| `Enter` | 컨테이너 열기 |
| `c` | 컨테이너 전환 (목록으로 돌아가기) |
| `l` | 로그 화면 전환 / 복귀 |
| `p` | 일시정지 / 재개 |
| `r` | 히스토리 초기화 (선택 화면에서는 목록 새로고침) |
| `q` / `Esc` | 종료 (로그 화면에서는 대시보드로 복귀) |
| `Ctrl+C` | 종료 |

---

## 🇺🇸 English

### Installation

#### 1) Download from Releases (Recommended)

```bash
# Download latest binary
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64
chmod +x vllmtop-linux-x86_64
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop

# Or via tar.gz
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64.tar.gz
tar xzf vllmtop-linux-x86_64.tar.gz
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop
```

For ARM64 servers, use `vllmtop-linux-aarch64`.

#### 2) Using the Install Script

The `tar.gz` archive includes `install.sh`.

```bash
tar xzf vllmtop-linux-x86_64.tar.gz
./install.sh              # to /usr/local/bin (falls back to ~/.local/bin if not writable)
./install.sh --user       # to ~/.local/bin
./install.sh --system     # to /usr/local/bin
./install.sh --prefix /opt/bin  # custom path
```

If installed to `~/.local/bin`, add it to your PATH if needed:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 3) Manual Install

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

- **Throughput**: Generation / Prompt tokens/s with sparklines
- **KV Cache**: GPU KV cache usage
- **Prefix Cache**: Hit rate
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

---

## 🇯🇵 日本語

### インストール

#### 1) リリースからダウンロード（推奨）

```bash
# 最新バイナリをダウンロード
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64
chmod +x vllmtop-linux-x86_64
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop

# または tar.gz
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64.tar.gz
tar xzf vllmtop-linux-x86_64.tar.gz
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop
```

ARM64 サーバーの場合は `vllmtop-linux-aarch64` を使用してください。

#### 2) インストールスクリプトを使用

`tar.gz` には `install.sh` が含まれています。

```bash
tar xzf vllmtop-linux-x86_64.tar.gz
./install.sh              # /usr/local/bin（書き込み不可なら ~/.local/bin）
./install.sh --user       # ~/.local/bin へ
./install.sh --system     # /usr/local/bin へ
./install.sh --prefix /opt/bin  # 任意のパス
```

`~/.local/bin` にインストールした場合は PATH を追加してください。

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 3) 手動インストール

```bash
sudo install -m 755 ./vllmtop /usr/local/bin/vllmtop
# またはユーザー領域
install -Dm755 ./vllmtop ~/.local/bin/vllmtop
```

#### 権限

```bash
ls -l /var/run/docker.sock
# srw-rw---- root docker ...
docker ps   # これが動けば vllmtop も動きます
```

`docker ps` が権限エラーになる場合は Docker グループに追加するか `sudo` で実行してください。

```bash
sudo usermod -aG docker $USER
newgrp docker
# または
sudo vllmtop
```

#### アンインストール

```bash
./uninstall.sh              # /usr/local/bin と ~/.local/bin から削除
./uninstall.sh --user       # ~/.local/bin のみ
./uninstall.sh --system     # /usr/local/bin のみ
# 手動
sudo rm -f /usr/local/bin/vllmtop
rm -f ~/.local/bin/vllmtop
```

### 使い方

#### 対話式 — コンテナ選択

```bash
vllmtop
```

実行中の全コンテナが表示されます。

```
vllmtop

Select Docker container

  NAME                    IMAGE
> qwen38                  provsalt/qwen3.8-flash...
  glm53                   glm53-vllm:latest

↑/↓ 選択   Enter 開く   q 終了
```

対象で `Enter`。

#### 直接接続 — 選択画面をスキップ

```bash
vllmtop qwen38
vllmtop glm53
vllmtop a1b2c3d4e5f6   # コンテナIDの前方一致も可
```

#### オプション

```bash
vllmtop --help
vllmtop --version
vllmtop --socket /var/run/docker.sock
vllmtop qwen38 --tail 500
vllmtop --socket /other/docker.sock qwen38 --tail 200
```

| オプション | 説明 | デフォルト |
|------------|------|------------|
| `CONTAINER` | 直接接続するコンテナ名/ID | なし（選択画面） |
| `--socket PATH` | Docker ソケットパス | `/var/run/docker.sock` |
| `--tail N` | 起動時に取得する過去ログ行数 | `100` |

#### ダッシュボード

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

- **Throughput**: Generation / Prompt トークン/秒とスパークライン
- **KV Cache**: GPU KVキャッシュ使用率
- **Prefix Cache**: ヒット率
- **Requests**: 実行中 / 待機中
- **Speculative Decoding**: 採択率、平均採択長、スループット
- **PLE mmap**: 処理時間、gather時間、行数、サイズ

#### キー

| キー | 動作 |
|------|------|
| `↑` `↓` / `k` `j` | リスト移動（選択画面）、ログスクロール（ログ画面） |
| `Enter` | コンテナを開く |
| `c` | コンテナ切り替え（選択画面に戻る） |
| `l` | ログ表示切り替え / ダッシュボードに戻る |
| `p` | 一時停止 / 再開 |
| `r` | 履歴リセット（選択画面ではリスト更新） |
| `q` / `Esc` | 終了（ログ画面ではダッシュボードに戻る） |
| `Ctrl+C` | 終了 |

---

## 🇨🇳 中文

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
