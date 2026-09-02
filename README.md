# vllmtop

btop 스타일 vLLM 모니터링 TUI — Docker 로그를 실시간으로 파싱해 처리량, KV 캐시, 요청 상태를 한눈에 표시

> 단일 바이너리, Docker 소켓만으로 동작

**Language:** [한국어](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [中文](README.zh.md)

---

### 설치

#### 원라인 설치 (GitHub) — 권장

가장 간단한 방법. GitHub에서 설치 스크립트를 바로 받아 실행합니다. 아키텍처(x86_64 / aarch64)를 자동 감지해 최신 릴리즈를 다운로드합니다.

```bash
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash
```

`wget` 만 있는 경우:

```bash
wget -qO- https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash
```

특정 버전 설치:

```bash
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | VLLMTOP_VERSION=v0.1.0 bash
```

옵션:

```bash
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash -s -- --user
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash -s -- --system
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/install.sh | bash -s -- --prefix /opt/bin
```

`~/.local/bin` 에 설치된 경우 PATH 추가가 필요할 수 있습니다.

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 릴리즈에서 직접 다운로드

GitHub에서 직접 받고 싶거나 오프라인 환경인 경우: [Releases](https://github.com/hanana-kick/vllmtop/releases) 에서 원하는 파일을 받아 설치하세요.

```bash
# 바이너리 직접 다운로드
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64
chmod +x vllmtop-linux-x86_64
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop

# 또는 tar.gz (install.sh 포함)
curl -LO https://github.com/hanana-kick/vllmtop/releases/latest/download/vllmtop-linux-x86_64.tar.gz
tar xzf vllmtop-linux-x86_64.tar.gz
sudo install -m 755 vllmtop-linux-x86_64 /usr/local/bin/vllmtop
# 또는 포함된 스크립트로
./install.sh
./install.sh --user
./install.sh --system
```

ARM64 서버인 경우 `vllmtop-linux-aarch64` 를 사용하세요.

수동 설치:

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
# 원라인으로 받은 경우에도 동일
curl -fsSL https://raw.githubusercontent.com/hanana-kick/vllmtop/main/uninstall.sh | bash
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

- **Header**: RUNNING/WAITING, uptime, last log 시간, 컨테이너 이름

- **Throughput**: Generation / Prompt 토큰 처리량
- **KV Cache**: GPU KV 캐시 사용률 (바 표시)
- **Prefix Cache**: 히트율 (바 표시)
- **Requests**: Running / Waiting 요청 수

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
