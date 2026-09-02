# vllmtop

btop風 vLLM モニタリング TUI — Docker ログをリアルタイムに解析し、スループット、KVキャッシュ、リクエスト状態を一目で表示

> 単一バイナリ、Docker ソケットだけで動作

**Language:** [한국어](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [中文](README.zh.md)

---

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
