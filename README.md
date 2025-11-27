# FLUX.1 [schnell] Text‑to‑Image + LoRA Testアプリ

🤗 Diffusers を使って **`black-forest-labs/FLUX.1-schnell`** をローカルで動かす、シンプルなクライアント / サーバーアプリです。  
React 製のフロントエンドと、FastAPI 製のバックエンド、さらにオプションで **LoRA** に対応しています。

- **バックエンド:** FastAPI (`flux-server/server.py`)
- **フロントエンド:** React + Vite (`flux-app/`)
- **ベースモデル:** `black-forest-labs/FLUX.1-schnell`（Hugging Face 上でライセンス承認が必要）

## 1. flux-server

プロジェクトルートから:

```bash
cd flux-server

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
````

FLUX.1 のモデルはサーバ起動時にダウンロードされますが、**事前にライセンスに同意しておく必要** があります。

1. HaggingFace → ログイン → [settings/tokens](https://huggingface.co/settings/tokens) → Token発行
2. HaggingFace → [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main) → Files & versions → Agree
3. コマンドラインから `(.venv)$ huggingface-cli login` → Tokenを入力

API の起動:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

動作確認:

* [http://localhost:8000/health](http://localhost:8000/health)
  → `{"status": "ok", ...}` のような JSON が返ってくれば OK です。

## 2. flux-app

別のターミナルで、プロジェクトルートから:

```bash
cd flux-app
npm install
npm run dev
```

ブラウザで以下にアクセス:

* [http://localhost:5173](http://localhost:5173)

フロントエンドは `http://localhost:8000` のバックエンドと通信するよう設定済みです。

## 3. 使い方

1. **バックエンド** を起動（`uvicorn ...` ポート `8000`）。
2. **フロントエンド** を起動（`npm run dev -- --port 5173`）。
3. ブラウザで **[http://localhost:5173](http://localhost:5173)** を開く。

左側のフォーム:

* **Prompt**
  生成したい画像のテキスト説明。
* **Width / Height**
  解像度（256–1536 px）。デフォルトは 768×768。
* **Inference steps**
  1–4。FLUX.1 [schnell] はこの範囲に最適化されています。
* **Max sequence length**
  トークン長の上限。通常は 256 のままで OK。
* **# Images**
  一度に生成する画像枚数（1–4）。
* **Seed**
  乱数シード。空欄なら毎回ランダム、数値を入れると再現性のある結果に。
* **Output format**
  PNG または JPEG。
* **LoRA**（任意）
  LoRA を使う場合に入力します（例は後述）。

右側の結果パネル:

* モデル名、デバイス、解像度、ステップ数、使用中の LoRA（あれば）を表示
* 生成された画像の一覧

  * それぞれのシード値（あれば）
  * 各画像のダウンロードリンク

> LoRA を使わずベースモデルだけ試したい場合は、LoRA の入力欄をすべて空にしておきます。

## 4. LoRA の使い方とサンプル

フォーム内の **LoRA** セクションには、次の 3 つのフィールドがあります:

* **LoRA model (repo or path)**
  Hugging Face のリポジトリ ID（例: `user/repo`）またはローカルディレクトリのパス。

* **LoRA weight file**（任意）
  LoRA の重みファイル名（**拡張子 `.safetensors` まで含めたフルファイル名**）。
  例: `my-style-lora.safetensors`
  空欄の場合、diffusers 側のデフォルトが使われるケースもあります。

* **LoRA strength**
  LoRA の強さ。

  * `0.0` … ほぼ効果なし
  * `1.0` … 標準の強さ
  * `0.7–1.3` あたりを中心に調整するのがおすすめ

**LoRA model** が空欄のときは LoRA は適用されません。

> なるべく **FLUX.1-schnell 用に学習された LoRA** を使うと結果が安定します。

### 5 LoRA サンプル一覧

下記は、UI の **LoRA** 入力欄にそのまま使えるサンプルです。  
各行には **LoRA の指定方法** と **実際に生成テストした際のプロンプト例** を記載しています。

| スタイル            | LoRA model                              | LoRA weight file                             | プロンプト例 |
|---------------------------|-----------------------------------------|----------------------------------------------|--------------|
| フォトリアリスティック | `hugovntr/flux-schnell-realism`        | `schnell-realism_v2.3.safetensors` または `schnell-realism_v1.safetensors` | *A cozy, moody kitchen at dusk, warm golden lighting, ultra detailed, 35mm photography* |
| フィギュア風 | `p1atdev/flux.1-schnell-pvc-style-lora` | `pvc-shnell-7250+7500.safetensors`           | *pvc figure, nendoroid, cute anime girl with blue hair standing in a cozy bedroom, soft lighting, full body* |
| ポートレート | `Octree/flux-schnell-lora`             | `flux-schnell-lora.safetensors`              | *A beautiful woman with a slight warm smile in a bustling cafe, 4k, be4u7y* |

> **LoRA weight file** は必ず `.safetensors` まで含めた名前で入力してください。

## 5. トラブルシューティング

* **モデルアクセスのエラー（401 / 403 など）**

  Hugging Face で `black-forest-labs/FLUX.1-schnell` のページを開き、
  利用規約に同意してから再度実行してください。

* **LoRA 読み込みエラー**

  * LoRA のリポジトリ ID が正しいか (`hugovntr/flux-schnell-realism` など)。
  * weight file 名が **完全一致** しているか（拡張子 `.safetensors` を含む）。
  * その LoRA が FLUX.1-schnell 向けか、モデルカードを確認してください。

