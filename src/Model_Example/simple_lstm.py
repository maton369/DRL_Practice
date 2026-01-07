"""
overview:
    PyTorch（torch）で、CSV の短文（映画コメント）を対象に
    Embedding + LSTM + Dense（2値分類）を行う最小例。

    元の TensorFlow/Keras 実装（Tokenizer → pad_sequences → Embedding → LSTM → Dense）
    の流れを PyTorch に置き換える。

    Keras 実装のポイント:
      - Tokenizer(num_words=vocab_size) で語彙上限を設定し、単語列をID列へ変換
      - pad_sequences(maxlen=max_length, padding='post') で末尾PADして固定長へ
      - Embedding(..., mask_zero=True) で PAD(0) を “系列モデルに無視させる”
      - LSTM(3, activation='sigmoid') で系列を処理
      - Dense(1, sigmoid) で確率（0〜1）を出して binary_crossentropy で学習

    PyTorch 版の設計（重要差分）:
      - 2値分類の損失は BCEWithLogitsLoss を使う（数値安定）
        → モデルの最後に sigmoid を入れず “logits” を返す
      - mask_zero 相当（PAD 無視）は pack_padded_sequence を使うのが定石
        → 今回は「最小例」として、PAD の長さ情報を計算し pack を使って LSTM に渡す
           （Keras の mask_zero=True に近い挙動になる）

args:
    各種パラメータ設定値は、本コード中に明記される

input:
    movie_comment_sample.csv
      - 1列目: テキスト
      - 2列目: ラベル（0/1）

output:
    学習後に train データで loss と accuracy を表示し、重みを .pth で保存する

usage-example:
    python3 simple_lstm_torch.py
"""

# =========================
# 依存ライブラリ
# =========================
import csv
from collections import Counter
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# =========================
# ハイパーパラメータ（元コードに合わせる）
# =========================
vocab_size = 10  # 語彙数の上限（頻出語から採用）
max_length = 4  # 系列長（トークン数）を 4 に固定
embedding_dim = 2  # Keras: Embedding(vocab_size, 2, ...)
lstm_hidden = 3  # Keras: LSTM(3, ...)
batch_size = 8  # データが小さいので適当（ミニバッチ学習の形を作る）
epochs = 1000
lr = 0.001  # Keras Adam のデフォルト相当

# =========================
# デバイス設定
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 学習・テストデータの読み込み
# =========================
def load_movie_comment_data() -> Tuple[List[str], List[int]]:
    """
    CSV からテキストとラベルを読み込む。
    元コードと同様に 1 行目はヘッダとして読み飛ばす。

    想定CSV:
        movie_comment_sample.csv
            text,label
            good movie,1
            ...
    """
    docs = csv.reader(open("src/Model_Example/movie_comment_sample.csv", encoding="utf-8"))
    next(docs, None)  # ヘッダをスキップ
    docs = list(docs)

    texts = [d[0] for d in docs]
    labels = [int(d[1]) for d in docs]
    return texts, labels


# =========================
# 前処理（Tokenizer + pad_sequences 相当）
# =========================
def basic_tokenize(text: str) -> List[str]:
    """
    最小のトークナイズ:
      - 小文字化
      - 前後空白除去
      - 空白で split

    実務では正規化・記号処理・日本語形態素解析なども入れることが多いが、
    ここでは「Keras Tokenizer の代替としての最小例」を重視している。
    """
    return text.lower().strip().split()


def build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
    """
    語彙辞書を構築する。
    Keras Tokenizer(num_words=vocab_size) の考え方に合わせ、
    「頻出語から vocab_size-2 個」を採用する。

    special token:
      0: <PAD> ・・・ padding 用
      1: <UNK> ・・・ 語彙外（未知語）用
    """
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))

    # 実単語として採用できるのは最大 vocab_size-2 個
    most_common = counter.most_common(max(0, vocab_size - 2))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (token, _freq) in enumerate(most_common, start=2):
        vocab[token] = idx

    return vocab


def encode_and_pad(
    texts: List[str], vocab: Dict[str, int], max_length: int
) -> List[List[int]]:
    """
    Keras の
      - tokenizer.texts_to_sequences
      - pad_sequences(..., maxlen=max_length, padding='post')
    に相当。

    - token を id に変換（語彙外は <UNK>）
    - 長ければ切り捨て（post truncation）
    - 短ければ末尾に <PAD>(0) を付ける（post padding）
    """
    padded = []
    for t in texts:
        tokens = basic_tokenize(t)
        ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

        # 切り捨て
        ids = ids[:max_length]

        # 末尾PAD
        if len(ids) < max_length:
            ids = ids + [vocab["<PAD>"]] * (max_length - len(ids))

        padded.append(ids)

    return padded


def compute_lengths(padded_docs: List[List[int]], pad_id: int = 0) -> List[int]:
    """
    pack_padded_sequence 用に「本来の系列長」を計算する。
    Keras の mask_zero=True（PADを無視）に近い挙動を作るために必要。

    ここでは「末尾に PAD が付く」前提（post padding）なので、
    先頭から走査して PAD に当たるまでの長さを取る。

    例:
      [5, 7, 0, 0] -> length=2
      [3, 4, 9, 0] -> length=3
      [0, 0, 0, 0] -> length=0（今回はデータ上ほぼ出ない想定）
    """
    lengths = []
    for seq in padded_docs:
        l = 0
        for tok_id in seq:
            if tok_id == pad_id:
                break
            l += 1
        # LSTM に 0 長を入れるのは扱いづらいので、最低1に丸める
        # （完全PADのみの入力が来ない想定なら実害はない）
        lengths.append(max(1, l))
    return lengths


def preprocessing(texts: List[str], labels: List[int]):
    """
    元コードの preprocessing() に対応。
    - vocab 構築
    - id 化 + padding
    - train/test 分割（元コード同様: 先頭6件をtrain、7件目をtest）
    - pack 用に lengths も返す
    """
    vocab = build_vocab(texts, vocab_size=vocab_size)
    padded_docs = encode_and_pad(texts, vocab=vocab, max_length=max_length)
    lengths = compute_lengths(padded_docs, pad_id=vocab["<PAD>"])

    train_padded_docs = padded_docs[:6]
    test_padded_docs = padded_docs[6:7]

    train_lengths = lengths[:6]
    test_lengths = lengths[6:7]

    train_labels = labels[:6]
    test_labels = labels[6:7]

    return (
        train_padded_docs,
        test_padded_docs,
        train_labels,
        test_labels,
        train_lengths,
        test_lengths,
        vocab,
    )


# =========================
# モデル定義（Embedding + LSTM + Dense）
# =========================
class SimpleLSTMClassifier(nn.Module):
    """
    Keras の
        Embedding(vocab_size, 2, input_length=max_length, mask_zero=True)
        LSTM(3, activation='sigmoid')
        Dense(1, activation='sigmoid')
    に対応する PyTorch 実装。

    重要差分:
    - PyTorch の nn.LSTM は内部活性（ゲートや候補）は固定の設計で、
      Keras の activation='sigmoid' を “完全に同一” には再現しない。
      ただし LSTM の標準的構造に沿って学習するという意味でアルゴリズムの本質は一致する。
    - 損失には BCEWithLogitsLoss を使うので、出力は sigmoid 前の logits を返す。

    PAD マスク:
    - Keras の mask_zero=True は PAD(0) を系列処理から除外する。
    - PyTorch では pack_padded_sequence を使って、PAD 部分を LSTM が処理しないようにする。
      forward には lengths を渡して pack する形にしている。
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()

        # 単語ID -> 埋め込みベクトル
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        # LSTM: 入力E次元 -> 隠れ状態H次元
        # batch_first=True で (N, L, E) を入力にできる
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # 最終隠れ状態 -> ロジット（1次元）
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        """
        x: (N, L) token id
        lengths: (N,) 実長（PAD 以外の長さ）

        手順:
        1) Embedding で (N, L, E) を得る
        2) pack_padded_sequence で PAD 部分を除外した “可変長系列” に変換
        3) LSTM に通すと、最後の隠れ状態 h_n が得られる
        4) h_n を Linear に通して logits を作る

        戻り値:
          logits: (N,) ・・・ BCEWithLogitsLoss に渡す
        """
        emb = self.embedding(x)  # (N, L, E)

        # pack することで、PAD 部分を LSTM が “時間方向に処理しない” ようにする
        # enforce_sorted=False にすると lengths の降順ソートが不要（小規模例では便利）
        packed = pack_padded_sequence(
            emb, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM 実行
        # packed_out は packed のまま返る（今回は使わない）
        # (h_n, c_n) は shape=(num_layers*directions, N, H)
        _packed_out, (h_n, _c_n) = self.lstm(packed)

        # 最終層の最終隠れ状態（1層・単方向なので (1,N,H)）
        last_h = h_n.squeeze(0)  # (N, H)

        logits = self.fc(last_h).squeeze(1)  # (N,)
        return logits


def build_lstm_model(vocab_size: int):
    """
    Keras の build_lstm_model() 相当。
    """
    model = SimpleLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=lstm_hidden,
    ).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    return model


# =========================
# 学習・評価ループ
# =========================
def train(model, train_loader, epochs: int = 1000):
    """
    Keras の model.fit(..., verbose=0) 相当。

    2値分類:
      - logits を出す
      - BCEWithLogitsLoss で損失計算
      - Adam で更新
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            logits = model(x, lengths)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 元コードは verbose=0 なので基本は黙るが、
        # 進捗確認用に間引き表示を入れる
        if epoch % 200 == 0:
            print(f"epoch: {epoch:04d} loss: {loss.item():.4f}")


@torch.no_grad()
def evaluate(model, data_loader):
    """
    Keras の model.evaluate(...) 相当。
    train データで loss/accuracy を出す挙動を踏襲する。
    """
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, lengths, y in data_loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_count += bs

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count

    print("-" * 10)
    print(f"loss: {avg_loss}")
    print(f"accuracy: {avg_acc * 100}")


# =========================
# エントリポイント
# =========================
if __name__ == "__main__":
    # --- CSV 読み込み ---
    texts, labels = load_movie_comment_data()

    # --- 前処理（tokenize + vocab + padding + split + lengths） ---
    (
        train_docs,
        test_docs,
        train_labels,
        test_labels,
        train_lengths,
        test_lengths,
        vocab,
    ) = preprocessing(texts, labels)

    # --- Tensor 化 ---
    # 入力: Embedding の入力なので int64（LongTensor）
    x_train = torch.tensor(train_docs, dtype=torch.long)
    x_test = torch.tensor(test_docs, dtype=torch.long)

    # lengths: pack 用の実長情報（int64 で OK）
    l_train = torch.tensor(train_lengths, dtype=torch.long)
    l_test = torch.tensor(test_lengths, dtype=torch.long)

    # ラベル: BCEWithLogitsLoss は float の 0/1 を期待
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    # --- DataLoader 化 ---
    # 1サンプルが (x, lengths, y) になるように TensorDataset を作る
    train_loader = DataLoader(
        TensorDataset(x_train, l_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, l_test, y_test), batch_size=batch_size, shuffle=False
    )

    # --- モデル構築 ---
    # vocab_size は「辞書サイズ（special token含む）」に合わせる
    model = build_lstm_model(vocab_size=len(vocab))

    # --- 学習 ---
    train(model, train_loader, epochs=epochs)

    # --- 重み保存（Keras の .h5 相当） ---
    torch.save(model.state_dict(), "simple_lstm_weight.pth")
    print("Saved weights to: simple_lstm_weight.pth")

    # --- 評価（元コードは train データで評価していたので踏襲） ---
    evaluate(model, train_loader)
