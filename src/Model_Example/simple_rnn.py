"""
overview:
    PyTorch（torch）で、CSV の短文（映画コメント）を対象に
    Embedding + SimpleRNN + Dense（シグモイド）で2値分類する最小例。

    元の TensorFlow/Keras 実装（Tokenizer → pad_sequences → Embedding → SimpleRNN → Dense）
    の流れを、PyTorch で以下の形に置き換える。

    - Tokenizer / pad_sequences（Keras）:
        → 「語彙辞書を作って token id に変換」+「max_length に合わせてパディング」
          を Python で明示実装（本サンプルでは最小構成のため自前で書く）

    - Embedding, RNN, Dense（Keras）:
        → torch.nn.Embedding, torch.nn.RNN, torch.nn.Linear を用いて構築

    - loss/optimizer:
        → 2値分類なので torch.nn.BCEWithLogitsLoss を利用（数値安定）
          ※この場合、モデル最後に sigmoid を入れず「logits」を出す
          （BCEWithLogitsLoss が内部で sigmoid を含むため）

args:
    各種パラメータ設定値は、本コード中に明記される

input:
    movie_comment_sample.csv
      - 1列目: テキスト
      - 2列目: ラベル（0/1 を想定）

output:
    学習後に train データで loss と accuracy を表示し、重みを .pth で保存する

usage-example:
    python3 simple_rnn_torch.py
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

# =========================
# ハイパーパラメータ（元コードに合わせる）
# =========================
vocab_size = 10  # 「語彙数の上限」を 10 に制限（頻出語から採用）
max_length = 4  # 文章長を 4 トークンに揃える（足りない分は PAD）
embedding_dim = 2  # Keras: Embedding(vocab_size, 2, ...)
rnn_hidden = 3  # Keras: SimpleRNN(3, ...)
batch_size = 8  # データが小さいので適当（元コードは fit のデフォルト）
epochs = 1000
lr = 0.001  # Keras Adam のデフォルト（概ね 1e-3）

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
    最小の分かち書き（トークナイズ）。
    Keras Tokenizer はより賢い前処理もあるが、ここでは最小例として:
      - 小文字化
      - 空白で split
    のみ行う。

    実データでは、
      - 句読点除去
      - 正規化（記号、絵文字、数字など）
      - 日本語なら MeCab/Sudachi など
    を入れるのが一般的。
    """
    return text.lower().strip().split()


def build_vocab(texts: List[str], vocab_size: int) -> Dict[str, int]:
    """
    語彙辞書を作る。
    Keras Tokenizer(num_words=vocab_size) に相当して、
    “頻出語から vocab_size-2 個” を採用する。

    ここでは special token を用意する:
      0: PAD（パディング）
      1: UNK（未知語）
    よって、実単語に割り当てられるのは最大で vocab_size-2 個。

    戻り値:
      token -> id の辞書
    """
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))

    # 頻度順に並べる（最頻出から）
    most_common = counter.most_common(max(0, vocab_size - 2))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (token, _freq) in enumerate(most_common, start=2):
        vocab[token] = idx

    return vocab


def encode_and_pad(
    texts: List[str], vocab: Dict[str, int], max_length: int
) -> List[List[int]]:
    """
    texts を token id の列に変換し、max_length に合わせて padding する。
    Keras の:
        tokenizer.texts_to_sequences
        pad_sequences(..., maxlen=max_length, padding='post')
    に相当。

    padding='post' なので、末尾に PAD(0) を足す。
    長すぎる場合は末尾を切る（post-truncation と同等）。
    """
    padded = []
    for t in texts:
        tokens = basic_tokenize(t)
        ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

        # 長さ調整（切り捨て）
        ids = ids[:max_length]

        # 足りない分を PAD で埋める（post padding）
        if len(ids) < max_length:
            ids = ids + [vocab["<PAD>"]] * (max_length - len(ids))

        padded.append(ids)

    return padded


def preprocessing(texts: List[str], labels: List[int]):
    """
    元コードの preprocessing() に対応する処理。
    - vocab を構築
    - texts を id 化 + padding
    - train/test 分割（元コード同様: 先頭6件をtrain、7件目をtest）

    注意:
    - データが極端に少ないので、ここでは例として同じ分割を踏襲する。
    - 実務ではランダム分割や交差検証を使うのが普通。
    """
    vocab = build_vocab(texts, vocab_size=vocab_size)
    padded_docs = encode_and_pad(texts, vocab=vocab, max_length=max_length)

    train_padded_docs = padded_docs[:6]
    test_padded_docs = padded_docs[6:7]

    train_labels = labels[:6]
    test_labels = labels[6:7]

    return train_padded_docs, test_padded_docs, train_labels, test_labels, vocab


# =========================
# モデル定義（Embedding + SimpleRNN + Dense）
# =========================
class SimpleRNNClassifier(nn.Module):
    """
    Keras の
        Embedding(vocab_size, 2, input_length=max_length, mask_zero=True)
        SimpleRNN(3, activation='sigmoid')
        Dense(1, activation='sigmoid')
    に対応する PyTorch モデル。

    ただし PyTorch では以下のように設計するのが一般的:
    - 損失に BCEWithLogitsLoss を使う（数値安定）
      → 出力は sigmoid 前の logits を返す（モデル最後に sigmoid を入れない）

    mask_zero=True（PAD を無視）について:
    - Keras は 0 を mask として RNN に渡せるが、
      PyTorch の素の nn.RNN は mask を自動で扱わない。
    - 厳密に PAD を無視するには:
        - pack_padded_sequence を使う
        - もしくは attention / pooling を工夫する
      などが必要。
    - 今回は最小例として「PAD も埋め込みを通って RNN に入る」実装にしている。
      （データが短く小さい例では動作確認として十分）

    入力形状:
      x: (N, max_length) ・・・ token id 列

    Embedding 出力:
      (N, max_length, embedding_dim)

    nn.RNN は batch_first=True にすれば:
      入力 (N, L, E) を受け取れる
      出力:
        - out: (N, L, H)（各時刻の隠れ状態）
        - h_n: (1, N, H)（最後の時刻の隠れ状態）
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()

        # 単語ID -> ベクトルへの変換（埋め込み）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        # SimpleRNN 相当（PyTorch の nn.RNN はデフォルトが tanh）
        # Keras の activation='sigmoid' に合わせたい場合は nonlinearity='relu' のようにはできないので、
        # “RNNセル内部の活性” を完全一致させるのは簡単ではない。
        # ここでは「最小例として nn.RNN(tanh)」を使い、
        # Keras との差分はコメントで明示する。
        #
        # もし “sigmoid をセル内活性に使いたい” を厳密にやるなら、
        # - 独自セルを実装する
        # - nn.RNNCell を自作する
        # などが必要になる。
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh",
        )

        # 最終隠れ状態 -> 1次元ロジット
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: (N, L) の token id
        戻り値:
          logits: (N,) ・・・ BCEWithLogitsLoss に渡すスカラー出力
        """
        emb = self.embedding(x)  # (N, L, E)
        out, h_n = self.rnn(emb)  # h_n: (1, N, H)
        last_h = h_n.squeeze(0)  # (N, H)
        logits = self.fc(last_h).squeeze(1)  # (N,)
        return logits


def build_rnn_model(vocab_size: int):
    """
    Keras の build_rnn_model() 相当。
    """
    model = SimpleRNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=rnn_hidden,
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
    Keras の model.fit(...) 相当。
    - loss: BCEWithLogitsLoss（logits + 0/1 ラベル）
    - optimizer: Adam
    - verbose=0 相当として、ここでも途中出力は最低限にする
      （必要なら数十epochごとに print する）
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (N,)
            loss = criterion(logits, y)  # y は float の 0/1 が必要

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 元コードは verbose=0 なので基本は黙る。
        # 学習状況を少し見たい場合は間引いて表示する。
        if epoch % 200 == 0:
            print(f"epoch: {epoch:04d} loss: {loss.item():.4f}")


@torch.no_grad()
def evaluate(model, data_loader):
    """
    Keras の model.evaluate(...) 相当。
    - loss と accuracy を算出して表示する。

    2値分類の accuracy:
      - logits を sigmoid に通して確率にし
      - 0.5 で閾値判定して 0/1 を得る
      - 正解ラベルと一致した割合を計算
    """
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs

        probs = torch.sigmoid(logits)  # (N,)
        preds = (probs >= 0.5).float()  # 0/1
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

    # --- 前処理（tokenize + vocab + padding + split） ---
    train_docs, test_docs, train_labels, test_labels, vocab = preprocessing(
        texts, labels
    )

    # --- Tensor 化 ---
    # 入力: token id の int64（Embedding の入力は LongTensor が必要）
    x_train = torch.tensor(train_docs, dtype=torch.long)
    x_test = torch.tensor(test_docs, dtype=torch.long)

    # ラベル: BCEWithLogitsLoss は float を期待する（0.0/1.0）
    y_train = torch.tensor(train_labels, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    # --- DataLoader 化 ---
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    # --- モデル構築 ---
    # vocab_size は「辞書のサイズ」（special token 含む）に合わせる
    model = build_rnn_model(vocab_size=len(vocab))

    # --- 学習 ---
    train(model, train_loader, epochs=epochs)

    # --- 重み保存（Keras の .h5 相当） ---
    torch.save(model.state_dict(), "simple_rnn_weight.pth")
    print("Saved weights to: simple_rnn_weight.pth")

    # --- 評価（元コードは train データで evaluate していたので踏襲） ---
    # ただし通常は test_loader でも評価して汎化を見る。
    evaluate(model, train_loader)
