"""
overview:
    PyTorch（torch）で MNIST を全結合ネットワーク（Dense/MLP）で学習する最小例。
    元の TensorFlow/Keras 版（Flatten→Dense(64,ReLU)→Dense(10,Softmax)）と同等の構造を、
    PyTorch の学習ループ（forward / loss / backward / step）で書き直している。

    注意:
    - Keras の categorical_crossentropy + one-hot とは違い、
      PyTorch では通常 CrossEntropyLoss を使い「ラベルは整数（0〜9）」のまま扱う。
    - CrossEntropyLoss は内部で log-softmax を含むため、モデルの最後に softmax は付けない。
      （推論時に確率が欲しい場合だけ softmax を使う）

args:
    各種パラメータ設定値は、本コード中に明記される

output:
    エポックごとに train loss / train acc / test loss / test acc を表示する

usage-example:
    python3 simple_mnist_dense_torch.py
"""

# =========================
# 依存ライブラリ
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =========================
# ハイパーパラメータ
# =========================
batch_size = 128
num_classes = 10
epochs = 12

# Keras の SGD() のデフォルト学習率は 0.01 相当なので、それに合わせる
lr = 0.01

# =========================
# デバイス設定
# =========================
# GPU が使える環境なら GPU を使う（学習が速い）
# ない場合は CPU で動く
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# データ読み込み・前処理
# =========================
def load_data_for_dense(batch_size: int):
    """
    Keras 版の load_data_for_dense() に相当する処理。

    Keras 版:
        - mnist.load_data() で numpy 配列を取得
        - float32 へ変換
        - 255 で割って 0〜1 に正規化
        - to_categorical で one-hot 化（10クラス）

    PyTorch 版:
        - torchvision.datasets.MNIST を使う
        - transform で前処理を書くのが一般的
            - ToTensor(): [0,255] の uint8 を float32 にし、[0,1] にスケールしてくれる
            - Flatten はモデル側でやるので transform では画像形状 (1,28,28) のままでOK
        - ラベルは one-hot にはせず「整数のまま」（CrossEntropyLoss が前提）
    """
    transform = transforms.ToTensor()  # 画像を float32 Tensor にして [0,1] に正規化

    train_ds = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # DataLoader:
    # - batch_size ごとにミニバッチを作る
    # - shuffle=True は学習データを毎エポックでシャッフルして汎化を助ける
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =========================
# モデル定義
# =========================
class SimpleDenseMNIST(nn.Module):
    """
    Keras の
        Flatten(input_shape=(28, 28))
        Dense(64, activation='relu')
        Dense(10, activation='softmax')
    に対応する PyTorch モデル。

    重要ポイント:
    - PyTorch の CrossEntropyLoss を使う場合、最後に softmax を付けない（logits を出す）
      CrossEntropyLoss は内部で softmax + log をやるので、softmax を二重にすると学習が崩れる。
    """

    def __init__(self):
        super().__init__()

        # Flatten: (N,1,28,28) -> (N, 784)
        self.flatten = nn.Flatten()

        # 全結合: 784 -> 64
        self.fc1 = nn.Linear(28 * 28, 64)

        # 活性化関数（ReLU）
        self.relu = nn.ReLU()

        # 全結合: 64 -> 10（クラス数）
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        forward は「入力 x から出力（logits）まで」を定義する。

        x の形:
        - torchvision MNIST は (N, 1, 28, 28)
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)  # ここは softmax しない（loss 側で処理する）
        return logits


def build_dense_model():
    """
    Keras の build_dense_model() に相当。
    モデルを作って device に載せ、構造を表示する。
    """
    model = SimpleDenseMNIST().to(device)

    # Keras の model.summary() の代わりに、簡易的に構造とパラメータ数を表示する
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    return model


# =========================
# 学習・評価ループ
# =========================
def accuracy_from_logits(logits, labels):
    """
    logits: (N,10) の未正規化スコア
    labels: (N,) の正解クラス（0〜9）

    予測は argmax を取ればよい（softmax を取らなくても argmax は同じ）。
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


@torch.no_grad()
def evaluate(model, data_loader, criterion):
    """
    テスト（または検証）用の評価関数。
    - 勾配は不要なので torch.no_grad() を付ける（高速・省メモリ）
    - model.eval() で Dropout/BatchNorm がある場合に推論モードになる
      （今回のモデルには無いが、習慣として入れる）
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        # loss はバッチ平均なので、全体平均にするならバッチサイズを掛けて足し合わせる
        batch_size_local = y.size(0)
        total_loss += loss.item() * batch_size_local

        c, t = accuracy_from_logits(logits, y)
        total_correct += c
        total_count += t

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def train(model, train_loader, test_loader, epochs=10):
    """
    Keras の model.fit(...) に相当する学習ループ。

    Keras 版は compile で
        optimizer=SGD()
        loss=categorical_crossentropy
    を設定して fit で学習していた。

    PyTorch では明示的に以下を毎バッチ繰り返す:
        - forward
        - loss 計算
        - backward（勾配計算）
        - optimizer.step（パラメータ更新）
        - optimizer.zero_grad（勾配リセット）
    """
    # 損失関数:
    # - CrossEntropyLoss は「logits」と「整数ラベル」を取る
    # - 内部で log-softmax を使うので、モデル出力は logits のままでOK
    criterion = nn.CrossEntropyLoss()

    # 最適化手法:
    # - Keras の SGD デフォルトに合わせて lr=0.01
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # --- forward ---
            logits = model(x)
            loss = criterion(logits, y)

            # --- backward ---
            # まず勾配をゼロクリア（前のバッチの勾配が残るのを防ぐ）
            optimizer.zero_grad()

            # 誤差逆伝播で勾配を計算
            loss.backward()

            # パラメータ更新
            optimizer.step()

            # --- logging 用の集計 ---
            batch_size_local = y.size(0)
            total_loss += loss.item() * batch_size_local

            c, t = accuracy_from_logits(logits, y)
            total_correct += c
            total_count += t

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        # テスト評価
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print(
            f"epoch: {epoch:02d} "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}"
        )


# =========================
# エントリポイント
# =========================
if __name__ == "__main__":
    # データ読み込み
    train_loader, test_loader = load_data_for_dense(batch_size=batch_size)

    # モデル構築
    model = build_dense_model()

    # 学習
    train(model, train_loader, test_loader, epochs=epochs)

    # 重み保存（Keras の .h5 相当）
    # - PyTorch では state_dict（重みだけ）を保存するのが一般的
    # - 再利用時は同じモデル定義を用意して load_state_dict する
    torch.save(model.state_dict(), "simple_mnist_dense_weight.pth")

    print("Saved weights to: simple_mnist_dense_weight.pth")
