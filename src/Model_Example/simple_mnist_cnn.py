"""
overview:
    PyTorch（torch）で MNIST を CNN（畳み込みニューラルネットワーク）で学習する最小例。
    元の TensorFlow/Keras 版（Conv32→Conv64→MaxPool→Flatten→Dense(10,Softmax)）と同等の構造を、
    PyTorch の Dataset/DataLoader と学習ループで書き直している。

    重要な差分（Keras → PyTorch）:
    - Keras は入力形状が (N, 28, 28, 1) の NHWC だが、PyTorch は (N, 1, 28, 28) の NCHW が標準。
    - Keras の categorical_crossentropy は one-hot ラベルを想定することが多いが、
      PyTorch の CrossEntropyLoss は整数ラベル（0〜9）を想定する。
    - CrossEntropyLoss は内部で log-softmax を含むので、モデル最後に softmax を付けない（logits を出す）。

args:
    各種パラメータ設定値は、本コード中に明記される

output:
    エポックごとに train loss / train acc / test loss / test acc を表示する

usage-example:
    python3 simple_mnist_cnn_torch.py
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

# Keras の Adam() のデフォルト学習率は 0.001 相当なので合わせる
lr = 0.001

# =========================
# デバイス設定
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# データ読み込み・前処理
# =========================
def load_data_for_cnn(batch_size: int):
    """
    Keras 版 load_data_for_cnn() に対応する処理。

    Keras 版の要点:
    - mnist.load_data() で (N,28,28) の numpy を取得
    - reshape(-1,28,28,1) で NHWC（チャンネル最後）にする
    - 255 で割って [0,1] に正規化
    - to_categorical で one-hot 化

    PyTorch 版の要点:
    - torchvision.datasets.MNIST で取得すると、画像は PIL -> transform で Tensor 化される
    - transforms.ToTensor() は
        - float32 Tensor に変換
        - 値域を [0,1] に正規化
        - 形状を (C,H,W) = (1,28,28) にしてくれる
    - ラベルは one-hot ではなく整数のまま（CrossEntropyLoss が前提）
    """
    transform = transforms.ToTensor()

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

    # DataLoader はミニバッチを作り、学習時は shuffle=True でシャッフルする
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =========================
# モデル定義（CNN）
# =========================
class SimpleCNNMNIST(nn.Module):
    """
    Keras の
        Conv2D(32, 3x3, relu, input_shape=(28,28,1))
        Conv2D(64, 3x3, relu)
        MaxPool2D(2x2)
        Flatten()
        Dense(10, softmax)
    に対応する PyTorch 実装。

    形状の流れ（PyTorch は NCHW）:
        入力: (N, 1, 28, 28)
        conv1 (3x3, padding=0): (N, 32, 26, 26)
        conv2 (3x3, padding=0): (N, 64, 24, 24)
        maxpool (2x2):         (N, 64, 12, 12)
        flatten:                (N, 64*12*12)
        fc:                     (N, 10)  ※logits

    注意:
    - Keras と同様に padding='valid' 相当（padding=0）にしているため空間サイズが縮む。
    - CrossEntropyLoss を使うので、最後は softmax せず logits を返す。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 1ch -> 32ch の畳み込み（カーネル 3x3）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

        # 32ch -> 64ch の畳み込み（カーネル 3x3）
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # ReLU 活性化
        self.relu = nn.ReLU()

        # 2x2 の最大プーリング
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Flatten: (N,64,12,12) -> (N, 64*12*12)
        self.flatten = nn.Flatten()

        # 全結合: 64*12*12 -> 10
        self.fc = nn.Linear(64 * 12 * 12, num_classes)

    def forward(self, x):
        """
        forward は入力から出力（logits）までの計算グラフを定義する。
        """
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.flatten(x)

        logits = self.fc(x)  # softmax は付けない
        return logits


def build_cnn_model():
    """
    Keras の build_cnn_model() に相当。
    PyTorch では compile がないので、
    - モデル構築
    - 損失関数と最適化手法は train() 内で用意
    という分担にしている。
    """
    model = SimpleCNNMNIST(num_classes=num_classes).to(device)

    # Keras の model.summary() の簡易代替: モデルとパラメータ数を表示
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")

    return model


# =========================
# 学習・評価関数
# =========================
def accuracy_from_logits(logits, labels):
    """
    logits: (N,10)
    labels: (N,) 整数ラベル

    softmax を取らなくても argmax は変わらないので、
    推論クラスは argmax(logits) で求める。
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total


@torch.no_grad()
def evaluate(model, data_loader, criterion):
    """
    評価ループ:
    - torch.no_grad() で勾配計算を止め、メモリと計算を節約
    - model.eval() で推論モード（Dropout/BN がある場合に重要）
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

        bs = y.size(0)
        total_loss += loss.item() * bs

        c, t = accuracy_from_logits(logits, y)
        total_correct += c
        total_count += t

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def train(model, train_loader, test_loader, epochs: int = 10):
    """
    Keras の model.fit(...) に相当する学習ループ。

    Keras 版:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=epochs)

    PyTorch 版:
        - loss: CrossEntropyLoss（logits + 整数ラベル）
        - optimizer: Adam（デフォルト β, ε は Keras と概ね近い）
        - 1エポックごとに train/test の loss/acc を表示
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

            # --- backward & update ---
            optimizer.zero_grad()  # 勾配をリセット
            loss.backward()  # 逆伝播で勾配を計算
            optimizer.step()  # Adam で更新

            # --- logging ---
            bs = y.size(0)
            total_loss += loss.item() * bs

            c, t = accuracy_from_logits(logits, y)
            total_correct += c
            total_count += t

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

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
    # データロード
    train_loader, test_loader = load_data_for_cnn(batch_size=batch_size)

    # モデル構築
    model = build_cnn_model()

    # 学習
    train(model, train_loader, test_loader, epochs=epochs)

    # 重み保存（Keras の .h5 相当）
    # PyTorch では state_dict（重みのみ）を保存するのが一般的
    torch.save(model.state_dict(), "simple_mnist_cnn_weight.pth")
    print("Saved weights to: simple_mnist_cnn_weight.pth")
