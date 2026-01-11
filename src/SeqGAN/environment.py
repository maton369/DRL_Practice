import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Discriminator(nn.Module):
    """
    SeqGAN の Discriminator（識別器）に相当するネットワーク。

    Keras 実装は以下の構成だった：
      Input(ids) -> Embedding -> LSTM -> Dropout -> Dense(sigmoid)

    PyTorch 版でも同等の計算グラフを実装する。
    - 入力: shape [B, T] の int64（トークンID列）
    - 出力: shape [B, 1] の確率（1=real の確率）
    """

    def __init__(
        self, vocab_size: int, emb_size: int, hidden_size: int, dropout: float
    ):
        super().__init__()

        # Embedding: token id -> 埋め込みベクトル
        # Keras の mask_zero=False に対応し、特に padding を “無視しない” 実装にしている。
        # （padding を無視したい場合は、pack_padded_sequence 等が必要）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_size,
        )

        # LSTM: 系列を要約する（最後の hidden state を使う）
        # Keras の LSTM(hidden_size) は「最後の出力のみ」を返すので、
        # PyTorch でも最終時刻の hidden state（h_n）を利用する。
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,  # 入力を [B, T, E] で受ける
        )

        # Dropout: 過学習抑制（Keras の Dropout(dropout)）
        self.dropout = nn.Dropout(p=dropout)

        # Dense(sigmoid): 2値分類の出力層
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        1) Embedding: [B, T] -> [B, T, E]
        2) LSTM: [B, T, E] -> h_n: [1, B, H]（1層なので layer=1）
        3) Dropout + Linear + Sigmoid: [B, H] -> [B, 1]
        """
        # token_ids は int64（LongTensor）が必要
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()

        emb = self.embedding(token_ids)  # [B, T, E]

        # LSTM の出力:
        # - output: 全時刻の出力 [B, T, H]（今回は使わない）
        # - (h_n, c_n): 最終時刻の hidden/cell
        _, (h_n, _) = self.lstm(emb)

        # h_n は [num_layers * num_directions, B, H]
        # 今回は 1層・片方向なので [1, B, H] -> [B, H]
        h_last = h_n[-1]

        x = self.dropout(h_last)
        logits = self.fc(x)  # [B, 1]
        prob = torch.sigmoid(logits)  # [B, 1]（Keras の Dense(sigmoid) と同等）

        return prob


class Environment:
    """
    SeqGAN の “環境” 側クラス（主に Discriminator を管理するラッパ）。

    Keras 版では
      - self.discriminator を Keras Model として保持
      - pre_train() で compile + fit_generator + save_weights
      - initialize() で load_weights + compile
    を行っていた。

    PyTorch 版では
      - self.discriminator を nn.Module として保持
      - optimizer / criterion を Python 側で保持
      - 学習ループは DataLoader からバッチを受け取り手書きで回す
    という構造になる。
    """

    def __init__(
        self,
        batch_size: int,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        T: int,
        dropout: float,
        lr: float,
        device: Optional[str] = None,
    ):
        # --- ハイパーパラメータ（Keras版の引数と同じ意味） ---
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.T = T
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr

        # --- device 設定 ---
        # device が None の場合は GPU があれば cuda、なければ cpu を使う
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # --- Discriminator ネットワーク構築 ---
        self.discriminator = Discriminator(
            vocab_size=self.vocab_size,
            emb_size=self.emb_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)

        # --- 損失関数（binary_crossentropy 相当） ---
        # Keras の 'binary_crossentropy' に対応するのは BCE 系。
        # ここでは「出力が sigmoid 済み」なので BCELoss を使用。
        # （logits を返す形に変えるなら BCEWithLogitsLoss が一般に安定）
        self.criterion = nn.BCELoss()

        # --- optimizer（initialize で学習率を変えるため、ここでは仮） ---
        self.optimizer = Adam(self.discriminator.parameters(), lr=self.lr)

    def _set_optimizer(self, lr: float):
        """
        学習率を指定して optimizer を作り直す。

        Keras 版の
          optimizer = Adam(lr)
          model.compile(optimizer, ...)
        に対応する。
        """
        self.optimizer = Adam(self.discriminator.parameters(), lr=lr)

    def pre_train(
        self, d_loader, d_pre_episodes: int, d_pre_weight: str, d_pre_lr: float
    ):
        """
        Discriminator を事前学習する。

        入力:
          d_loader: PyTorch DataLoader を想定
            - (token_ids, label) を返す
            - token_ids: [B, T] の LongTensor（文章ID列）
            - label: [B, 1] もしくは [B] の float（1=real, 0=fake）
          d_pre_episodes: 事前学習のエポック数
          d_pre_weight: 重み保存先
          d_pre_lr: 事前学習時の学習率

        Keras 版の
          compile(Adam(d_pre_lr), 'binary_crossentropy')
          fit_generator(..., epochs=d_pre_episodes)
          save_weights(...)
        に対応する。
        """
        self.discriminator.train()
        self._set_optimizer(d_pre_lr)

        for epoch in range(1, d_pre_episodes + 1):
            total_loss = 0.0
            n_batches = 0

            for token_ids, labels in d_loader:
                # device に転送
                token_ids = token_ids.to(self.device)
                labels = labels.to(self.device)

                # labels の形を [B, 1] に揃える（Keras の Dense(1) 出力に合わせる）
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                labels = labels.float()

                # forward
                probs = self.discriminator(token_ids)  # [B, 1]
                loss = self.criterion(probs, labels)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"[D pre-train] epoch={epoch}/{d_pre_episodes}  loss={avg_loss:.6f}")

        # 重み保存（Keras の save_weights 相当）
        os.makedirs(os.path.dirname(d_pre_weight) or ".", exist_ok=True)
        torch.save(self.discriminator.state_dict(), d_pre_weight)

    def initialize(self, d_pre_weight: str):
        """
        事前学習済みの重みを読み込み、RL フェーズ用の学習率で optimizer を再設定する。

        Keras 版の
          load_weights(d_pre_weight)
          compile(Adam(self.lr), 'binary_crossentropy')
        に対応する。
        """
        state_dict = torch.load(d_pre_weight, map_location=self.device)
        self.discriminator.load_state_dict(state_dict)

        self.discriminator.train()
        self._set_optimizer(self.lr)

    @torch.no_grad()
    def predict(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Discriminator の推論（報酬計算に使うことが多い）。

        Keras 版では env.discriminator.predict(Y) のように呼んでいた部分に対応する。

        戻り値:
          shape [B, 1] の確率（1=real）
        """
        self.discriminator.eval()
        token_ids = token_ids.to(self.device)
        probs = self.discriminator(token_ids)
        return probs.detach().cpu()
