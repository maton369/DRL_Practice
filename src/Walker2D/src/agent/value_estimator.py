import numpy as np
import torch
import torch.nn as nn


class ValueEstimator(nn.Module):
    """
        ValueEstimator（PyTorch版）
        状態価値関数 V(s) をニューラルネットで近似する Critic / Baseline の実装。

        このモジュールがやること:
          - 入力:  状態ベクトル s（連続値の観測）  shape=(dim_state,)
          - 出力:  状態価値 V(s)（スカラー）       shape=(1,)

        Actor-Critic / REINFORCE with baseline の文脈では、
          - Actor（Policy）が使う advantage を安定化するための baseline として V(s) を推定し、
          - V(s) はモンテカルロリターン G_t や TDターゲット y_t に回帰する
        という役割を担う。

        典型的な学習目的:
          - 目標値（target）を G_t あるいは TDターゲット y_t とすると、
            MSE により

    $$
    \mathcal{L}(\phi)=\mathbb{E}\left[(V_\phi(s)-target)^2\right]
    $$

        を最小化する。

        本実装は元の TensorFlow/Keras 版に合わせて
          - tanh を用いた MLP（3隠れ層）
          - 最終出力は線形（linear）
          - optimizer は Adam
        という構成を踏襲している。
    """

    def __init__(self, dim_state, leaning_rate=1e-3, device=None):
        """
        引数:
          dim_state:
            状態次元（観測ベクトルの次元）
          leaning_rate:
            学習率（元コードの変数名に合わせているが意味は learning_rate）
          device:
            torch.device を指定したい場合に指定（未指定なら CUDA があれば CUDA）
        """
        super().__init__()

        self.dim_state = int(dim_state)
        self.leaning_rate = float(leaning_rate)

        # 実行デバイス（GPU があるなら GPU）
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ------------------------------------------------------------
        # ネットワーク構造（元TF版のユニット数設計を踏襲）
        # ------------------------------------------------------------
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = 5
        nb_dense_2 = int(np.sqrt(nb_dense_1 * nb_dense_3))

        # tanh MLP
        self.fc1 = nn.Linear(self.dim_state, nb_dense_1)
        self.fc2 = nn.Linear(nb_dense_1, nb_dense_2)
        self.fc3 = nn.Linear(nb_dense_2, nb_dense_3)

        # 価値出力（スカラー）
        self.v_head = nn.Linear(nb_dense_3, 1)

        # optimizer（元TF版は Adam）
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.leaning_rate)

        # 損失（MSE）
        # TF版は squared_difference → mean なので MSE と同等
        self.criterion = nn.MSELoss()

        # デバイスへ移動
        self.to(self.device)

    def forward(self, state):
        """
        価値関数 V(s) の順伝播。

        入力:
          state: torch.Tensor
            shape=(B, dim_state) または (dim_state,)

        出力:
          v: torch.Tensor
            shape=(B, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, dim_state)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        v = self.v_head(x)  # linear
        return v

    def predict(self, state):
        """
        推論用: V(s) を返す（学習勾配は不要）。

        入力:
          state: np.ndarray など shape=(dim_state,)

        出力:
          v: np.ndarray shape=(1, 1)
            元TF版が sess.run で (1,1) を返す形に寄せている。
            学習側が [0][0] でスカラーを取り出す前提なら互換性が高い。
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.eval()
        with torch.no_grad():
            v = self.forward(state_t)  # (1,1)
        self.train()

        return v.detach().cpu().numpy()

    def update(self, state, target):
        """
        1バッチ分で ValueEstimator を更新する。

        入力:
          state:
            shape=(B, dim_state) の list / np.ndarray / torch.Tensor
          target:
            shape=(B, 1) の list / np.ndarray / torch.Tensor
            （モンテカルロリターン G_t、または TDターゲット y_t を想定）

        出力:
          loss（float）

        アルゴリズム的な意味:
          - Critic/Baseline を target に近づけることで
            advantage = target - V(s)
            の推定が改善し、Actor の方策勾配推定の分散が下がりやすくなる。
        """
        # 入力を torch.Tensor に統一
        state_t = torch.tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        )
        target_t = torch.tensor(
            np.asarray(target), dtype=torch.float32, device=self.device
        )

        # 予測値 V(s): (B,1)
        v = self.forward(state_t)

        # MSE: mean( (V(s) - target)^2 )
        loss = self.criterion(v, target_t)

        # 勾配更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
