import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Critic クラス（状態価値関数 V(s) の近似器）の PyTorch 実装
# ============================================================
class Critic(nn.Module):
    """
    Critic（状態価値関数 V(s) を近似するネットワーク）を PyTorch で実装したクラス。

    もとの TensorFlow 版がしていたこと（要点）:
      - 入力: 状態 s（ベクトル）
      - 出力: 状態価値 V(s)（スカラー）
      - 学習: 教師信号 target（例えば TD ターゲット）に対して
              平均二乗誤差（MSE）で回帰

    典型的な Actor-Critic における役割:
      - Critic は V(s) を推定し、Actor の学習に使う advantage を作る材料になる
        例: 1-step TD を使う場合
            target = r + γ(1-done) V(s')
            advantage = target - V(s)
      - つまり Critic が良い V 推定を持つほど advantage の分散が下がり、
        Actor の学習が安定しやすくなる

    PyTorch 版での方針:
      - TF1 の placeholder / Session / optimizer.minimize を廃止し、
        forward + loss + backward + optimizer.step の標準ループにする
      - もとのネットワーク構造（tanh MLP、層サイズの決め方）を踏襲する
      - 損失は MSELoss（TF版の squared_difference の平均に対応）

    注意:
      - 本クラス単体では target（教師信号）をどう作るかは扱わない。
        それは環境ループや学習コード側で計算して update に渡す。
    """

    def __init__(self, num_states, learning_rate=1e-3, device=None):
        """
        引数:
            num_states: int
                状態ベクトルの次元数（入力次元）

            learning_rate: float
                Adam の学習率

            device: torch.device or None
                None の場合は CUDA が使えるなら cuda、無ければ cpu を使用
        """
        super().__init__()

        self.num_states = num_states
        self.learning_rate = learning_rate
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ------------------------------------------------------------
        # ネットワークの層サイズ（元コード踏襲）
        # ------------------------------------------------------------
        # TF版:
        #   num_dense_1 = num_states * 10
        #   num_dense_3 = 5
        #   num_dense_2 = sqrt(num_dense_1 * num_dense_3)
        num_dense_1 = self.num_states * 10
        num_dense_3 = 5
        num_dense_2 = int(np.sqrt(num_dense_1 * num_dense_3))

        # ------------------------------------------------------------
        # Critic ネットワーク本体（tanh MLP → V(s) を出力）
        # ------------------------------------------------------------
        self.fc1 = nn.Linear(self.num_states, num_dense_1)
        self.fc2 = nn.Linear(num_dense_1, num_dense_2)
        self.fc3 = nn.Linear(num_dense_2, num_dense_3)

        # 出力はスカラー V(s)
        self.v_out = nn.Linear(num_dense_3, 1)

        # 最適化手法（TF版は AdamOptimizer）
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # TF版 squared_difference の平均に対応する MSE
        self.criterion = nn.MSELoss()

        # device へ転送
        self.to(self.device)

    # ============================================================
    # forward（状態価値 V(s) を返す）
    # ============================================================
    def forward(self, state):
        """
        入力:
            state: torch.Tensor
                shape=(B, num_states) または (num_states,)
        出力:
            v: torch.Tensor
                shape=(B, 1) の状態価値推定 V(s)

        注意:
            1サンプルの (num_states,) でも呼べるように、
            その場合は内部で (1, num_states) に整形する。
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        v = self.v_out(x)  # linear
        return v

    # ============================================================
    # Critic による予測（numpy入力 → numpy出力）
    # ============================================================
    def predict(self, state):
        """
        もとの TF 版 predict(sess, state) に対応する推論関数。

        入力:
            state: array-like, shape=(num_states,)
        出力:
            v: np.ndarray, shape=(1, 1)
                V(s) の推定値

        注意:
            - 学習時は forward を直接呼んでも良いが、
              外部コードが numpy 前提の場合に合わせて predict を用意する。
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.eval()
        with torch.no_grad():
            v = self.forward(state_t)  # shape=(1,1)
        self.train()

        return v.detach().cpu().numpy()

    # ============================================================
    # Critic の更新（MSE 回帰）
    # ============================================================
    def update(self, state, target):
        """
        もとの TF 版 update(sess, state, target) に対応する更新関数。

        入力:
            state:
                shape=(B, num_states) の numpy 配列または list
                ※1サンプルなら (num_states,) でも可

            target:
                shape=(B, 1) または (B,) の教師信号
                例: TD ターゲット
                    target = r + γ(1-done) V(s')
                あるいは MC 目標など

        アルゴリズム:
          1) V(s) を推定
          2) MSELoss(V(s), target) を計算
          3) 逆伝播してパラメータ更新

        戻り値:
            loss_value: float
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(target, dtype=torch.float32, device=self.device)

        # target の shape を (B,1) に揃える（(B,) でも受け取れるように）
        if target_t.dim() == 1:
            target_t = target_t.unsqueeze(1)

        # 予測
        v_pred = self.forward(state_t)  # (B,1)

        # 損失（MSE）
        loss = self.criterion(v_pred, target_t)

        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
