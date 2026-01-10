import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyEstimator(nn.Module):
    """
    PolicyEstimator（PyTorch版）
    連続行動（continuous action）環境向けの「ガウス方策（対角共分散）」を表す Actor 実装。

    もとの TensorFlow/Keras 版がやっていたことを、そのまま PyTorch の流儀で置き換えている。

    ─────────────────────────────────────────────
    1) 方策の形（モデル化）
    ─────────────────────────────────────────────
    状態 s を入力すると、行動 a の確率分布 π(a|s) のパラメータを出力する。

      - 平均:      μ(s)        shape=(dim_action,)
      - 分散の対数: log_var(s)  shape=(dim_action,)

    各行動次元は独立（対角共分散）と仮定して

        a ~ Normal( μ(s), diag(σ(s)^2) )

    とする。ここで

        σ^2 = exp(log_var)
        σ   = sqrt(exp(log_var))

    である。

    ─────────────────────────────────────────────
    2) 学習（方策勾配: REINFORCE with baseline）
    ─────────────────────────────────────────────
    advantage A（例: G_t - V(s_t)）を重みとして、選択した行動が出やすくなるように更新する。

    目的（最大化したい量のイメージ）:
        E[ log π(a|s) * A ]

    実装上は最小化問題として
        loss = - E[ log π(a|s) * A ]
    を最小化する。

    ─────────────────────────────────────────────
    3) 重要な実装メモ
    ─────────────────────────────────────────────
    - log_prob は多次元行動の各次元ぶんが返るため、通常は action 次元で sum して
      「1サンプルあたりの log π(a|s)（スカラー）」にするのが一般的。
      もとの TF 版は次元方向の sum を明示していなかったが、ここでは意図を明確にするため sum する。

    - μ を tanh にしているのは、環境の action が [-1,1] に近い範囲を想定しているため。
      ただし「サンプリング後の a」は正規分布なので範囲外へ出る可能性がある。
      元コードもクリップしていないため、ここでも基本は同じ挙動にしている。
      必要なら predict() 側で action_space に合わせて clip する。

    - log_var も tanh にしているため、log_var が概ね [-1,1] に収まりやすい。
      これは分散が極端になりにくく探索が暴れにくい一方、探索不足になる可能性もある。
    """

    def __init__(self, dim_state, dim_action, leaning_rate=1e-3, device=None):
        """
        引数:
          dim_state:
            状態ベクトル次元
          dim_action:
            行動ベクトル次元（連続行動）
          leaning_rate:
            学習率（元コードの変数名に合わせているが意味は learning_rate）
          device:
            torch.device を明示したい場合に指定（未指定なら自動でCUDAがあればCUDA）
        """
        super().__init__()

        self.dim_state = int(dim_state)
        self.dim_action = int(dim_action)
        self.leaning_rate = float(leaning_rate)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ─────────────────────────────────────────
        # ネットワーク構造（元TF版のユニット数設計を踏襲）
        # ─────────────────────────────────────────
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = self.dim_action * 10
        nb_dense_2 = int(np.sqrt(nb_dense_1 * nb_dense_3))

        # MLP（tanh）
        self.fc1 = nn.Linear(self.dim_state, nb_dense_1)
        self.fc2 = nn.Linear(nb_dense_1, nb_dense_2)
        self.fc3 = nn.Linear(nb_dense_2, nb_dense_3)

        # 出力ヘッド（平均と log 分散）
        self.mu_head = nn.Linear(nb_dense_3, self.dim_action)
        self.log_var_head = nn.Linear(nb_dense_3, self.dim_action)

        # 最適化（元TF版は RMSProp）
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.leaning_rate)

        # デバイスへ
        self.to(self.device)

    def forward(self, state):
        """
        入力:
          state: torch.Tensor
            shape=(B, dim_state) または (dim_state,)

        出力:
          mu:      shape=(B, dim_action)
          log_var: shape=(B, dim_action)

        ※ 推論時はこの出力から Normal 分布を作って action をサンプルする。
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, dim_state)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        # 平均μと log_var を tanh で制約（元TF版の挙動を再現）
        mu = torch.tanh(self.mu_head(x))
        log_var = torch.tanh(self.log_var_head(x))
        return mu, log_var

    def _dist(self, state):
        """
        state からガウス分布 Normal(μ, σ) を構築する内部関数。

        注意:
          - σ^2 = exp(log_var)
          - σ   = sqrt(exp(log_var)) = exp(0.5 * log_var)
        """
        mu, log_var = self.forward(state)
        std = torch.exp(0.5 * log_var)  # 数値的に安定しやすい形
        dist = torch.distributions.Normal(loc=mu, scale=std)
        return dist, mu, log_var

    def logprob(self, state, action):
        """
        log π(a|s) を計算する。

        入力:
          state:  shape=(B, dim_state)
          action: shape=(B, dim_action)

        出力:
          logp: shape=(B, 1)
            1サンプルあたりの（多次元行動の）log確率。
            通常は各次元の log_prob を合計してスカラーにする。

        実装の意図:
          - torch.distributions.Normal の log_prob は (B, dim_action) を返す
          - 多次元行動の logπ はそれを action 次元で sum して (B,) にする
        """
        dist, _, _ = self._dist(state)

        # (B, dim_action): 各次元の log_prob
        logp_each_dim = dist.log_prob(action)

        # (B,): 多次元行動の logπ
        logp = logp_each_dim.sum(dim=-1)

        # (B,1) に揃えて返す（advantage と掛けやすい）
        return logp.unsqueeze(-1)

    def predict(self, state):
        """
        方策 π(a|s) から行動をサンプルして返す（推論用）。

        入力:
          state: np.ndarray など shape=(dim_state,)

        出力:
          action: np.ndarray shape=(dim_action,)
        """
        # numpy -> torch
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.eval()
        with torch.no_grad():
            dist, mu, log_var = self._dist(state_t)

            # 正規分布からサンプル（元TF版の np.random.normal と同じ役割）
            action_t = dist.sample().squeeze(0)  # (dim_action,)

            # 必要ならここでクリップ（元TF版はしていないのでデフォルトはしない）
            # action_t = torch.clamp(action_t, -1.0, 1.0)

        self.train()
        return action_t.detach().cpu().numpy()

    def update(self, state, action, advantage):
        """
        1バッチ分のデータで方策ネットワークを更新する。

        入力:
          state:
            shape=(B, dim_state) の list / np.ndarray / torch.Tensor
          action:
            shape=(B, dim_action) の list / np.ndarray / torch.Tensor
          advantage:
            shape=(B, 1) の list / np.ndarray / torch.Tensor
            ※ (B,) でも受け取れるように内部で整形する

        出力:
          loss（float）

        学習の直感:
          - advantage が正のサンプル: logπ(a|s) を増やしたい → loss を下げる方向に更新
          - advantage が負のサンプル: logπ(a|s) を減らしたい → loss を下げる方向に更新
        """
        # 入力を torch に統一
        state_t = torch.tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        )
        action_t = torch.tensor(
            np.asarray(action), dtype=torch.float32, device=self.device
        )
        adv_t = torch.tensor(
            np.asarray(advantage), dtype=torch.float32, device=self.device
        )

        # advantage を (B,1) に揃える
        if adv_t.dim() == 1:
            adv_t = adv_t.unsqueeze(-1)

        # logπ(a|s): (B,1)
        logp = self.logprob(state_t, action_t)

        # 方策勾配の損失（最小化形）
        # loss = - E[ logπ * advantage ]
        loss = -(logp * adv_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
