import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Actor クラス（方策ネットワーク）の PyTorch 実装
# ============================================================
class Actor(nn.Module):
    """
    Actor（方策 π(a|s) を出力するネットワーク）を PyTorch で実装したクラス。

    もとの TensorFlow 版がしていたこと（要点）:
      - 入力: 状態 s（ベクトル）
      - 出力: 離散行動の確率分布 π(a|s)（softmax）
      - 学習: 方策勾配の基本形
          J(θ) = E[ log πθ(a|s) * advantage ]
        を最大化したいので、最小化問題として
          L(θ) = - E[ log πθ(a|s) * advantage ]
        を最小化する

    PyTorch 版での方針:
      - Keras/TF の placeholder/sess.run を廃止し、
        forward + loss計算 + backward + optimizer.step の標準ループにする
      - 数値安定のため、学習時は softmax→log ではなく log_softmax を直接使う
        （log(softmax(.)) を安定に計算できる）

    重要な前提:
      - 行動は actions_list に列挙された「離散集合」
      - advantage は外部（CriticやMC推定）から渡される想定
        ※advantage を作るネットワークが別にある場合、通常 advantage には勾配を流さない
    """

    def __init__(self, num_states, actions_list, learning_rate=1e-3, device=None):
        """
        引数:
            num_states: int
                状態ベクトルの次元（入力次元）

            actions_list: list
                取りうる行動の値のリスト
                例: [-1, 1] や [0, 1, 2] など

            learning_rate: float
                最適化の学習率
                元コードは RMSProp なので PyTorch でも RMSprop を使う

            device: torch.device or None
                None なら CUDA が使える場合 cuda、無ければ cpu を使う
        """
        super().__init__()

        self.num_states = num_states
        self.actions_list = list(actions_list)
        self.num_actions = len(self.actions_list)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------
        # ネットワークの層サイズ（元コードの決め方を踏襲）
        # ------------------------------------------------------------
        # TF版:
        #   num_dense_1 = num_states * 10
        #   num_dense_3 = num_actions * 10
        #   num_dense_2 = sqrt(num_dense_1 * num_dense_3)
        num_dense_1 = self.num_states * 10
        num_dense_3 = self.num_actions * 10
        num_dense_2 = int(np.sqrt(num_dense_1 * num_dense_3))

        # ------------------------------------------------------------
        # ネットワーク本体（tanh MLP → 行動の logits を出す）
        # ------------------------------------------------------------
        # TF/Keras版は最後に softmax を付けて prob を出していたが、
        # PyTorchでは「logits を出して、必要なときに softmax/log_softmax」を使うのが定石。
        self.fc1 = nn.Linear(self.num_states, num_dense_1)
        self.fc2 = nn.Linear(num_dense_1, num_dense_2)
        self.fc3 = nn.Linear(num_dense_2, num_dense_3)
        self.out = nn.Linear(num_dense_3, self.num_actions)

        # 元コードの optimizer = RMSProp(lr)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

        # device へ転送
        self.to(self.device)

    # ============================================================
    # forward（π(a|s) の確率を返す）
    # ============================================================
    def forward(self, state):
        """
        入力:
            state: torch.Tensor, shape=(B, num_states) もしくは (num_states,)
        出力:
            probs: torch.Tensor, shape=(B, num_actions)
                各行動の確率 π(a|s)

        注意:
            学習（loss計算）では log_softmax を使う方が安定なので、
            update() 内では logits から log_probs を計算する。
        """
        if state.dim() == 1:
            # (num_states,) を (1, num_states) に揃える（1サンプル推論を許可）
            state = state.unsqueeze(0)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        logits = self.out(x)

        # 推論用に確率へ変換
        probs = F.softmax(logits, dim=-1)
        return probs

    # ============================================================
    # 内部ユーティリティ: logits と log_probs を得る
    # ============================================================
    def _forward_logits_and_log_probs(self, state):
        """
        update で数値安定な log π(a|s) を使うための内部関数。

        log_probs は log_softmax を使うことで
            log(softmax(logits))
        を安定に計算する。
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        logits = self.out(x)

        log_probs = F.log_softmax(logits, dim=-1)  # shape=(B, num_actions)
        return logits, log_probs

    # ============================================================
    # 行動サンプリング（方策に従って1行動を返す）
    # ============================================================
    def predict(self, state):
        """
        もとの TF 版 predict(sess, state) に対応する「行動サンプリング」関数。

        入力:
            state: array-like, shape=(num_states,)
        出力:
            action: actions_list の要素（実際に環境へ渡す行動値）

        アルゴリズム:
          - π(a|s) を計算
          - Categorical 分布として扱い、確率に従ってサンプル
          - サンプルした「行動インデックス」を actions_list の値へ変換
        """
        # numpy -> torch
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        # 推論モード（勾配不要）
        self.eval()
        with torch.no_grad():
            probs = self.forward(state_t)  # shape=(1, num_actions)
            probs_1d = probs.squeeze(0)    # shape=(num_actions,)

            # 確率分布からサンプル（πに従う方策）
            dist = torch.distributions.Categorical(probs=probs_1d)
            action_index = int(dist.sample().item())

        # 学習モードへ戻す（外側が train/eval を管理しているなら不要だが安全側で戻す）
        self.train()

        # actions_list の値に変換
        return self.actions_list[action_index]

    # ============================================================
    # Actor の更新（方策勾配）
    # ============================================================
    def update(self, state, act_onehot, advantage):
        """
        もとの TF 版 update(sess, state, act_onehot, advantage) に対応する更新関数。

        入力:
            state:
                shape=(B, num_states) の numpy 配列または list
                ※1サンプルなら (num_states,) でも可

            act_onehot:
                shape=(B, num_actions) の one-hot 行列
                「実際に選択した行動」を表す。
                例: 行動2を選んだなら [0,0,1,0,...]

            advantage:
                shape=(B, 1) または (B,) の advantage
                例: A(s,a) や (G_t - V(s)) など。
                正ならその行動確率を上げ、負なら下げる方向に学習が進む。

        損失:
            TF 版は
                loss = - sum(log π(a|s) * onehot, axis=1) * advantage
                loss = mean(loss)
            なので、PyTorch でも同じ形を実装する。

        注意:
            advantage を別ネットワークから計算している場合、
            通常は advantage 側へ勾配を流さない（detach）想定。
            ここでは入力が numpy なので自然に勾配は流れない。
        """
        # numpy -> torch（バッチ対応）
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        act_onehot_t = torch.tensor(act_onehot, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantage, dtype=torch.float32, device=self.device)

        # advantage の shape を (B,) に揃える（(B,1) でも (B,) でも扱えるように）
        if adv_t.dim() == 2 and adv_t.size(1) == 1:
            adv_t = adv_t.squeeze(1)

        # ------------------------------------------------------------
        # 1) log π(a|s) を計算（数値安定な log_softmax を使う）
        # ------------------------------------------------------------
        _logits, log_probs = self._forward_logits_and_log_probs(state_t)  # (B, num_actions)

        # ------------------------------------------------------------
        # 2) 選択行動の log確率だけを取り出す
        #    one-hot との内積を取れば「選んだ行動の logπ」になる
        #       selected_logp = sum_a logπ(a|s) * onehot(a)
        # ------------------------------------------------------------
        selected_logp = torch.sum(log_probs * act_onehot_t, dim=1)  # shape=(B,)

        # ------------------------------------------------------------
        # 3) 方策勾配の損失
        #    目的: E[ logπ * advantage ] を最大化
        #    → 最小化問題として負号を付ける
        # ------------------------------------------------------------
        loss_per_sample = -(selected_logp * adv_t)  # shape=(B,)
        loss = torch.mean(loss_per_sample)

        # ------------------------------------------------------------
        # 4) 逆伝播して更新
        # ------------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())