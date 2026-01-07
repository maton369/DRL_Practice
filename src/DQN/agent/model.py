import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from util import idx2mask


class QNetwork(nn.Module):
    """
    Deep Q-Network（DQN）用の Q 関数近似器を管理するクラス（PyTorch 版）。

    元コード（Keras版）がやっていたこと:
    - main_network（オンラインネットワーク）: Q(s, a) を出す
    - target_network（ターゲットネットワーク）: TD 目標値 y を計算するために使う（安定化）
    - trainable_network（状態 + action_mask を入れて Q(s,a_taken) を取り出して学習するモデル）
      → Keras では Dot で「選択行動の Q」を抽出し、それを MSE で学習していた

    PyTorch版での方針:
    - main_network / target_network は同じ MLP（全結合ネット）
    - 「選択行動の Q 抽出」は、Keras の Dot を真似して
        (Q(s, :) * one-hot(action)) の和
      としてもよいが、
      PyTorch では一般に gather を使って
        Q(s, action_index)
      を直接取り出すのが自然で高速・安全である。
      ただし元コードとの対応関係が分かりやすいように、
      action_mask（one-hot）も作れるように残しつつ、
      学習計算では gather を用いる。

    double_mode:
    - True  : Double DQN
        argmax は main_network で選び、その行動の値は target_network で評価する
    - False : 通常の DQN
        target_network の max_a Q(s',a) をそのまま使う
    """

    def __init__(
        self,
        dim_state,
        actions_list,
        gamma=0.99,
        lr=1e-3,
        double_mode=True,
        device=None,
    ):
        super().__init__()

        # 状態ベクトルの次元（例: 観測が 4 次元なら dim_state=4）
        self.dim_state = dim_state

        # 行動を列挙したリスト（例: actions_list=[0,1] など）
        # このクラス内では「行動そのもの」ではなく「インデックス（0..action_len-1）」で扱う
        self.actions_list = actions_list
        self.action_len = len(actions_list)

        # 割引率 γ
        self.gamma = gamma

        # Double DQN を使うかどうか
        self.double_mode = double_mode

        # 使用デバイス（GPUがあればGPU）
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # main_network / target_network を構築
        # 元コードの build_graph() と同じ層サイズの決め方を踏襲
        self.main_network = self._build_mlp().to(self.device)
        self.target_network = self._build_mlp().to(self.device)

        # ターゲットネットワークは最初に main と同じ重みにしておくのが一般的
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()  # ターゲットは学習しないので eval モードに固定（BN/Dropoutがある場合に重要）

        # 最適化手法（Adam）
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=lr)

        # 損失関数（Keras版は mse）
        self.criterion = nn.MSELoss()

    # -------------------------
    # ネットワーク構築（MLP）
    # -------------------------
    def _build_mlp(self):
        """
        元コード build_graph() の構造に対応する MLP を PyTorch で作る。

        Keras版:
            Dense(nb_dense_1, relu)
            Dense(nb_dense_2, relu)
            Dense(nb_dense_3, relu)
            Dense(action_len, linear)

        層サイズ:
            nb_dense_1 = dim_state * 10
            nb_dense_3 = action_len * 10
            nb_dense_2 = sqrt((dim_state*10) * (action_len*10)) のような中間サイズ

        出力:
            Q(s, :) のベクトル（shape=(batch, action_len)）
        """
        nb_dense_1 = self.dim_state * 10
        nb_dense_3 = self.action_len * 10

        # Keras版の nb_dense_2 の定義を踏襲
        nb_dense_2 = int(np.sqrt(self.action_len * 10 * self.dim_state * 10))

        model = nn.Sequential(
            nn.Linear(self.dim_state, nb_dense_1),
            nn.ReLU(),
            nn.Linear(nb_dense_1, nb_dense_2),
            nn.ReLU(),
            nn.Linear(nb_dense_2, nb_dense_3),
            nn.ReLU(),
            nn.Linear(nb_dense_3, self.action_len),  # linear（活性化なし）
        )
        return model

    # -------------------------
    # ターゲットネットワーク同期（soft update）
    # -------------------------
    def sync_target_network(self, soft):
        """
        ターゲットネットワークを main_network に近づける（Polyak averaging / soft update）。

        Keras版:
            target = (1-soft)*target + soft*main

        soft の意味:
        - soft=1.0 なら完全に同じ重みにする（hard update）
        - soft が小さいほどゆっくり追従し、学習が安定しやすい

        注意:
        - torch.no_grad() でパラメータ更新を “勾配計算なし” で行う
        """
        with torch.no_grad():
            for tgt_param, main_param in zip(
                self.target_network.parameters(), self.main_network.parameters()
            ):
                tgt_param.data.mul_(1.0 - soft)
                tgt_param.data.add_(soft * main_param.data)

    # -------------------------
    # 1バッチ更新（DQN / Double DQN）
    # -------------------------
    def update_on_batch(self, exps):
        """
        経験バッチ（experience replay のサンプル）を使って main_network を1回更新する。

        exps:
            [(state, action, reward, next_state, done), ...] のリスト
            - state      : 状態ベクトル（dim_state 次元）
            - action     : actions_list の要素（例: 0/1 など）
            - reward     : 即時報酬（スカラー）
            - next_state : 次状態ベクトル
            - done       : 終端フラグ（1なら終端、0なら継続）※元コード踏襲

        学習で作る TD 目標値（1-step TD）:
            y = r + γ * (1-done) * future_return

        future_return:
            - Double DQN:
                a* = argmax_a Q_main(next_state, a)
                future_return = Q_target(next_state, a*)
            - 通常DQN:
                future_return = max_a Q_target(next_state, a)

        損失:
            loss = MSE( Q_main(state, action_taken), y )

        戻り値:
            (loss_value, td_error_mae)
            - Keras版 train_on_batch の metrics=['mae'] 相当として、
              ここでは |Q - y| の平均（MAE）を td_error として返す。
        """

        # ------------------------------------------------------------
        # 1) バッチを分解して numpy 配列へ
        # ------------------------------------------------------------
        (state, action, reward, next_state, done) = zip(*exps)

        # 行動を「actions_list 内のインデックス」に変換
        # 例: actions_list=[-1, +1] で action=+1 のとき index=1
        action_index = [self.actions_list.index(a) for a in action]

        # Keras版は action_mask を作って Dot していたので、同様の one-hot も用意できる
        # （ただし PyTorch では gather の方が自然なので、学習では gather を使う）
        action_mask = np.array(
            [idx2mask(a, self.action_len) for a in action_index], dtype=np.float32
        )

        # numpy 化
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        # ------------------------------------------------------------
        # 2) torch Tensor に変換して device に載せる
        # ------------------------------------------------------------
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        )  # (B, dim_state)
        next_state_t = torch.tensor(
            next_state, dtype=torch.float32, device=self.device
        )  # (B, dim_state)

        # reward, done は shape=(B,) のベクトルとして扱う
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)  # (B,)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device)  # (B,)

        # 行動インデックス（gather 用）は long 型が必要
        action_index_t = torch.tensor(
            action_index, dtype=torch.long, device=self.device
        )  # (B,)

        # action_mask も Tensor 化（学習計算には必須ではないが、対応関係として残す）
        action_mask_t = torch.tensor(
            action_mask, dtype=torch.float32, device=self.device
        )  # (B, action_len)

        # ------------------------------------------------------------
        # 3) 次状態の future_return を計算（ターゲット側は勾配不要）
        # ------------------------------------------------------------
        with torch.no_grad():
            # Q_target(next_state, :) ・・・ターゲットネットの出力（B, action_len）
            next_target_q = self.target_network(next_state_t)

            # Q_main(next_state, :) ・・・オンライン（main）ネットの出力（B, action_len）
            next_main_q = self.main_network(next_state_t)

            if self.double_mode:
                # Double DQN:
                #   next_action = argmax_a Q_main(next_state, a)
                #   future_return = Q_target(next_state, next_action)
                next_action = torch.argmax(next_main_q, dim=1)  # (B,)

                # gather で「各サンプルごとに選ばれた行動」のQだけ取り出す
                future_return = next_target_q.gather(
                    1, next_action.view(-1, 1)
                ).squeeze(
                    1
                )  # (B,)
            else:
                # 通常 DQN:
                #   future_return = max_a Q_target(next_state, a)
                future_return = torch.max(next_target_q, dim=1).values  # (B,)

            # TD 目標値 y:
            # done=1 なら将来項を消す（終端）
            y = reward_t + self.gamma * (1.0 - done_t) * future_return  # (B,)

        # ------------------------------------------------------------
        # 4) 現状態の Q(state, action_taken) を計算（勾配あり）
        # ------------------------------------------------------------
        # Q_main(state, :) を出す（B, action_len）
        q_all = self.main_network(state_t)

        # Keras版の Dot と同等の “マスクによる抽出” を書くと以下（参考）:
        #   q_taken_dot = (q_all * action_mask_t).sum(dim=1)
        # ただし gather の方がスッキリで速いので、学習は gather を採用する。
        q_taken = q_all.gather(1, action_index_t.view(-1, 1)).squeeze(1)  # (B,)

        # ------------------------------------------------------------
        # 5) 損失計算 -> 逆伝播 -> 更新
        # ------------------------------------------------------------
        # MSE( Q(s,a), y )
        loss = self.criterion(q_taken, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------------------------------------------------
        # 6) TD誤差（MAE 相当）を返す
        # ------------------------------------------------------------
        # Keras版の metrics=['mae'] は「平均絶対誤差」を返すのでそれに合わせる
        td_error_mae = torch.mean(torch.abs(q_taken.detach() - y)).item()

        return loss.item(), td_error_mae
