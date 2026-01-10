import os
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 目的:
#   TensorFlow(Keras + placeholder + sess.run) で書かれた
#   Actor-Critic エージェントを PyTorch に移植する。
#
# このコードがやっていること（アルゴリズムの要点）:
#   - Critic:  状態価値 V(s) を近似し、価値回帰（Huber loss）で学習する
#   - Actor:   方策 π(a|s) を近似し、方策勾配（REINFORCE形式）で学習する
#   - TD誤差:  δ = (ターゲット値) - V(s) を advantage として Actor に渡す
#   - ターゲット値:
#       - TD(0) 版（1-step bootstrap）と、
#       - 「報酬列を積算 + 末端価値を足す」multi-step（元コードの TD(λ) っぽい実装）
#     のどちらかを選ぶ（元コードは常に後者を使っている）
#
# 注意:
#   - TensorFlow版の API（save_graph/restore_graph/update_model/predict_loss 等）を
#     できる限り踏襲し、呼び出し側の修正を最小化する方針で実装している。
#   - PyTorch では「計算グラフは forward のたびに生成」されるので、
#     sess.run の代わりに loss.backward() → optimizer.step() を実行する。
# ============================================================


# ------------------------------------------------------------
# one-hot 変換（TensorFlow/Keras の to_categorical 代替）
# ------------------------------------------------------------
def to_onehot(
    indices: Union[np.ndarray, torch.Tensor], num_classes: int
) -> torch.Tensor:
    """
    行動インデックス列を one-hot 行列に変換する。

    入力:
      indices: [B]（int）
      num_classes: 行動数 |A|

    出力:
      onehot: [B, num_classes]（float32, 0/1）
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long)
    else:
        indices = indices.to(dtype=torch.long)

    onehot = F.one_hot(indices, num_classes=num_classes).to(dtype=torch.float32)
    return onehot


# ------------------------------------------------------------
# ネットワーク構築（TensorFlow版の build_model 相当）
# ------------------------------------------------------------
class CriticNet(nn.Module):
    """
    Critic: 状態 s を入力して状態価値 V(s) を出力するネットワーク。

    役割:
      - V(s) は「その状態から将来どれくらい報酬が得られそうか」の期待値
      - 教師信号（ターゲット値）に回帰して学習する（ここでは Huber loss）
    """

    def __init__(
        self, input_shape: Tuple[int, ...], hidden: int = 128, out_dim: int = 1
    ):
        super().__init__()
        self.flatten = nn.Flatten()

        # 入力次元（多次元入力でも flatten して MLP へ）
        in_dim = int(np.prod(input_shape))

        # できるだけ元コードの「シンプルなモデル」意図に沿った MLP
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),  # V(s) はスカラー
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)


class ActorNet(nn.Module):
    """
    Actor: 状態 s を入力して方策 π(a|s)（行動確率）を出力するネットワーク。

    役割:
      - π(a|s) は「この状態でどの行動を選ぶか」の確率分布
      - 方策勾配（log π(a|s) * advantage）で学習する
    """

    def __init__(
        self, input_shape: Tuple[int, ...], hidden: int = 128, num_actions: int = 2
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        in_dim = int(np.prod(input_shape))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.net(x)

        # 重要:
        #   TF/Keras版は actor_model が softmax 出力（確率）を返す想定。
        #   PyTorchでも確率を返すほうが呼び出し側の互換性が高い。
        prob = F.softmax(logits, dim=-1)
        return prob


def build_model(
    input_shape: Tuple[int, ...], num_value: int, num_action: int
) -> Tuple[nn.Module, nn.Module]:
    """
    TensorFlow版の build_model(input_shape, num_value, num_action) を置き換える関数。

    戻り値:
      critic_model: V(s) を出すモデル
      actor_model:  π(a|s) を出すモデル
    """
    # hidden ユニット数は固定にしているが、必要なら引数化可能
    critic_model = CriticNet(input_shape=input_shape, hidden=128, out_dim=num_value)
    actor_model = ActorNet(input_shape=input_shape, hidden=128, num_actions=num_action)
    return critic_model, actor_model


# ------------------------------------------------------------
# 損失関数（TensorFlow版 agent.losses の置き換え）
# ------------------------------------------------------------
def huber_loss(
    target: torch.Tensor, pred: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    """
    Huber loss（Smooth L1）:
      - 誤差が小さいときは L2（安定）
      - 誤差が大きいときは L1（外れ値に強い）

    Critic の学習を安定させるために DQN などでもよく使われる。

    入力:
      target: [B, 1]
      pred:   [B, 1]
    """
    return F.smooth_l1_loss(pred, target, beta=delta)


def policy_gradient_loss(
    act_onehot: torch.Tensor,
    td_err: torch.Tensor,
    act_prob: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    方策勾配損失（REINFORCE / Actor-Critic の基本形）。

    入力:
      act_onehot: [B, |A|]
        - 実際に選択した行動 a を one-hot で表したもの
      td_err: [B, 1]
        - advantage として TD誤差 δ を使う（元コードと同様の発想）
      act_prob: [B, |A|]
        - Actor が出した方策確率 π(.|s)

    目的:
      maximize: E[ log π(a|s) * advantage ]
    なので最小化損失は:
      loss = -E[ log π(a|s) * advantage ]

    重要（stop_gradient 相当）:
      - advantage（ここでは td_err）は critic から計算されるため、
        actor 更新で critic 側へ勾配が流れるのは基本的に避けたい。
      - PyTorch では td_err.detach() で遮断する。
    """
    # 確率→対数確率（数値安定のため eps を足す）
    logp_all = torch.log(act_prob + eps)  # [B, |A|]

    # one-hot で「実際に選んだ行動」の log π(a|s) を抽出
    logp_taken = (act_onehot * logp_all).sum(dim=-1, keepdim=True)  # [B, 1]

    # td_err を advantage として使用（critic へ勾配が流れないよう detach）
    advantage = td_err.detach()  # [B, 1]

    # 方策勾配損失
    loss = -(logp_taken * advantage).mean()
    return loss


# ============================================================
# エージェント本体（TensorFlow版 ActorCriticAgent を PyTorch化）
# ============================================================
class ActorCriticAgent(nn.Module):
    """
    PyTorch版 ActorCriticAgent。

    互換性のため、TensorFlow版の概念をできるだけ残す:
      - p_holders（placeholder）は PyTorch では不要だが、属性として残しておく
      - model_prds: (value, act_prob) に相当する出力を返す関数を用意
      - losses: (total, vloss, ploss)
      - opts: (critic optimizer, actor optimizer)
      - save_graph/restore_graph: .pt 保存/復元で置き換え
      - update_model/predict_loss: TF版に近い引数で動くようにする

    ただし PyTorch では:
      - forward(state) で value と act_prob を得る
      - loss.backward() → optimizer.step() で更新する
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        action_list: Sequence[Any],
        gamma: float = 0.99,
        critic_learning_rate: float = 1.0e-5,
        actor_learning_rate: float = 1.0e-4,
        use_multistep_return: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # ----------------------------
        # setup variables（元コードの変数を維持）
        # ----------------------------
        self.input_shape = tuple(state_shape)

        self.action_list = list(action_list)
        self.num_value = 1
        self.num_action = len(self.action_list)

        self.gamma = float(gamma)
        self.val_lr = float(critic_learning_rate)
        self.pol_lr = float(actor_learning_rate)

        # 元コードは if 1: の分岐で常に multi-step 側を使うので、デフォルト True
        self.use_multistep_return = bool(use_multistep_return)

        # device（CPU/GPU）
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ----------------------------
        # model predictions（actor/critic ネットワーク）
        # ----------------------------
        critic_model, actor_model = build_model(
            self.input_shape, self.num_value, self.num_action
        )
        self.critic = critic_model
        self.actor = actor_model

        # ----------------------------
        # optimizers（TF版に合わせ RMSProp）
        # ----------------------------
        self.v_optim = torch.optim.RMSprop(self.critic.parameters(), lr=self.val_lr)
        self.p_optim = torch.optim.RMSprop(self.actor.parameters(), lr=self.pol_lr)

        # 互換性のための属性（TF版の雰囲気を残す）
        self.model_prds = [self.critic, self.actor]
        self.losses = None
        self.opts = [self.v_optim, self.p_optim]
        self.p_holders = None  # placeholder は PyTorch では使わない

        # device に移動
        self.to(self.device)

    # --------------------------------------------------------
    # forward: value と act_prob を返す（TF版 _build_agent_network 相当）
    # --------------------------------------------------------
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        入力 state から
          - value: V(s)
          - act_prob: π(.|s)
        を計算して返す。

        入力:
          state: [B, *state_shape]

        出力:
          value: [B, 1]
          act_prob: [B, |A|]
        """
        state = state.to(self.device)
        value = self.critic(state)
        act_prob = self.actor(state)
        return value, act_prob

    # --------------------------------------------------------
    # save / restore（TF版 Saver 置き換え）
    # --------------------------------------------------------
    def save_graph(self, sess: Any, log_dir: str, args: Tuple[int, float, float]):
        """
        TF版互換のため sess 引数は残すが、PyTorchでは使用しない。
        """
        os.makedirs(log_dir, exist_ok=True)
        fname = "model.{0:06d}-{1:3.3f}-{2:3.5f}.pt".format(*args)
        path = os.path.join(log_dir, fname)

        ckpt = {
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "v_optim": self.v_optim.state_dict(),
            "p_optim": self.p_optim.state_dict(),
            "meta": {
                "input_shape": self.input_shape,
                "action_list": self.action_list,
                "gamma": self.gamma,
                "val_lr": self.val_lr,
                "pol_lr": self.pol_lr,
                "use_multistep_return": self.use_multistep_return,
            },
        }
        torch.save(ckpt, path)

    def restore_graph(self, sess: Any, model_path: str):
        """
        TF版互換のため sess 引数は残すが、PyTorchでは使用しない。
        """
        ckpt = torch.load(model_path, map_location=self.device)
        self.critic.load_state_dict(ckpt["critic"])
        self.actor.load_state_dict(ckpt["actor"])
        if "v_optim" in ckpt:
            self.v_optim.load_state_dict(ckpt["v_optim"])
        if "p_optim" in ckpt:
            self.p_optim.load_state_dict(ckpt["p_optim"])

    # --------------------------------------------------------
    # critic prediction（TF版 predict_value 相当）
    # --------------------------------------------------------
    @torch.no_grad()
    def predict_value(
        self, sess: Any, state: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        状態価値 V(s) を推定して numpy で返す（TF版の戻り値互換）。

        入力:
          state: [B, *state_shape] または [*state_shape]（1サンプル）
        出力:
          state_value: [B, 1] の numpy
        """
        self.eval()

        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = state.to(self.device, dtype=torch.float32)

        # 1サンプルなら [1, ...] に整形
        if state_t.dim() == len(self.input_shape):
            state_t = state_t.unsqueeze(0)

        value, _ = self.forward(state_t)
        self.train()
        return value.detach().cpu().numpy()

    # --------------------------------------------------------
    # actor prediction（TF版 predict_policy 相当）
    # --------------------------------------------------------
    @torch.no_grad()
    def predict_policy(
        self, sess: Any, state: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        方策確率 π(.|s) を推定して numpy で返す。
        """
        self.eval()

        if not isinstance(state, torch.Tensor):
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_t = state.to(self.device, dtype=torch.float32)

        if state_t.dim() == len(self.input_shape):
            state_t = state_t.unsqueeze(0)

        _, act_prob = self.forward(state_t)
        self.train()
        return act_prob.detach().cpu().numpy()

    # --------------------------------------------------------
    # get action（TF版 get_action 相当: 確率に従いサンプル）
    # --------------------------------------------------------
    def get_action(self, sess: Any, state: Union[np.ndarray, torch.Tensor]) -> Any:
        """
        1状態入力から行動をサンプリングして返す。
        """
        action_prob = self.predict_policy(sess, state)  # [1, |A|]
        prob = action_prob[0]
        action = np.random.choice(self.action_list, p=prob)
        return action

    # --------------------------------------------------------
    # greedy action（TF版 get_greedy_action 相当: argmax）
    # --------------------------------------------------------
    def get_greedy_action(
        self, sess: Any, state: Union[np.ndarray, torch.Tensor]
    ) -> Any:
        """
        1状態入力から greedy（最大確率）行動を返す。
        """
        action_prob = self.predict_policy(sess, state)  # [1, |A|]
        idx = int(np.argmax(action_prob[0]))
        return self.action_list[idx]

    # --------------------------------------------------------
    # roll_out（元コードの雰囲気を維持：環境依存の set_state あり）
    # --------------------------------------------------------
    def roll_out(self, sess: Any, env: Any, steps: int, state: np.ndarray) -> float:
        """
        1-D input（単一状態）に対して、先読みで追加報酬を推定する補助関数。

        使いどころ:
          - 先読み（n-step lookahead）で advantage 推定の分散を下げたいなど
          - 元コードでは「分散低減のため multiprocess 可能」とコメントがある

        注意:
          - env.step の戻り値形式や env.set_state の有無は環境依存。
          - ここは「元コードの意図を保持」しているため、呼び出し側環境に合わせて調整が必要。
        """
        extra_reward = 0.0
        _state = state.copy()
        done = [False]

        for i_step in range(steps):
            action = self.get_action(sess, _state)
            next_state, reward, done, _ = env.step(action)

            # 割引して加算
            extra_reward += (self.gamma**i_step) * reward[0]
            _state = next_state

            if done[0]:
                break

        # done で終わらなかった場合、末端価値を bootstrap
        if not done[0]:
            v_s = self.predict_value(sess, [_state])  # [1,1]
            extra_reward += (self.gamma**steps) * float(v_s[0][0])

        # additional gamma factor（元コード踏襲）
        extra_reward *= self.gamma

        # 環境状態を戻す（環境側が対応している前提）
        env.set_state(state)

        return float(extra_reward)

    # --------------------------------------------------------
    # ターゲット値（return）の計算: TD(0) or multi-step（元コードの if 1 側）
    # --------------------------------------------------------
    def _compute_target_values(
        self,
        sess: Any,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """
        価値学習の教師信号 target_val（[T,1]）を計算する。

        入力（元コードの前提に合わせる）:
          reward: [T,1] or [T]（各ステップ報酬）
          next_state: [T, *state_shape]（次状態列）
          done: [T,1] or [T]（終端フラグ）

        出力:
          target_val: [T,1]
        """
        # shape 正規化
        reward = np.asarray(reward, dtype=np.float32)
        done = np.asarray(done)

        if reward.ndim == 1:
            reward = reward.reshape(-1, 1)
        if done.ndim == 1:
            done = done.reshape(-1, 1)

        T = reward.shape[0]

        # ----------------------------
        # TD(0)（元コードの if 0 ブロック）
        # ----------------------------
        if not self.use_multistep_return:
            next_st_val = self.predict_value(sess, next_state)  # [T,1]
            target_val = np.where(done, reward, reward + self.gamma * next_st_val)
            return target_val.astype(np.float32)

        # ----------------------------
        # multi-step（元コードの if 1 ブロック）
        # 「報酬列の割引和 + 末端価値の割引」 で G_t を作る
        # ----------------------------
        # terminal state-value の加算（if not done）
        # done[-1][0] が True なら末端価値は 0
        if bool(done[-1][0]):
            v_s_terminal = 0.0
        else:
            v_s = self.predict_value(sess, [next_state[-1]])  # [1,1]
            v_s_terminal = float(v_s[0][0])

        # terminal_val = γ^T * V(s_T)（Tステップ先の価値を割引して足す）
        terminal_val = (self.gamma**T) * v_s_terminal

        # 各 t について G_t を構成
        target_val: List[List[float]] = []
        for i_step in range(T):
            # rwd_seq = [γ^0 r_t, γ^1 r_{t+1}, ..., γ^{k} r_{t+k}, ..., ] + terminal
            rwd_seq = [
                (self.gamma**i) * float(reward[i_step + i][0])
                for i in range(T - i_step)
            ]
            rwd_seq.append(terminal_val)

            g_t = float(np.sum(rwd_seq))
            target_val.append([g_t])

        return np.asarray(target_val, dtype=np.float32)

    # --------------------------------------------------------
    # モデル更新（TF版 update_model 相当）
    # --------------------------------------------------------
    def update_model(
        self,
        sess: Any,
        state: np.ndarray,
        action: Sequence[Any],
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> List[float]:
        """
        1エピソード（またはロールアウト断片）分の列データで Actor と Critic を更新する。

        入力:
          state: [T, *state_shape]
          action: 長さ T の行動列（action_list の要素）
          reward: [T,1] or [T]
          next_state: [T, *state_shape]
          done: [T,1] or [T]

        出力（TF版互換）:
          losses: [loss_total, vloss, ploss]（float）
        """
        self.train()

        # ----------------------------
        # 1) 行動を one-hot に変換
        # ----------------------------
        action_index = np.array(
            [self.action_list.index(a) for a in action], dtype=np.int64
        )  # [T]
        act_onehot = to_onehot(action_index, self.num_action).to(
            self.device
        )  # [T, |A|]

        # ----------------------------
        # 2) ターゲット値（return）と TD 誤差を計算
        #    td_error = target_val - V(s)
        # ----------------------------
        target_val_np = self._compute_target_values(
            sess, reward, next_state, done
        )  # [T,1]

        # torch 化
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        )  # [T, ...]
        target_val_t = torch.tensor(
            target_val_np, dtype=torch.float32, device=self.device
        )  # [T,1]

        # 現在の価値推定 V(s)
        value_pred, act_prob = self.forward(
            state_t
        )  # value_pred: [T,1], act_prob: [T,|A|]

        # TD誤差（advantage の材料）
        td_error_t = target_val_t - value_pred  # [T,1]

        # ----------------------------
        # 3) Critic 更新（Huber loss）
        #    目的: V(s) を target_val に回帰
        # ----------------------------
        vloss = huber_loss(target_val_t, value_pred)

        self.v_optim.zero_grad()
        vloss.backward()
        self.v_optim.step()

        # ----------------------------
        # 4) Actor 更新（policy gradient loss）
        #    目的: log π(a|s) * advantage を最大化
        #    ここでは advantage として td_error を使用（detach して critic へ勾配を流さない）
        # ----------------------------
        # 注意:
        #   Critic 更新後に Actor 更新を行うと、より新しい baseline に基づく advantage を使える。
        #   ただし「同一データで2回 forward する」ことになるので、挙動を TF版に近づけたいなら
        #   forward を取り直すのも手だが、ここではコストを抑えて同じ act_prob を使う。
        ploss = policy_gradient_loss(act_onehot, td_error_t, act_prob)

        self.p_optim.zero_grad()
        ploss.backward()
        self.p_optim.step()

        # ----------------------------
        # 5) 合成 loss（ログ用途）
        # ----------------------------
        loss_total = vloss + ploss
        losses_out = [
            float(loss_total.detach().cpu().item()),
            float(vloss.detach().cpu().item()),
            float(ploss.detach().cpu().item()),
        ]
        return losses_out

    # --------------------------------------------------------
    # loss 推定（TF版 predict_loss 相当：更新はしない）
    # --------------------------------------------------------
    @torch.no_grad()
    def predict_loss(
        self,
        sess: Any,
        state: np.ndarray,
        action: Sequence[Any],
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> List[float]:
        """
        update_model と同じ計算を行うが、optimizer.step を行わずに損失だけ返す。
        """
        self.eval()

        action_index = np.array(
            [self.action_list.index(a) for a in action], dtype=np.int64
        )  # [T]
        act_onehot = to_onehot(action_index, self.num_action).to(
            self.device
        )  # [T, |A|]

        target_val_np = self._compute_target_values(
            sess, reward, next_state, done
        )  # [T,1]

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        target_val_t = torch.tensor(
            target_val_np, dtype=torch.float32, device=self.device
        )

        value_pred, act_prob = self.forward(state_t)  # [T,1], [T,|A|]
        td_error_t = target_val_t - value_pred  # [T,1]

        vloss = huber_loss(target_val_t, value_pred)
        ploss = policy_gradient_loss(act_onehot, td_error_t, act_prob)
        loss_total = vloss + ploss

        self.train()
        return [
            float(loss_total.detach().cpu().item()),
            float(vloss.detach().cpu().item()),
            float(ploss.detach().cpu().item()),
        ]
