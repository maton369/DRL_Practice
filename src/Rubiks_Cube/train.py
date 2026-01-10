import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ここで import する Encoder / ActorDecoder / CriticDecoder は
# 「PyTorch版（nn.Module 実装）」である前提です。
from agent.models import Encoder, ActorDecoder, CriticDecoder


# ============================================================
# 損失・報酬（TensorFlow版 agent.losses の PyTorch 置き換え）
# ============================================================


def tour_distance(coords: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """
    巡回路（tour）の総距離を計算します（TSP系の「コスト」）。

    入力:
      coords: [B, T, coord_dim]
        - B: バッチサイズ
        - T: 都市数（= seq_length）
        - coord_dim: 座標次元（2なら (x,y)）
      tour: [B, T+1]
        - ActorDecoder が「閉路にするために先頭を末尾に追加」して返す想定
        - 各要素は都市インデックス（0..T-1）

    出力:
      dist: [B, 1]
        - 各サンプルの巡回路長（距離の合計）

    補足:
      - tour は離散インデックス列なので、距離はネットワークに対して微分不可能です。
        （ただし baseline の回帰ターゲットとしては問題ありません）
    """
    B, T, D = coords.shape

    # tour の順番に従って座標を並び替える（gather）
    # tour: [B, T+1] -> idx: [B, T+1, D]
    idx = tour.unsqueeze(-1).expand(B, tour.size(1), D)  # [B, T+1, D]
    ordered = coords.gather(dim=1, index=idx)  # [B, T+1, D]

    # 隣接点間の距離を合計
    diffs = ordered[:, 1:, :] - ordered[:, :-1, :]  # [B, T, D]
    seg = torch.sqrt((diffs**2).sum(dim=-1) + 1e-12)  # [B, T]
    dist = seg.sum(dim=-1, keepdim=True)  # [B, 1]
    return dist


def rms_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Critic（baseline）を tour_dist に回帰させる損失。
    TF版の rms_loss は実質 MSE 系として扱うのが自然です。

    target: [B, 1]  （tour_dist）
    pred:   [B, 1]  （baseline）
    """
    return F.mse_loss(pred, target)


def policy_gradient_loss(
    advantage: torch.Tensor, log_prob: torch.Tensor
) -> torch.Tensor:
    """
    Actor の方策勾配損失（REINFORCE / Actor-Critic の基本形）。

    目的:
      - 最大化: E[ log π(a|s) * advantage ]
      - 最小化（損失）としては符号を反転:
        loss = -E[ log π(a|s) * advantage ]

    advantage: [B, 1] または [B]
    log_prob:  [B]
    """
    if advantage.dim() == 2 and advantage.size(-1) == 1:
        advantage = advantage.squeeze(-1)  # [B]
    return -(log_prob * advantage).mean()


# ============================================================
# Actor-Critic Agent（TensorFlow版のクラス構造を保ちつつ PyTorch化）
# ============================================================


class ActorCriticAgent(nn.Module):
    """
    Pointer Network 系の Actor-Critic Agent（PyTorch版）。

    TensorFlow版（placeholder + sess.run）との差分の要点:
      - placeholder:
          -> forward(state) の引数として受け取る
      - sess.run(tensors, feed_dict):
          -> forward で計算 -> loss.backward() -> optimizer.step()
      - tf.stop_gradient:
          -> PyTorch の .detach() で勾配を遮断する

    ネットワーク構成（TF版と同じ思想）:
      1) Encoder（共通）
      2) ActorDecoder: tour をサンプルし log_prob を返す
      3) CriticDecoder: baseline（価値）を推定する

    目的（組合せ最適化/TSP系で典型）:
      - 距離（tour_dist）を最小化したい
      - RLの報酬最大化に合わせて reward = -tour_dist とする
      - Actor は「短い tour を出す」ように更新
      - Critic は tour_dist を当てる baseline を学習して advantage の分散を下げる
    """

    def __init__(
        self,
        n_neurons: int = 128,
        batch_size: int = 4,
        seq_length: int = 10,
        coord_dim: int = 2,
        critic_learning_rate: float = 1.0e-3,
        actor_learning_rate: float = 1.0e-3,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # ----------------------------
        # ハイパーパラメータ（TF版と同名）
        # ----------------------------
        self.n_neurons = int(n_neurons)
        self.batch_size = int(batch_size)
        self.seq_length = int(seq_length)
        self.coord_dim = int(coord_dim)
        self.val_lr = float(critic_learning_rate)
        self.pol_lr = float(actor_learning_rate)

        # 実行デバイス
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ----------------------------
        # ネットワーク（PyTorch版）
        # ----------------------------
        # Encoder は入力が座標列なので input_dim=coord_dim を渡す
        self.encoder = Encoder(
            input_dim=self.coord_dim,
            n_neurons=self.n_neurons,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
        )
        self.actor_decoder = ActorDecoder(
            n_neurons=self.n_neurons,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
        )
        self.critic_decoder = CriticDecoder(
            n_neurons=self.n_neurons,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
        )

        # ----------------------------
        # Optimizer（TF版は RMSProp）
        # ----------------------------
        # 重要: TF版では Encoder が actor/critic の両方に共有されており、
        #       critic 更新でも actor 更新でも Encoder が更新される構造です。
        #
        #       したがって PyTorch でも
        #         - critic optimizer: (encoder + critic_decoder)
        #         - actor  optimizer: (encoder + actor_decoder)
        #       とし、Encoder を両方の更新で学習させます。
        self.v_optim = torch.optim.RMSprop(
            list(self.encoder.parameters()) + list(self.critic_decoder.parameters()),
            lr=self.val_lr,
        )
        self.p_optim = torch.optim.RMSprop(
            list(self.encoder.parameters()) + list(self.actor_decoder.parameters()),
            lr=self.pol_lr,
        )

        # device に載せる
        self.to(self.device)

    # --------------------------------------------------------
    # forward: TF版の「graph で log_prob/tour/state_value を作る」相当
    # --------------------------------------------------------
    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        入力 state（座標列）から、
          - log_prob（Actor がサンプルした tour の対数尤度）
          - tour（サンプルされたインデックス列）
          - baseline（Critic の価値推定）
          - tour_dist（tour の距離）
        を計算して返します。

        入力:
          state: [B, T, coord_dim]

        出力:
          log_prob:  [B]
          tour:      [B, T+1]
          baseline:  [B, 1]
          tour_dist: [B, 1]
        """
        state = state.to(self.device)

        # 1) 共通 Encoder
        enc_outputs, enc_state = self.encoder(state)

        # 2) Actor（tour をサンプルし log_prob を返す）
        log_prob, tour = self.actor_decoder(enc_outputs, enc_state)

        # 3) Critic（baseline）
        baseline = self.critic_decoder(enc_outputs, enc_state)  # [B,1]

        # 4) 距離（コスト）
        tour_dist = tour_distance(state, tour)  # [B,1]

        return log_prob, tour, baseline, tour_dist

    # --------------------------------------------------------
    # loss 構成（TF版 _build_loss 相当）
    # --------------------------------------------------------
    def _build_loss(
        self,
        log_prob: torch.Tensor,
        baseline: torch.Tensor,
        tour_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TF版の _build_loss を PyTorch で再現します。

        TF版:
          _advantage = -1.0 * stop_gradient(tour_dist - baseline)
          vloss = rms_loss(tour_dist, baseline)
          ploss = policy_gradient_loss(_advantage, log_prob)
          loss  = vloss + ploss

        PyTorch版:
          - stop_gradient は .detach() で再現
          - advantage は baseline の影響“値”として使うが、勾配は流さない
        """
        # advantage（Actor 更新用）
        # TF版の式を踏襲し、(tour_dist - baseline) の勾配を遮断して符号反転
        advantage = -1.0 * (tour_dist - baseline).detach()  # [B,1]

        # Critic loss: baseline を tour_dist に回帰
        vloss = rms_loss(tour_dist, baseline)

        # Actor loss: -E[log_prob * advantage]
        ploss = policy_gradient_loss(advantage, log_prob)

        # 合成（重みは TF版と同じ 1.0, 1.0）
        loss = vloss + ploss
        return loss, vloss, ploss

    # --------------------------------------------------------
    # save/restore（TF版 saver 相当）
    # --------------------------------------------------------
    def save_graph(self, sess: Any, log_dir: str, args: Tuple[int, float, float]):
        """
        TF版互換のために sess 引数は残していますが、PyTorchでは使用しません。

        args:
          (step, metric1, metric2) のようなタプルを想定し、ファイル名に埋め込みます。
        """
        os.makedirs(log_dir, exist_ok=True)
        fname = "model.{0:06d}-{1:3.3f}-{2:3.5f}.pt".format(*args)
        path = os.path.join(log_dir, fname)

        # 重要: optimizer も含めて保存しておくと学習再開が楽です
        ckpt = {
            "encoder": self.encoder.state_dict(),
            "actor_decoder": self.actor_decoder.state_dict(),
            "critic_decoder": self.critic_decoder.state_dict(),
            "v_optim": self.v_optim.state_dict(),
            "p_optim": self.p_optim.state_dict(),
            "meta": {
                "n_neurons": self.n_neurons,
                "batch_size": self.batch_size,
                "seq_length": self.seq_length,
                "coord_dim": self.coord_dim,
                "val_lr": self.val_lr,
                "pol_lr": self.pol_lr,
            },
        }
        torch.save(ckpt, path)

    def restore_graph(self, sess: Any, model_path: str):
        """
        TF版互換のために sess 引数は残していますが、PyTorchでは使用しません。
        """
        ckpt = torch.load(model_path, map_location=self.device)

        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor_decoder.load_state_dict(ckpt["actor_decoder"])
        self.critic_decoder.load_state_dict(ckpt["critic_decoder"])

        # optimizer の復元（任意）
        if "v_optim" in ckpt:
            self.v_optim.load_state_dict(ckpt["v_optim"])
        if "p_optim" in ckpt:
            self.p_optim.load_state_dict(ckpt["p_optim"])

    # --------------------------------------------------------
    # update_model（TF版: critic更新→actor更新 を sess.run で実行）
    # --------------------------------------------------------
    def update_model(
        self,
        sess: Any,
        state: Union[torch.Tensor, "numpy.ndarray"],
    ) -> Tuple[List[float], List[float], List[Any]]:
        """
        TF版 update_model(sess, state) 相当。
        1バッチ入力 state に対して
          1) critic（baseline回帰）更新
          2) actor（方策勾配）更新
        を順に実行します。

        戻り値の形式は TF版に合わせて:
          losses:  [loss_total, vloss, ploss]
          rewards: [reward, tour_dist]
          model_prds: [log_prob, tour, state_value(baseline)]
        とします（ただし PyTorch では Tensor ではなく値/配列に変換して返します）。

        注意:
          - TF版は「critic更新」でも「actor更新」でも encoder が更新されます。
            PyTorch版も optimizer のパラメータ構成により同様です。
          - 2回 backward するため、各更新で forward を取り直して計算グラフを分けます。
            （retain_graph を多用しない方が事故が少ない）
        """
        # 入力を torch.Tensor に統一
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        # ========= 1) critic update =========
        self.train()
        log_prob, tour, baseline, tour_dist = self.forward(state)
        loss_total, vloss, ploss = self._build_loss(log_prob, baseline, tour_dist)

        # critic は vloss のみで更新（baselineを tour_dist に回帰）
        self.v_optim.zero_grad()
        vloss.backward()
        self.v_optim.step()

        # ========= 2) actor update =========
        # critic 更新後のパラメータで forward を取り直す（TF版の逐次更新に近い挙動）
        log_prob, tour, baseline, tour_dist = self.forward(state)
        loss_total, vloss, ploss = self._build_loss(log_prob, baseline, tour_dist)

        # actor は ploss で更新（advantage は detach 済みなので critic へ勾配は流れない）
        self.p_optim.zero_grad()
        ploss.backward()
        self.p_optim.step()

        # ========= rewards / outputs =========
        # 距離最小化なので reward = -tour_dist
        reward = -1.0 * tour_dist

        # TF版互換の形にして返す（ログ用途を想定し、基本は Python float / CPU に落とす）
        losses_out = [
            float(loss_total.detach().cpu().item()),
            float(vloss.detach().cpu().item()),
            float(ploss.detach().cpu().item()),
        ]
        rewards_out = [
            float(reward.detach().cpu().mean().item()),
            float(tour_dist.detach().cpu().mean().item()),
        ]
        model_prds_out = [
            log_prob.detach().cpu().numpy(),  # [B]
            tour.detach().cpu().numpy(),  # [B, T+1]
            baseline.detach().cpu().numpy(),  # [B, 1]
        ]
        return losses_out, rewards_out, model_prds_out

    # --------------------------------------------------------
    # predict_loss（TF版: sess.run で損失推定）
    # --------------------------------------------------------
    @torch.no_grad()
    def predict_loss(
        self,
        sess: Any,
        state: Union[torch.Tensor, "numpy.ndarray"],
    ) -> Tuple[List[float], List[float], List[Any]]:
        """
        TF版 predict_loss(sess, state) 相当。
        学習（optimizer.step）は行わず、損失・報酬・出力だけを計算して返します。
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        self.eval()
        log_prob, tour, baseline, tour_dist = self.forward(state)
        loss_total, vloss, ploss = self._build_loss(log_prob, baseline, tour_dist)

        reward = -1.0 * tour_dist

        losses_out = [
            float(loss_total.detach().cpu().item()),
            float(vloss.detach().cpu().item()),
            float(ploss.detach().cpu().item()),
        ]
        rewards_out = [
            float(reward.detach().cpu().mean().item()),
            float(tour_dist.detach().cpu().mean().item()),
        ]
        model_prds_out = [
            log_prob.detach().cpu().numpy(),
            tour.detach().cpu().numpy(),
            baseline.detach().cpu().numpy(),
        ]
        self.train()
        return losses_out, rewards_out, model_prds_out
