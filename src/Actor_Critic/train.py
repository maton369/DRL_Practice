"""
overview:
    Actor-Critic 法（方策勾配 + 価値関数近似）により Pendulum の学習を行う（PyTorch版）

    もとの TensorFlow 版の構造（要点）を維持しつつ、学習基盤を PyTorch に置き換える。
    このスクリプトは「固定長バッチ（batch_size ステップ）」を 1 回分の学習単位として扱う。

    学習の大枠（Actor-Critic / REINFORCE with baseline）:
      1) Actor（方策）:
           状態 s から 行動分布 π(a|s) を出し、そこから行動 a をサンプルする（確率的方策）
      2) Critic（価値）:
           状態価値 V(s) を推定する（スカラー）
      3) バッチ（固定長 trajectory 断片）を集めた後、TDターゲット target と advantage を計算する
           - Critic は target へ回帰（MSE）して V を改善
           - Actor は advantage で重み付けした logπ を最大化（損失最小化）して方策を改善

    multi_step_td の意味:
      - True:
          バッチ末尾の V(s_last) をブートストラップに使い、後ろ向きに
              target_t = r_t + γ target_{t+1}
          を計算する（固定長 n-step return のイメージ）
      - False:
          1-step TD を使う（ただし元コード同様、後ろ向きに target を作り、value_last を更新していく）

output:
    result_dir に以下が出力される
      - options.csv: 学習パラメータ設定
      - history.csv: バッチごとの score（報酬合計）, Actor loss, Critic loss
      - history.png: history.csv の各メトリックの移動平均の可視化
      - batch_xxx.pth: interval ごとに保存する Actor の重み（PyTorch state_dict）

usage:
    python3 train.py

注意:
    - Pendulum は本来「連続行動」環境であるが、本コードは actions_list=(-1,1) のように
      離散化した 2 値行動を環境へ与える（簡易な離散化設定）
    - Gym のバージョン差異により reset()/step() の戻り値が異なる場合があるため、
      できるだけ互換的に扱う（obs, info を返すケースなど）
"""

import csv
from datetime import datetime
import os
import collections

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 学習パラメータ
# ============================================================
train_config = {
    "num_batches": 40000,   # バッチ更新回数（学習反復回数）
    "gamma": 0.99,          # 割引率 γ
    "interval": 10000,      # Actor 重み保存の間隔（バッチ単位）
    "batch_size": 50,       # 1バッチで収集するステップ数（固定長）
    "multi_step_td": True,  # True: 複数ステップTD（後ろ向きのn-step） / False: 1-step TD
}


# ============================================================
# 便利関数
# ============================================================
def now_str(str_format="%Y%m%d%H%M"):
    """現在時刻を文字列にして返す（保存ディレクトリ名などに使う）"""
    return datetime.now().strftime(str_format)


def write_options(csv_path, train_config):
    """
    バッチ学習のパラメータ設定を options.csv として出力する関数。

    目的:
      - 実験結果の再現性を高めるため、学習時に使った設定を必ず記録する
      - 実験比較（ハイパーパラメータ比較）のときに参照しやすくする
    """
    with open(csv_path, "w") as f:
        fieldnames = ["Option Name", "Option Value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in train_config.items()]
        writer.writerows(data)


def visualize_history(csv_path, png_path, metrics, window_size=0.1):
    """
    history.csv の推移を可視化して history.png を保存する関数。

    window_size:
      - 0 < window_size < 1 の場合: データ全体の割合として窓幅を決める（例: 0.1 なら全体の10%）
      - window_size >= 1 の場合: その値を移動平均の窓幅（整数）として使う

    可視化の意図:
      - 強化学習はノイズが大きく、単純な折れ線だと傾向が見えにくい
      - 移動平均（rolling mean）で「学習が改善しているか」を視認しやすくする
    """
    df = pd.read_csv(csv_path)

    # 窓幅を整数に変換
    if window_size < 1:
        window_size = max(int(len(df) * window_size), 1)
    else:
        window_size = int(window_size)

    rolled = df.rolling(window=window_size, center=True).mean()

    fig = plt.figure(figsize=(12, 3 * len(metrics)))
    axes = fig.subplots(nrows=len(metrics), sharex=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        rolled[metric].plot(ax=ax, title=metric, grid=True, legend=True)
        ax.get_legend().remove()

    ax.set_xlabel("batch", fontsize=12)
    fig.suptitle("window size is {0:d}.".format(window_size))
    fig.savefig(png_path)
    plt.close(fig)


def _safe_reset(env):
    """
    Gym の reset() の戻り値差異を吸収するヘルパ。

    - 旧 gym: obs
    - 新しいAPI: (obs, info)

    どちらでも obs を返すようにする。
    """
    out = env.reset()
    if isinstance(out, tuple) and len(out) >= 1:
        return out[0]
    return out


def _safe_step(env, action):
    """
    Gym の step() の戻り値差異を吸収するヘルパ。

    - 旧 gym: (obs, reward, done, info)
    - 新しいAPI: (obs, reward, terminated, truncated, info)

    どちらでも (obs, reward, done, info) を返すようにする。
    """
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = out
    return obs, reward, bool(done), info


def idx2onehot(idx, num_classes):
    """
    行動インデックスを one-hot ベクトルへ変換する。

    例:
      idx=2, num_classes=4 -> [0,0,1,0]

    Actor の更新（logπ(a|s) を抽出）で one-hot を使うために利用する。
    """
    onehot = np.zeros(num_classes, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot


# ============================================================
# Actor（方策ネットワーク）: PyTorch 実装
# ============================================================
class Actor(nn.Module):
    """
    離散行動の確率分布 π(a|s) を出力し、方策勾配で更新する Actor。

    ネットワーク:
      - MLP（tanh）で logits を出す
      - 推論時に softmax で確率に変換し、Categorical でサンプル

    学習（損失）:
      目的:
        E[ log π(a|s) * advantage ] を最大化
      最小化形:
        L = - E[ log π(a|s) * advantage ]
    """

    def __init__(self, num_states, actions_list, learning_rate=1e-3, device=None):
        super().__init__()
        self.num_states = int(num_states)
        self.actions_list = list(actions_list)
        self.num_actions = len(self.actions_list)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 元コード（TF版）を踏襲した層サイズ
        num_dense_1 = self.num_states * 10
        num_dense_3 = self.num_actions * 10
        num_dense_2 = int(np.sqrt(num_dense_1 * num_dense_3))

        # MLP（tanh）
        self.fc1 = nn.Linear(self.num_states, num_dense_1)
        self.fc2 = nn.Linear(num_dense_1, num_dense_2)
        self.fc3 = nn.Linear(num_dense_2, num_dense_3)
        self.out = nn.Linear(num_dense_3, self.num_actions)

        # 元コードは RMSProp を使っていたが、Actor-Critic では RMSProp/Adam どちらも使われる。
        # ここでは TF版Actor と同じ意図を尊重して RMSprop を採用する。
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)

        self.to(self.device)

    def forward(self, state):
        """
        確率 π(a|s) を返す forward。

        入力:
          state: torch.Tensor
            shape=(B, num_states) または (num_states,)

        出力:
          probs: torch.Tensor
            shape=(B, num_actions)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        logits = self.out(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    def _log_probs(self, state):
        """
        学習用に log π(a|s) を安定に計算する（log_softmax を使う）。

        出力:
          log_probs: torch.Tensor, shape=(B, num_actions)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        logits = self.out(x)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def predict(self, state):
        """
        方策 π(a|s) に従って行動をサンプリングし、actions_list の値として返す。

        入力:
          state: array-like, shape=(num_states,)

        出力:
          action_value: actions_list の要素（環境へ渡す実際の行動値）
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.eval()
        with torch.no_grad():
            probs = self.forward(state_t).squeeze(0)  # (num_actions,)
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = int(dist.sample().item())
        self.train()

        return self.actions_list[action_idx]

    def update(self, states, act_onehots, advantages):
        """
        Actor を 1 バッチ分更新する。

        入力:
          states:
            shape=(B, num_states) の list/np.ndarray
          act_onehots:
            shape=(B, num_actions) の list/np.ndarray
          advantages:
            shape=(B, 1) または (B,) の list/np.ndarray

        損失:
          L = - mean( logπ(a|s) * advantage )
        """
        states_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        act_onehots_t = torch.tensor(np.asarray(act_onehots), dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(np.asarray(advantages), dtype=torch.float32, device=self.device)

        # advantage を (B,) に揃える
        if adv_t.dim() == 2 and adv_t.size(1) == 1:
            adv_t = adv_t.squeeze(1)

        # logπ(a|s) を計算
        log_probs = self._log_probs(states_t)  # (B, num_actions)

        # 選択行動の log確率だけ取り出す（one-hot との内積）
        selected_logp = torch.sum(log_probs * act_onehots_t, dim=1)  # (B,)

        # 方策勾配の最小化形（負号付き）
        loss = torch.mean(-(selected_logp * adv_t))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


# ============================================================
# Critic（状態価値ネットワーク）: PyTorch 実装
# ============================================================
class Critic(nn.Module):
    """
    状態価値 V(s) を近似し、target への回帰（MSE）で更新する Critic。

    学習の基本:
      - target（教師信号）を外部で作る
      - Critic は V(s) を target に近づけるように回帰する

    典型的な target:
      - 1-step TD: target = r + γ(1-done) V(s')
      - n-step TD: target = r_t + γ r_{t+1} + ... + γ^n V(s_{t+n})
    """

    def __init__(self, num_states, learning_rate=1e-3, device=None):
        super().__init__()
        self.num_states = int(num_states)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 元コード（TF版）を踏襲した層サイズ
        num_dense_1 = self.num_states * 10
        num_dense_3 = 5
        num_dense_2 = int(np.sqrt(num_dense_1 * num_dense_3))

        self.fc1 = nn.Linear(self.num_states, num_dense_1)
        self.fc2 = nn.Linear(num_dense_1, num_dense_2)
        self.fc3 = nn.Linear(num_dense_2, num_dense_3)
        self.v_out = nn.Linear(num_dense_3, 1)

        # 元コードの Critic は Adam
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.to(self.device)

    def forward(self, state):
        """
        入力:
          state: torch.Tensor
            shape=(B, num_states) または (num_states,)

        出力:
          v: torch.Tensor
            shape=(B, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        v = self.v_out(x)
        return v

    def predict(self, state):
        """
        推論用：V(s) を numpy で返す。

        入力:
          state: array-like, shape=(num_states,)
        出力:
          v: float
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        self.eval()
        with torch.no_grad():
            v = self.forward(state_t).squeeze(0).squeeze(0).item()
        self.train()

        return float(v)

    def update(self, states, targets):
        """
        Critic を 1 バッチ分更新する（target への回帰）。

        入力:
          states:
            shape=(B, num_states)
          targets:
            shape=(B, 1) または (B,)
        """
        states_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        targets_t = torch.tensor(np.asarray(targets), dtype=torch.float32, device=self.device)

        if targets_t.dim() == 1:
            targets_t = targets_t.unsqueeze(1)

        v_pred = self.forward(states_t)  # (B,1)
        loss = self.criterion(v_pred, targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


# ============================================================
# 学習実行関数（train）
# ============================================================
def train(train_config=train_config, actions_list=(-1, 1)):
    """
    学習の入口関数。

    ポイント:
      - 本コードは「エピソード終了」まで回すのではなく、
        各バッチで batch_size ステップ分だけ環境を回して更新する。
      - Pendulum は明示的な done がほぼ発生しない（あるいはバージョンにより扱いが異なる）ため、
        固定ステップで切る設計は理解しやすい。
    """
    # 環境生成（元コード踏襲）
    env = gym.make("Pendulum-v0")

    # 状態次元と行動数
    # Gym の observation_space は shape=(num_states,) を想定
    num_states = env.env.observation_space.shape[0]
    num_actions = len(actions_list)
    print("NUM_STATE_{}".format(num_states))
    print("NUM_ACTIONS_{}".format(num_actions))

    # Actor/Critic を作成
    actor = Actor(num_states=num_states, actions_list=actions_list)
    critic = Critic(num_states=num_states)

    # バッチ学習を実行
    _train(env, actor, critic, train_config, actions_list)

    env.close()


def _train(env, actor, critic, train_config, actions_list):
    """
    バッチ TD 学習の本体。

    学習の流れ:
      for each batch:
        1) batch_size ステップ分サンプル（state, action, reward）
        2) バッチ末尾の V(s_last) を取得（ブートストラップ基点）
        3) 後ろ向きに target と advantage を計算
        4) Actor を advantage 付き logπ で更新
        5) Critic を target への回帰で更新
        6) ログ保存と可視化、定期保存
    """
    # 結果出力先ディレクトリ
    result_dir = "./result/{now_str}".format(now_str=now_str(str_format="%Y%m%d_%H%M%S"))
    os.makedirs(result_dir, exist_ok=True)
    print("result_dir_{}".format(result_dir))

    # ログ・可視化ファイル
    csv_path = os.path.join(result_dir, "history.csv")
    png_path = os.path.join(result_dir, "history.png")
    opt_path = os.path.join(result_dir, "options.csv")

    # 記録するメトリック（元コード踏襲）
    metrics = ["score", "loss", "loss_v"]
    header = ["batch"] + metrics

    # CSV 初期化
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # options.csv 出力
    write_options(opt_path, train_config)

    # ハイパーパラメータ読込
    num_batches = int(train_config["num_batches"])
    batch_size = int(train_config["batch_size"])
    gamma = float(train_config["gamma"])
    interval = int(train_config["interval"])
    multi_step_td = bool(train_config["multi_step_td"])

    # 1ステップ分の記録（state, act_onehot, reward）
    Step = collections.namedtuple("Step", ["state", "act_onehot", "reward"])

    # 直近100バッチのスコア平均を記録（学習の進捗確認用）
    last_100_score = np.zeros(100, dtype=np.float32)

    print("start_batches...")
    for i_batch in range(1, num_batches + 1):
        # バッチ開始時に環境を reset
        state = _safe_reset(env)

        batch = []
        score = 0.0
        steps = 0

        # ============================================================
        # 1) batch_size ステップ分サンプル
        # ============================================================
        while True:
            steps += 1

            # Actor による行動サンプリング（確率的方策）
            action = actor.predict(state)

            # one-hot は「選択した行動」を表現するために必要
            # actions_list のどれを選んだかを index に直して one-hot 化
            act_idx = actions_list.index(action)
            act_onehot = idx2onehot(act_idx, len(actions_list))

            # 環境を1ステップ進める
            # Pendulum の action_space は shape=(1,) を期待することが多いので [action] にする
            state_new, reward, done, info = _safe_step(env, [action])

            # reward clipping（元コード踏襲）
            # Pendulum の元の報酬は連続値（主に負）になりがちなので、ここでは -1 / +1 に丸める
            if reward < -1:
                c_reward = -1.0
            else:
                c_reward = 1.0

            score += c_reward

            # バッチ（軌跡断片）へ記録
            batch.append(Step(state=state, act_onehot=act_onehot, reward=c_reward))

            # 次状態へ
            state = state_new

            # バッチを固定長で終了
            if steps >= batch_size:
                break

        # ============================================================
        # 2) バッチ末尾の状態価値 V(s_last) を取得（ブートストラップ）
        # ============================================================
        value_last = critic.predict(state)

        # ============================================================
        # 3) 後ろ向きに target と advantage を計算
        # ============================================================
        targets = []
        states = []
        act_onehots = []
        advantages = []

        # multi_step_td=True の場合:
        #   target を「将来側の target」から再帰的に作る
        #   target_{t} = r_t + γ target_{t+1}
        #
        # multi_step_td=False の場合:
        #   1-step TD: target = r + γ V(s')
        #   ただし元コード同様、後ろ向きループの中で value_last を更新していく
        target = float(value_last)

        # reversed で t=T-1,...,0 の順に処理
        for t, step in reversed(list(enumerate(batch))):
            current_value = critic.predict(step.state)

            if multi_step_td:
                # 複数ステップ TD（n-step return の後ろ向き計算）
                # target を末尾から畳み込むことで、固定長バッチ内の累積割引和を作る
                target = step.reward + gamma * target
            else:
                # 1-step TD（元コードの挙動踏襲）
                # value_last を「次状態の価値」とみなし、target を作る
                target = step.reward + gamma * value_last
                # 次の反復では value_last を current_value に更新する（後ろ向き処理の都合）
                value_last = current_value

            # advantage（TD誤差の形）
            advantage = target - current_value

            # 学習用に配列へ格納
            targets.append([target])           # shape=(1,)
            advantages.append([advantage])     # shape=(1,)
            states.append(step.state)
            act_onehots.append(step.act_onehot)

        # ============================================================
        # 4) Actor と Critic を更新
        # ============================================================
        # Actor: -E[ logπ(a|s) * advantage ] を最小化
        loss = actor.update(states, act_onehots, advantages)

        # Critic: MSE(V(s), target) を最小化
        loss_v = critic.update(states, targets)

        # ============================================================
        # 5) ログ保存（history.csv）
        # ============================================================
        last_100_score[i_batch % 100] = score
        last_100_score_avg = float(np.sum(last_100_score) / min(i_batch, 100))

        batch_score_loss = [i_batch, score, loss, loss_v]
        with open(csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(batch_score_loss)

        # ============================================================
        # 6) 定期ログ・可視化
        # ============================================================
        if i_batch % 100 == 0:
            print(
                "batch_{} score_{} avg_loss_{} avg_td2_{} last100_score_{}"
                .format(i_batch, score, loss, loss_v, last_100_score_avg)
            )
            visualize_history(csv_path, png_path, metrics)

        # ============================================================
        # 7) 定期保存（Actor の重み）
        # ============================================================
        # TF版は .h5 保存だったが、PyTorch では通常 state_dict を保存する
        if i_batch % interval == 0:
            model_path = os.path.join(result_dir, "batch_{}.pth".format(i_batch))
            torch.save(actor.state_dict(), model_path)


if __name__ == "__main__":
    train()