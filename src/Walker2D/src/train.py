"""
overview:
    pybullet-gym の Walker2DPyBulletEnv-v0 環境で Walker（2足歩行ロボット）を歩かせるように学習を行う。

    本コードは「方策勾配（REINFORCE 系） + ベースライン（状態価値関数）」の形になっている。
    具体的には、
      - PolicyEstimator: 確率的方策 π(a|s) を表すネットワーク（Actor 相当）
      - ValueEstimator : 状態価値 V(s) を表すネットワーク（Critic/Baseline 相当）
    を用意し、各エピソードで得た割引報酬和 G_t を教師信号として学習する。

    更新則の直感:
      - Policy は advantage = (G_t - V(s_t)) が正なら「その行動を選びやすく」し、負なら「選びにくく」する
      - Value は G_t を回帰することで V(s) を改善し、Policy の勾配推定の分散を下げる

args:
    各種パラメータ設定値は本コード中に明記される。
    学習が進まなくなる可能性があるので変更は非推奨。

    - result_dir:
        結果を出力するディレクトリの path
    - num_episodes:
        学習の繰り返しエピソード数 (default: 500000)
    - max_episode_steps:
        エピソードの最大ステップ数 (default: 200)
    - gamma:
        割引率 (default: 0.99)
    - model_save_interval:
        何エピソードごとに policy_estimator の重みを出力するか (default: 10000)

output:
    result_dir 配下に以下のファイルが出力される
      - episode_xxx.h5:
          xxx エピソード目まで学習した policy_estimator の重み
      - history.csv:
          エピソードごとのメトリック
            - score: 1エピソードで得られた報酬和
            - steps/episode: 1エピソードのステップ数（倒れずに進めた長さ）
            - loss: policy_estimator の学習損失
      - history.png:
          上記メトリックの推移（移動平均）を可視化した学習曲線

usage:
    python train.py

注意:
    - このコードは TensorFlow 1.x の Session を使う構成である（tf.Session）。
    - pybulletgym の環境登録のために `import pybulletgym.envs` を行っている。
"""

import collections
import csv
from datetime import datetime
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pybulletgym.envs  # 環境ID登録のために必要

from agent.policy_estimator import PolicyEstimator
from agent.value_estimator import ValueEstimator

# 学習曲線の見た目（グリッド背景）を統一する
plt.style.use("seaborn-darkgrid")


def now_str(str_format="%Y%m%d%H%M"):
    """
    現在時刻をフォーマットして返す関数。
    result_dir のユニークな名前付けに使う。
    """
    return datetime.now().strftime(str_format)


def visualize_history(csv_path, png_path, metrics, window_size=0.1):
    """
    学習ログ history.csv を読み込み、各メトリックの移動平均を描画して history.png に保存する。

    window_size:
      - 0 < window_size < 1 の場合: 全データ長の割合として窓幅を決める（例: 0.1 なら全体の10%）
      - window_size >= 1 の場合: その値を窓幅（整数）として使う

    可視化の意図:
      - 強化学習はノイズが大きく、逐次値をそのまま見ると傾向が掴みにくい
      - 移動平均で大局的な改善傾向を確認しやすくする
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
        # legend はタイトルで十分なので消す
        ax.get_legend().remove()

    ax.set_xlabel("episode", fontsize=12)
    fig.suptitle("Walker2DPyBulletEnv-v0")
    fig.savefig(png_path)
    plt.close(fig)


def train():
    """
    学習の入口関数。

    流れ:
      1) 環境生成
      2) 観測次元・行動次元を取得
      3) PolicyEstimator / ValueEstimator を生成
      4) tf.Session を開き、変数初期化して学習本体 _train を呼ぶ
    """
    # Walker2D の PyBullet 環境を生成
    env = gym.make("Walker2DPyBulletEnv-v0")

    # 観測（状態）次元と行動次元
    # env.env を参照しているのは、gym のラッパ階層の下の本体を使いたい意図と思われる
    dim_state = env.env.observation_space.shape[0]
    dim_action = env.env.action_space.shape[0]

    # Actor（方策）と Baseline/Critic（価値）のネットワークを用意
    policy_estimator = PolicyEstimator(dim_state=dim_state, dim_action=dim_action)
    value_estimator = ValueEstimator(dim_state=dim_state)

    # TF1 セッションで学習を実行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _train(sess, env, policy_estimator, value_estimator)


def _train(sess, env, policy_estimator, value_estimator):
    """
    学習本体（エピソードループ）。

    アルゴリズム的にやっていること:
      - 各エピソードで trajectory（state, action, reward の列）を収集
      - 各時刻 t について割引報酬和 G_t を計算
      - baseline = V(s_t) を Critic から得る
      - advantage = G_t - V(s_t) を作り、Policy を更新
      - Value は G_t に回帰するよう更新

    本実装の特徴:
      - target=G_t は「エピソード末端までのモンテカルロ推定」なので分散が大きい傾向がある
      - baseline を入れることで分散低減を狙っている（REINFORCE with baseline）
    """
    # 結果を出力するディレクトリ（タイムスタンプ付き）
    result_dir = "./result/walker2d/{now_str}".format(
        now_str=now_str(str_format="%Y%m%d_%H%M%S")
    )
    os.makedirs(result_dir, exist_ok=True)
    print("result_dir_{}".format(result_dir))

    # -------------------------------
    # 学習ハイパーパラメータ
    # -------------------------------
    num_episodes = 500000  # 学習エピソード数（非常に大きいので長時間実行前提）
    max_episode_steps = 200  # 1エピソードの最大ステップ数
    gamma = 0.99  # 割引率 γ
    model_save_interval = 10000  # 方策ネットワークの保存間隔（エピソード単位）

    # -------------------------------
    # ログファイルのパス
    # -------------------------------
    csv_path = os.path.join(result_dir, "history.csv")
    png_path = os.path.join(result_dir, "history.png")

    # 記録するメトリック
    metrics = ["steps/episode", "score", "loss"]
    header = ["episode"] + metrics

    # history.csv を作成（ヘッダ行を書き込む）
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # 1ステップのデータ構造（観測・行動・報酬）
    Step = collections.namedtuple("Step", ["state", "action", "reward"])

    # 直近100エピソードの統計（平均表示用）
    last_100_score = np.zeros(100)
    last_100_steps = np.zeros(100)

    print("start_episodes...")

    # ============================================================
    # エピソードループ
    # ============================================================
    for i_episode in range(1, num_episodes + 1):
        # エピソード開始: 初期状態を取得
        state = env.reset()

        # このエピソードの軌跡を貯めるリスト
        episode = []

        # 1エピソードの合計報酬（score）
        score = 0.0

        # 1エピソード内で進めたステップ数
        steps = 0

        # ------------------------------------------------------------
        # 1エピソードのステップループ（trajectory 収集）
        # ------------------------------------------------------------
        while True:
            steps += 1

            # 確率的方策 π(a|s) に従って行動をサンプリング（Actor）
            action = policy_estimator.predict(sess, state)

            # 環境を1ステップ進める
            state_new, r, done, _ = env.step(action)

            # 報酬を蓄積（ここではクリップせず生の報酬）
            score += r

            # 1ステップ分の経験を保存
            episode.append(Step(state=state, action=action, reward=r))

            # 状態更新（次のステップへ）
            state = state_new

            # 倒れる or 最大ステップ数でエピソード終了
            if steps > max_episode_steps or done:
                break

        # ============================================================
        # 1エピソード終了後: 各時刻 t の target と advantage を作る
        # ============================================================
        targets = []
        states = []
        actions = []
        advantages = []

        # t=0..T-1 それぞれについて G_t を計算する（モンテカルロ）
        # 注意:
        #   ここは O(T^2) になっている
        #   （各 t について episode[t:] を走査して sum を取るため）
        #   max_episode_steps=200 ならまだ許容だが、長いと重くなる
        for t, step in enumerate(episode):
            # --------------------------------------------------------
            # 割引報酬和（モンテカルロリターン）:
            #   G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
            # --------------------------------------------------------
            target = sum((gamma**i) * t2.reward for i, t2 in enumerate(episode[t:]))

            # --------------------------------------------------------
            # baseline（Critic）:
            #   V(s_t) を推定
            # --------------------------------------------------------
            baseline_value = value_estimator.predict(sess, step.state)[0][0]

            # --------------------------------------------------------
            # advantage:
            #   A_t = G_t - V(s_t)
            #   → その行動が「期待より良い/悪い」度合い
            # --------------------------------------------------------
            advantage = target - baseline_value

            # 学習用の配列に格納
            targets.append([target])
            advantages.append([advantage])
            states.append(step.state)
            actions.append(step.action)

        # ============================================================
        # Policy / Value の更新
        # ============================================================
        # PolicyEstimator:
        #   advantage を重みとして logπ を増やす（方策勾配）
        loss = policy_estimator.update(sess, states, actions, advantages)

        # ValueEstimator:
        #   V(s) を G_t に回帰（MSE など）
        _ = value_estimator.update(sess, states, targets)

        # ============================================================
        # ログ保存
        # ============================================================
        last_100_steps[i_episode % 100] = steps
        last_100_score[i_episode % 100] = score

        episode_steps_score_loss = [i_episode, steps, score, loss]

        # history.csv へ追記
        with open(csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(episode_steps_score_loss)

        # ============================================================
        # 定期ログ（100エピソードごと）+ 可視化更新
        # ============================================================
        if i_episode % 100 == 0:
            # 直近100エピソード平均（最初の100未満は i_episode で割る）
            denom = i_episode if i_episode < 100 else 100
            last_100_steps_avg = float(np.sum(last_100_steps) / denom)
            last_100_score_avg = float(np.sum(last_100_score) / denom)

            print(
                "episode_{} last100_steps_avg_{} last100_score_avg_{}".format(
                    i_episode, last_100_steps_avg, last_100_score_avg
                )
            )

            # 学習曲線（移動平均）を更新して保存
            visualize_history(csv_path, png_path, metrics)

        # ============================================================
        # 定期保存（Policy ネットワーク）
        # ============================================================
        if i_episode % model_save_interval == 0:
            model_path = os.path.join(result_dir, "episode_{}.h5".format(i_episode))
            policy_estimator.model.save(model_path)


if __name__ == "__main__":
    train()
