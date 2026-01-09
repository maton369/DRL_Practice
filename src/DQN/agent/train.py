"""
overview:
    Gymnasium の Pendulum 環境を用いて Double DQN の学習を行う。

    変更点（gym -> gymnasium）:
      - import gymnasium as gym に変更
      - env.reset() は (obs, info) を返すため、obs を受け取る
      - env.step(action) は (obs, reward, terminated, truncated, info) を返すため、
        done = terminated or truncated として終端判定を統一する

    注意:
      - Pendulum-v0 は古いIDで、Gymnasiumでは通常 'Pendulum-v1' を使う。
      - 本コードは「離散化した2値行動（-1, +1）」を Pendulum の連続行動へ渡す。
        Pendulum の action_space は shape=(1,) の Box なので、action は np.array([value], dtype=np.float32)
        のように渡すのが安全である。

args:
    各種パラメータの設定値は、本コード中に明記される
    - result_dir:
        結果を出力するディレクトリのpath
    - max_episode:
        学習の繰り返しエピソード数(default: 300)
    - max_step:
        1エピソード内の最大ステップ数(default: 200)
    - gamma:
        割引率(default: 0.99)

output:
    result_dirで指定したpathに以下のファイルが出力される
    - episode_xxx.h5:
        xxxエピソードまで学習したDouble_DQNネットワークの重み
    - history.csv:
        エピソードごとの以下の3つのメトリックを記録するcsv
        - loss: DoubleDQNモデルを更新する際のlossの平均値
        - td_error: TD誤差の平均値
        - reward_avg: １ステップあたりの平均報酬（ここではクリップ後報酬）

usage：
    python3 train.py
"""

import os
import random

import gymnasium as gym
import numpy as np

from agent.model import Qnetwork
from agent.policy import EpsilonGreedyPolicy
from util import now_str, RecordHistory


def train():
    # ============================================================
    # setup（ハイパーパラメータ / 実験設定）
    # ============================================================
    max_episode = 300
    max_step = 200
    n_warmup_steps = 10000
    interval = 1

    actions_list = [-1, 1]  # 離散化した2値行動
    gamma = 0.99
    epsilon = 0.1

    memory_size = 10000
    batch_size = 32

    result_dir = os.path.join("./result/pendulum", now_str())

    # ============================================================
    # インスタンス作成（環境・モデル・方策・ログ記録）
    # ============================================================
    os.makedirs(result_dir, exist_ok=True)
    print(result_dir)

    # Gymnasium では Pendulum-v1 が標準的
    env = gym.make("Pendulum-v1")

    # Gymnasium の observation_space から状態次元を取得
    dim_state = env.observation_space.shape[0]

    q_network = Qnetwork(dim_state, actions_list, gamma=gamma)
    policy = EpsilonGreedyPolicy(q_network, epsilon=epsilon)

    header = ["num_episode", "loss", "td_error", "reward_avg"]
    recorder = RecordHistory(os.path.join(result_dir, "history.csv"), header)
    recorder.generate_csv()

    # ============================================================
    # warmup（経験メモリの初期化）
    # ============================================================
    print("warming up {:,} steps...".format(n_warmup_steps))

    memory = []
    total_step = 0
    step = 0

    # Gymnasium reset() は (obs, info)
    state, _info = env.reset()

    while True:
        step += 1
        total_step += 1

        # warmup は完全ランダム
        action = random.choice(actions_list)

        # Pendulum の action は shape=(1,) を想定するので ndarray 化して渡す
        env_action = np.array([action], dtype=np.float32)

        # Gymnasium step() は (obs, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, info = env.step(env_action)
        done = bool(terminated or truncated)

        # reward clipping（元コード踏襲）
        if reward < -1:
            c_reward = -1
        else:
            c_reward = 1

        memory.append((state, action, c_reward, next_state, done))
        state = next_state

        # Pendulum は自然終端が分かりにくいので max_step で区切る（元コード踏襲）
        if step > max_step or done:
            state, _info = env.reset()
            step = 0

        if total_step > n_warmup_steps:
            break

    memory = memory[-memory_size:]
    print("warming up {:,} steps... done.".format(n_warmup_steps))

    # ============================================================
    # training（学習本体）
    # ============================================================
    print("training {:,} episodes...".format(max_episode))

    num_episode = 0
    episode_loop = True

    while episode_loop:
        num_episode += 1
        step = 0
        step_loop = True

        episode_reward_list, loss_list, td_list = [], [], []

        # Gymnasium reset() の戻り値に注意
        state, _info = env.reset()

        while step_loop:
            step += 1
            total_step += 1

            # ε-greedy による行動選択
            action, epsilon, q_values = policy.get_action(state, actions_list)

            env_action = np.array([action], dtype=np.float32)
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)

            # reward clipping
            if reward < -1:
                c_reward = -1
            else:
                c_reward = 1

            memory.append((state, action, c_reward, next_state, done))
            episode_reward_list.append(c_reward)

            # experience replay で更新
            exps = random.sample(memory, batch_size)
            loss, td_error = q_network.update_on_batch(exps)
            loss_list.append(loss)
            td_list.append(td_error)

            # target の soft update
            q_network.sync_target_network(soft=0.01)

            state = next_state
            memory = memory[-memory_size:]

            # エピソード終了（元コードは max_step で打ち切り）
            if step >= max_step or done:
                step_loop = False

                reward_avg = float(np.mean(episode_reward_list))
                loss_avg = float(np.mean(loss_list))
                td_error_avg = float(np.mean(td_list))

                print(
                    "{}episode  reward_avg:{} loss:{} td_error:{}".format(
                        num_episode, reward_avg, loss_avg, td_error_avg
                    )
                )

                if num_episode % interval == 0:
                    model_path = os.path.join(
                        result_dir, "episode_{}.h5".format(num_episode)
                    )

                    # 注意: ここは Keras の save 前提
                    # Qnetwork を PyTorch 版に置き換えた場合は torch.save に変更する必要がある
                    q_network.main_network.save(model_path)

                    history = {
                        "num_episode": num_episode,
                        "loss": loss_avg,
                        "td_error": td_error_avg,
                        "reward_avg": reward_avg,
                    }
                    recorder.add_histry(history)

        if num_episode >= max_episode:
            episode_loop = False

    env.close()
    print("training {:,} episodes... done.".format(max_episode))


if __name__ == "__main__":
    train()
