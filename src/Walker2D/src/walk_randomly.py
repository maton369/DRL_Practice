"""
overview:
    Gymnasium + PyBulletGym の Walker2D 環境を 1 エピソードだけランダム行動で回し、
    各ステップの報酬を表示するサンプルコード。

ポイント（gym → gymnasium 変更で重要な点）:
  - import は gymnasium を使う: import gymnasium as gym
  - reset() の戻り値が (obs, info) になる
  - step() の戻り値が (obs, reward, terminated, truncated, info) になる
    → エピソード終了判定は done = terminated or truncated とする

注意:
  - pybulletgym は「gym 互換」の環境を提供するが、環境によっては gymnasium で動かす際に
    互換ラッパーが必要になる場合がある（バージョン依存）。
  - まずはこのコードで動作確認し、もし step/reset の戻りが旧形式なら分岐対応すると安全。
"""

import gymnasium as gym
import pybulletgym.envs  # これを import することで PyBulletGym の環境IDが登録される


# ------------------------------------------------------------
# 環境の生成
# ------------------------------------------------------------
# Walker2D の PyBullet 環境を作る
# - 'Walker2DPyBulletEnv-v0' は pybulletgym が提供する環境ID
env = gym.make("Walker2DPyBulletEnv-v0")

# ------------------------------------------------------------
# ここでは 1 エピソードだけ実行する
# ------------------------------------------------------------
for episode in range(1):

    # --------------------------------------------------------
    # reset:
    # Gymnasium は (obs, info) を返す
    # obs: 観測（状態ベクトル）
    # info: デバッグ情報など（辞書）
    # --------------------------------------------------------
    state, info = env.reset()

    # エピソード内のステップ数を数えておく（ログ用）
    step = 0

    while True:
        step += 1

        # ----------------------------------------------------
        # ランダム行動
        # action_space.sample() は環境の行動空間からランダムに行動を生成
        # ここでは学習ではなく「環境が動くか」を確認するデモ
        # ----------------------------------------------------
        action = env.action_space.sample()

        # ----------------------------------------------------
        # step:
        # Gymnasium は
        #   (obs, reward, terminated, truncated, info)
        # を返す
        #
        # terminated: タスクとしての終了条件（転倒・失敗など）で終わった
        # truncated : 時間制限など外部要因で打ち切られた
        # ----------------------------------------------------
        state_new, r, terminated, truncated, info = env.step(action)

        # 終了判定は terminated または truncated のどちらか
        done = bool(terminated or truncated)

        # ----------------------------------------------------
        # 報酬の表示
        # r はそのステップで得られた即時報酬
        # Walker2D では前進・姿勢維持などに応じて報酬が設計されている
        # ----------------------------------------------------
        print("step:", step, "reward:", r)

        # 次状態へ更新（このデモでは学習しないが、通常はここで経験を保存する）
        state = state_new

        # ----------------------------------------------------
        # エピソード終了
        # ----------------------------------------------------
        if done:
            print("episode done")
            # info に追加情報が入っている場合があるので、必要ならここで確認可能
            # print("info:", info)
            break

# 後片付け（GUI/物理エンジンリソースの解放）
env.close()
