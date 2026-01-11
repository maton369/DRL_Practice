import os

import numpy as np
import torch

from agent import (
    Agent,
)  # ※Torch版に置き換えた Agent を想定（Generator / Rollouter を内包）
from datagenerator import Vocab, DataForGenerator, DataForDiscriminator
from environment import (
    Environment,
)  # ※Torch版に置き換えた Environment を想定（Discriminator を内包）

"""
overview:
    SeqGANの学習（事前学習 + 強化学習による adversarial training）を行う（PyTorch版）

元コード（TF1 + Keras）でやっていたことを、Torch の「逐次実行 + autograd」に寄せて書き換える。
- TF の Session / placeholder / K.set_session は不要になる。
- その代わり、Agent/Environment 側で optimizer と forward/backward を持つ構成にする。

学習の流れ（SeqGANの定番構成）:
1) 生成器Gの事前学習（教師あり; MLE）:
   - 正例データ（実文）を次トークン予測する言語モデルとして学習
2) 識別器Dの事前学習（教師あり; 二値分類）:
   - 正例=実文, 負例=事前学習済みGの生成文 を見分ける
3) adversarial training（強化学習; REINFORCE）:
   - Gは「文を生成する方策」とみなし、Dのスコアを報酬として方策勾配で更新
   - 部分列の中間状態に対する報酬は Monte Carlo rollout（rollout policy）で推定

※この main は “制御ループ” を担当し、学習の具体（損失計算・最適化）は
  Agent/Environment 側に寄せる想定にしている。
"""

# ============================================================
# 1. 乱数・デバイス設定
# ============================================================

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. ハイパーパラメータ（元コードの値を踏襲）
# ============================================================

batch_size = 30

T = 25  # 文章の最大長（max_length）
emb_size = 128  # embedding size

g_hidden = 128  # generator hidden size
d_hidden = 64  # discriminator hidden size

g_lr = 1e-3  # adversarial training における G の学習率（RL）
d_lr = 1e-3  # adversarial training における D の学習率（RL）

dropout = 0.0

# 事前学習（pretraining）
g_pre_lr = 1e-2
d_pre_lr = 1e-2
g_pre_episodes = 10
d_pre_episodes = 4
d_epochs = 1  # 互換のため残す（Torch側では “epochs” で回すのが自然）

# adversarial training（強化学習）
adversarial_nums = 10  # adversarial の “外側ループ” 回数
g_train_nums = 1  # 1 adversarial loop 内で G を何回更新するか
d_train_nums = 1  # 1 adversarial loop 内で D を何回更新するか
g_episodes = 50  # 1回の G 更新で何文サンプルして学習するか
n_sampling = 16  # Monte Carlo rollout の本数
frequency = 1  # 何 adversarial ごとにサンプル文を保存するか

# ============================================================
# 3. 入出力パス（元コードの出力仕様を踏襲）
# ============================================================

input_data = os.path.join("data", "input.txt")
id_input_data = os.path.join("data", "id_input.txt")

pre_output_data = os.path.join("data", "pre_generated_sentences.txt")
pre_id_output_data = os.path.join("data", "pre_id_generated_sentences.txt")

output_data = os.path.join("data", "generated_sentences.txt")
id_output_data = os.path.join("data", "id_generated_sentences.txt")

# 旧TF/Kerasでは .h5 を保存していたが、Torch では state_dict を .pt/.pth にするのが一般的
# （拡張子だけの違いで意味は「重みを保存」）
os.makedirs("data/save", exist_ok=True)
g_pre_weight = os.path.join("data", "save", "pre_g_weights.pt")
d_pre_weight = os.path.join("data", "save", "pre_d_weights.pt")

# ============================================================
# 4. 前処理（語彙構築・id化）
# ============================================================

vocab = Vocab(input_data)
vocab_size = vocab.vocab_num
pos_sentence_num = vocab.sentence_num

# input.txt を単語ID列に変換した id_input.txt を生成
vocab.write_word2id(input_data, id_input_data)

sampling_num = vocab.data_num  # 生成サンプル数（元コード踏襲）

# ============================================================
# 5. Agent / Environment の構築（Torch版を想定）
# ============================================================

# Environment は Discriminator を持つ（= reward を返す “審判”）
# - pre_train: 教師ありで D を学習
# - initialize: 重みロード
# - discriminator: forward/predict を提供（推論は no_grad）
env = Environment(
    batch_size=batch_size,
    vocab_size=vocab_size,
    emb_size=emb_size,
    d_hidden=d_hidden,
    T=T,
    dropout=dropout,
    d_lr=d_lr,
    device=device,
)

# Agent は Generator（学習対象）と Rollouter（rollout policy; Gのコピー）を持つ
# - pre_train: 教師ありで G を学習
# - get_action / rollout: RL用のサンプリングAPI
# - generator.update: 方策勾配更新（REINFORCE）
agent = Agent(
    vocab_size=vocab_size,
    emb_size=emb_size,
    g_hidden=g_hidden,
    T=T,
    g_lr=g_lr,
    device=device,
)

# ============================================================
# 6. 事前学習（Generator MLE → Generatorで負例生成 → Discriminator 学習）
# ============================================================


def pre_train():
    # -------------------------
    # (1) Generator の事前学習
    # -------------------------
    # DataForGenerator は “正例の token 列” をバッチで供給するデータローダ役
    g_data = DataForGenerator(
        id_input_data,
        batch_size,
        T,
        vocab,
    )

    # G を「言語モデル」として教師あり学習
    # 典型: teacher forcing で次トークン予測の cross entropy を最小化
    agent.pre_train(
        g_data=g_data,
        epochs=g_pre_episodes,
        save_path=g_pre_weight,
        lr=g_pre_lr,
    )

    # -------------------------
    # (2) 事前学習済みGで負例を生成
    # -------------------------
    # “生成文ID列” をファイルに出す（SeqGANの典型実装がこれ）
    agent.generate_id_samples(
        generator=agent.generator,
        T=T,
        sampling_num=sampling_num,
        out_path=pre_id_output_data,
    )

    # 生成されたID列を単語列に戻して保存（可読なログ用）
    vocab.write_id2word(pre_id_output_data, pre_output_data)

    # -------------------------
    # (3) Discriminator の事前学習
    # -------------------------
    # DataForDiscriminator は “正例（実データ）と負例（生成データ）” を混ぜて返す
    d_data = DataForDiscriminator(
        id_input_data,
        pre_id_output_data,
        batch_size,
        T,
        vocab,
    )

    # D を教師あり学習（2値分類: real=1 / fake=0）
    env.pre_train(
        d_data=d_data,
        epochs=d_pre_episodes,
        save_path=d_pre_weight,
        lr=d_pre_lr,
    )


# ============================================================
# 7. adversarial training（SeqGANのRLパート）
# ============================================================


def train():
    # 事前学習重みをロードして adversarial の初期値にする
    agent.initialize(g_pre_weight)
    env.initialize(d_pre_weight)

    for adversarial_num in range(adversarial_nums):
        print("---------------------------------------------")
        print("Adversarial Training:", adversarial_num + 1)

        # -------------------------
        # (1) Generator をRLで更新
        # -------------------------
        for _ in range(g_train_nums):
            g_train()

        print("Generator is trained")

        # -------------------------
        # (2) Discriminator を更新（Gが変わるので見分け直し）
        # -------------------------
        for _ in range(d_train_nums):
            d_train()

        print("Discriminator is trained")

        # -------------------------
        # (3) 定期的に生成文を保存（学習の可視化）
        # -------------------------
        if adversarial_num % frequency == 0:
            sentences_history(
                adversarial_num,
                agent,
                T,
                vocab,
                sampling_num,
            )


def g_train():
    """
    Generator の方策勾配更新（REINFORCE）。

    SeqGANの見方:
    - 状態 s_t : これまで生成した prefix（トークン列）
    - 行動 a_t : 次に出すトークン
    - 方策 πθ(a_t|s_t) : Generator が出すカテゴリ分布
    - 報酬 R : 完成文に対する Discriminator のスコア（= real らしさ）

    ただし中間ステップ t<T の報酬は直接得られないので、
    Monte Carlo rollout で prefix を最後まで補完して D を評価し、
    期待報酬を近似して r_t（あるいは Q(s_t,a_t)）にする。
    """
    # ここでは “複数文を生成してまとめて更新” するため、バッファを作る。
    # 元コードが numpy concat でまとめていたのに合わせて、同様に構成する。
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_hs = []
    batch_cs = []

    for _ in range(g_episodes):
        # RNN 状態（LSTMの hidden/cell）を初期化
        agent.reset_rnn_states()

        # prefix の初期値は BOS（文頭トークン）
        states = torch.zeros((1, 1), dtype=torch.long, device=device)
        states[:, 0] = int(vocab.BOS)

        actions = []
        rewards = []

        # hidden/cell の初期値（Agent側が持つなら不要だが、元コードの構造に寄せる）
        # ここでは “次状態保存用” の配列として保持している。
        hs = []
        cs = []

        for step in range(T):
            # Generator（方策）から次トークンをサンプル
            # 戻り値:
            # - action: shape [1,1] の token id
            # - next_h/next_c: LSTM状態（rolloutの初期状態に使う）
            action, next_h, next_c = agent.get_action(states)

            # rollout policy（rollouter）を初期化し、現在の next_h/next_c をセット
            # これにより「このprefixの続き」を同じ内部状態からサンプルできる
            agent.rollouter.reset_rnn_state()

            # 途中stepの場合: rollout を n_sampling 回行って D の平均スコアを報酬にする
            reward = mc_search(step, states, action, next_h, next_c)

            # prefix を1トークン進める（states に action を連結）
            states = torch.cat([states, action], dim=-1)

            rewards.append(reward)  # shape [1,1] を想定
            actions.append(action)  # shape [1,1]

            hs.append(next_h)
            cs.append(next_c)

        # states は末尾まで含んでいるので、学習入力としては “行動前のprefix” に揃える
        # 例: states = [BOS, a0, a1, ..., a(T-1)]
        #     actions = [a0, a1, ..., a(T-1)]
        # よって states[:, :-1] を使う
        train_states = states[:, :-1]

        # まとめてバッファへ
        batch_states.append(train_states)
        batch_actions.append(torch.cat(actions, dim=-1))  # [1,T]
        batch_rewards.append(torch.cat(rewards, dim=-1))  # [1,T]

        # LSTM状態は Agent実装次第だが、元コード互換で “時刻ごとの状態” を渡す想定
        # 形式は Agent.generator.update が決める（ここではそのまま積む）
        batch_hs.append(torch.cat(hs, dim=0))  # 例: [T, hidden] など
        batch_cs.append(torch.cat(cs, dim=0))

    # バッチ方向に結合
    batch_states = torch.cat(batch_states, dim=0)  # [g_episodes, T]
    batch_actions = torch.cat(batch_actions, dim=0)  # [g_episodes, T]
    batch_rewards = torch.cat(batch_rewards, dim=0)  # [g_episodes, T]
    batch_hs = torch.cat(batch_hs, dim=0)
    batch_cs = torch.cat(batch_cs, dim=0)

    # Generator を方策勾配で更新
    # 典型: loss = -E[ sum_t (R_t * log π(a_t|s_t)) ]
    # R_t は rollout で推定した期待報酬（= Q近似）
    agent.generator.update(
        states=batch_states,
        actions=batch_actions,
        rewards=batch_rewards,
        hs=batch_hs,
        cs=batch_cs,
    )

    # rollout policy を最新の Generator 重みに追従させる
    agent.inherit_weights(agent.generator, agent.rollouter)


def d_train():
    """
    Discriminator の更新。
    - Generator が変わると fake 分布も変わるので、D は追従して学習し直す必要がある。
    - ここでは「最新Gで負例生成 → Dを1epoch学習」を行う。
    """
    # 最新Gで負例（生成文）を作る
    agent.generate_id_samples(
        generator=agent.generator,
        T=T,
        sampling_num=sampling_num,
        out_path=id_output_data,
    )
    vocab.write_id2word(id_output_data, output_data)

    # D用データセット（正例=実文ID, 負例=生成文ID）
    d_data = DataForDiscriminator(
        id_input_data,
        id_output_data,
        batch_size,
        T,
        vocab,
    )

    # Torch版では Keras の fit_generator 相当を Environment 側で提供する想定
    # 例: env.train_discriminator(d_data, epochs=1)
    env.train_discriminator(d_data=d_data, epochs=1)


def sentences_history(episode, agent, T, vocab, sampling_num):
    """
    学習過程の可視化用に、特定 adversarial episode 時点の生成文を保存する。
    """
    id_output_history = os.path.join(
        "data",
        "adversarial_{}_id_generated_sentences.txt".format(episode + 1),
    )
    output_history = os.path.join(
        "data",
        "adversarial_{}_generated_sentences.txt".format(episode + 1),
    )

    agent.generate_id_samples(
        generator=agent.generator,
        T=T,
        sampling_num=sampling_num,
        out_path=id_output_history,
    )
    vocab.write_id2word(id_output_history, output_history)


def mc_search(step, states, action, next_h, next_c):
    """
    Monte Carlo rollout による報酬推定。

    目的:
    - step < T-1 の途中段階では、prefix のままだと D が “完成文” を評価できない。
    - そこで rollout policy（rollouter）で残りを補完し、完成文を D に通してスコアを得る。
    - これを n_sampling 回繰り返して平均し、Q(s_t, a_t) の近似として reward_t を作る。

    実装上の注意:
    - Discriminator での推論は学習ではないので torch.no_grad() を使う。
    - rollout は確率的なので分散が大きい。n_sampling を増やすと分散は減るが計算が重い。
    """
    # reward_t は shape [1,1] のテンソルとして揃える（元コード互換）
    reward_t = torch.zeros((1, 1), dtype=torch.float32, device=device)

    agent.rollouter.reset_rnn_state()

    if step < T - 1:
        # rollout の初期RNN状態を “行動後” の状態に合わせる
        agent.rollouter.set_rnn_state(next_h, next_c)

        # n_sampling 回 rollout して、Dのスコア平均を推定
        with torch.no_grad():
            for _ in range(n_sampling):
                # rollout: prefix(states) + action を起点に、残りをサンプルして完成文Yを返す
                # Y は “トークンID列” で、D の入力形式に合わせた shape を返す想定
                Y = agent.rollout(step, states, action)

                # D は「real らしさ」を返す（例: sigmoid 出力）
                # env.discriminator.predict(Y) は shape [1,1] を返す想定
                reward_t += env.discriminator.predict(Y) / float(n_sampling)

    else:
        # 最終ステップは既に完成文になるので rollout 不要
        # states は [BOS, ..., prefix] なので BOS を除いたり整形を合わせる
        Y = torch.cat([states[:, 1:], action], dim=-1)
        with torch.no_grad():
            reward_t = env.discriminator.predict(Y)

    return reward_t


# ============================================================
# 8. エントリポイント
# ============================================================

if __name__ == "__main__":
    pre_train()
    train()
