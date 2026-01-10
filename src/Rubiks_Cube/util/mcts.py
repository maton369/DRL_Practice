"""
overview:
    ルービックキューブ環境に対して、ニューラルネット（Actor-Critic）の推論を組み合わせた
    MCTS（Monte Carlo Tree Search）を実行し、最良経路（行動列）を探索する。

    - Selection: UCB風の指標（探索項 U + 価値項 Q）で子ノードを辿る
    - Expansion: 末端ノードに到達したら、Actor（方策）で子を展開する
    - Evaluation: Critic（価値）で終端以外を評価し、未解決ペナルティも加える
    - Backup: 得られた評価値を経路上の各ノードへ加算（訪問回数と価値和を更新）

    ※ この実装は「純粋なランダムロールアウト」ではなく、
       NNの価値推定（state value）を使う“value-based evaluation”寄りの MCTS である。

args:
    agent:
        - predict_policy(sess, [state]) -> 行動確率（Actor）
        - predict_value(sess, [state])  -> 状態価値（Critic）
      を提供するエージェント（TF版のsessを渡す前提の設計）。

output:
    run_search() が以下を返す:
        best_reward: 割引報酬和（推定含む）の最大値
        best_solved: 解けたかどうか（done）
        best_states: 最良経路の状態列（root から辿った軌跡）
        best_actions: 最良経路の行動列

note:
    - env は RubiksCubeEnv を内部に保持し、run_search 中に env.set_state() で状態を巻き戻す。
    - Node.v は「平均価値」ではなく「価値の累積和（バックアップで加算される合計）」として使われ、
      UCB計算時に (1+n) で割って平均化したものを Q として利用している。
"""

import time
import numpy as np
from math import sqrt

from gym_env.rubiks_cube_env import RubiksCubeEnv

# MCTS のハイパーパラメータ（探索と評価のバランス、計算制約など）
mcts_config = {
    "gamma": 0.99,  # 割引率（深い手を評価するときに報酬を割引く）
    "mu": 1.0e00,  # exploration weight（探索項 U の重み）
    "nu": 1.0e-01,  # ※現状このコードでは未使用（将来の拡張用の可能性）
    "max_actions": 15,  # rollout（ランダム延長）する場合の最大手数（現状は rollout 無効）
    "unsolved_penalty": -10.0,  # 未解決のまま評価する際に加える罰則（探索を解へ誘導）
    "time_limit": 10,  # 探索の最大時間（秒）
    "max_runs": 1000,  # 探索の最大反復回数（シミュレーション回数）
}


class MCTS(object):
    """
    ニューラルネット（Actor-Critic）支援型の MCTS を実装するクラス。

    - env: RubiksCubeEnv（状態を set_state で巻き戻せることが前提）
    - agent: 方策（Actor）と価値（Critic）を推論できるモデル
             ※現在は TensorFlow の sess を受け取る API になっている。
    """

    # コンストラクタ
    def __init__(self, agent):
        # 探索で使う環境（MCTS 内部で状態を巻き戻しながら何度も step する）
        self.env = RubiksCubeEnv()

        # 方策・価値推定器（外部から注入）
        self.agent = agent

        # 行動の全候補（離散行動の列）
        self.act_list = self.env.get_action_list()

        # 割引率
        self.gamma = mcts_config["gamma"]

        # exploration weight（UCBの探索項の重み c に相当）
        self.mu = mcts_config["mu"]

        # rollout の最大手数（現状 rollout 部分は if 0 で無効）
        self.max_actions = mcts_config["max_actions"]

        # 未解決時の罰則
        self.unsolved_penalty = mcts_config["unsolved_penalty"]

        # 計算制約（時間/反復回数）
        self.time_limit = mcts_config["time_limit"]
        self.max_runs = mcts_config["max_runs"]

    # 探索の遂行
    def run_search(self, sess, root_state):
        """
        root_state から探索を開始し、制約時間/回数の範囲で MCTS を回す。

        アルゴリズムの流れ（典型的な MCTS 4ステップ）:
          1) Selection: 既に展開済みの木を UCB 指標で下降して leaf（未展開 or 終端）へ
          2) Expansion: leaf が非終端なら、Actor の確率で子ノード群を生成
          3) Evaluation: leaf の状態価値を Critic で推定（+ 未解決ペナルティ）
          4) Backup: 評価値を経路上のノードに加算し訪問回数を増やす

        返り値は「探索中に見つかった最良経路」に対応する (reward, solved, states, actions)。
        """

        # --- PRE-PROCESS ---
        # ルートノードを生成（親なし・行動なし・確率なし）
        # ※ root は「状態を表す」よりも「木の起点の器」として扱われる
        root_node = Node(None, None, None)

        # 最良経路を記録するバッファ（探索の最終出力用）
        best_reward = float("-inf")
        best_solved = False
        best_actions = []

        # --- SEARCH MAIN ---
        n_run, n_done = 0, 0
        start_time = time.time()

        # 経路探索ループ（シミュレーションの繰り返し）
        while True:
            # 1回のシミュレーションごとに root から辿り直す
            node = root_node
            state = root_state
            self.env.set_state(root_state)  # 環境状態を root に巻き戻す

            # このシミュレーションで得られた割引報酬和（評価値）
            weighted_reward = 0.0

            # env.step が返す done をリスト型で扱っている前提
            done = [False]

            # このシミュレーションで辿った行動列（最良経路更新に使う）
            actions = []

            # --- 1) Selection ---
            # 既に展開済み（child_nodes を持つ）なら、UCBで子を選びながら深く降りる
            n_depth = 0
            while node.child_nodes:
                # UCB 指標により次ノードを選択
                node = self._select_next_node(node.child_nodes)

                # 選択した行動を環境に適用し、遷移・報酬を取得
                next_state, reward, done, _ = self.env.step(node.action)

                # 割引報酬を加算（深いほど gamma^depth が効く）
                weighted_reward += (self.gamma**n_depth) * reward[0]

                n_depth += 1
                state = next_state
                actions.append(node.action)

            # --- 2) Expansion ---
            # leaf が終端でない場合、Actor（方策）で「全行動の確率」を出して子を展開する
            if not done[0]:
                action_probs = self.agent.predict_policy(
                    sess, [state]
                )  # shape: (1, |A|)
                node.child_nodes = [
                    Node(node, act, act_prob)
                    for act, act_prob in zip(self.act_list, action_probs[0])
                ]

            # --- 3) Evaluation ---
            # leaf が終端でない場合、Critic で状態価値を推定し、評価値として加える
            if not done[0]:
                # utilize state value（価値関数で葉を評価する）
                if 1:
                    # Critic による状態価値推定 V(s_leaf)
                    _v_s = self.agent.predict_value(sess, [state])

                    # 深さ分だけ割引して加算（leaf の価値も将来報酬として扱う）
                    weighted_reward += (self.gamma**n_depth) * _v_s[0][0]

                    # 「解けていない」状態に罰則を加えることで、
                    # 価値推定だけでは探索が迷走するのを抑える意図がある
                    weighted_reward += self.unsolved_penalty

                # --- (参考) rollout による評価 ---
                # ここは if 0 で無効。純ロールアウト評価は計算コストが高く、
                # ルービックキューブでは探索が非効率になりやすいので、
                # 現状は Critic を使った value evaluation に寄せている。
                if 0:
                    n_rollout_step = 0
                    state = None
                    done = [False]

                    while not done[0]:
                        action = np.random.choice(self.act_list)
                        state, reward, done, _ = self.env.step(action)
                        depth = n_depth + n_rollout_step
                        weighted_reward += (self.gamma**depth) * reward[0]

                        n_rollout_step += 1
                        actions.append(action)

                        # ロールアウトが長すぎるのを防ぐ
                        if len(actions) >= self.max_actions:
                            break

                    # まだ終端でなければ末端価値を足す（bootstrap）
                    if not done[0]:
                        _v_s = self.agent.predict_value(sess, [state])
                        depth = n_depth + n_rollout_step
                        weighted_reward += (self.gamma**depth) * _v_s[0][0]

            # --- 4) Backup ---
            # 得られた評価値（weighted_reward）を経路上へ反映する
            # node から親へ辿りながら:
            #   - n: 訪問回数を増やす
            #   - v: 評価値を累積する（合計）
            while node:
                node.n += 1
                node.v += weighted_reward
                node = node.parent_node

            # --- ベスト経路の更新 ---
            # 「このシミュレーションの評価値」が最良なら、行動列と結果を保存
            if best_reward < weighted_reward:
                best_reward = weighted_reward
                best_solved = done[0]
                best_actions = actions

            # --- 終了条件の判定 ---
            n_run += 1
            if done[0]:
                n_done += 1  # 解けた回数の統計（現状は返していないが、診断に使える）
            duration = time.time() - start_time
            if n_run >= self.max_runs or duration >= self.time_limit:
                break

        # --- POST-PROCESS ---
        # 探索で得た best_actions を root から実際に辿り直し、
        # 状態列 best_states と「実際の割引報酬和 best_reward」を再計算する
        self.env.set_state(root_state)

        best_reward = 0.0
        best_solved = False
        best_states = [root_state]

        for i_act, action in enumerate(best_actions):
            next_state, reward, done, _ = self.env.step(action)

            # 探索中の評価値は Critic/ペナルティ込みだったが、
            # ここでは「環境報酬のみ」の割引和を作り直している
            best_reward += (self.gamma**i_act) * reward[0]
            best_states.append(next_state)

            if done[0]:
                best_solved = True
                break

        return best_reward, best_solved, best_states, best_actions

    # select next node based on a metric (ucb)
    def _select_next_node(self, child_nodes):
        """
        子ノード集合から、UCB風の指標が最大のノードを選ぶ。

        metric = U + Q
          - U: 探索項（訪問回数が少ないノードを押し上げる）
          - Q: 価値項（平均価値が高いノードを押し上げる）
        """
        metric = [self._calc_metric(node) for node in child_nodes]
        best_node = child_nodes[np.argmax(metric)]
        return best_node

    # calc node evaluation (ucb) metric
    def _calc_metric(self, node):
        """
        UCB風評価値を計算する。

        この実装の形:
          U = mu * p * sqrt(N_parent) / (1 + N_child)
          Q = V_sum / (1 + N_child)

        - p は Actor が出した「その行動の事前確率」。
          AlphaZero の PUCT に近い形で、prior を探索に組み込んでいる。
        - node.v は累積価値（合計）として加算されているので、
          (1 + node.n) で割って平均価値 Q を作っている。

        注意:
          親ノードの訪問回数 node.parent_node.n が 0 の初期は sqrt(0)=0 になり、
          U が効かず Q だけで選択される（ただし子は未訪問なので Q=0 で並ぶことが多い）。
          実運用では「未訪問子を優先する」などの初期処理を入れることもある。
        """
        # 探索項: prior p を掛けることで、Actor の信念を探索に反映する
        _u = self.mu * node.p * sqrt(node.parent_node.n) / (1.0 + node.n)

        # 価値項: 累積価値を平均化（訪問回数が増えるほど安定化）
        _q = node.v / (1.0 + node.n)

        return _u + _q


class Node(object):
    """
    MCTS の探索木ノード。

    - parent_node: 親ノード参照（バックアップで親へ遡るために必要）
    - action: 親からこのノードへ遷移するための行動
    - p: その行動が選ばれる事前確率（Actor の出力）
    - v: このノードを通ったシミュレーションで得られた評価値の累積和
    - n: 訪問回数
    - child_nodes: 展開した子ノード群
    """

    def __init__(self, parent, action, prob):
        # 親ノード
        self.parent_node = parent

        # 親からの到達行動（root では None）
        self.action = action

        # Actor が出した prior probability（root では None）
        self.p = prob

        # 累積価値（バックアップで加算され続ける）
        self.v = 0.0

        # 訪問回数
        self.n = 0

        # 子ノードリスト（展開されるまでは空）
        self.child_nodes = []
