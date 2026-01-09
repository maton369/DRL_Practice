import numpy as np


class EpsilonGreedyPolicy:
    """
    ε-greedy 方策（探索と活用を混ぜる最も基本的な方策）を表すクラス。

    ε-greedy の考え方:
      - 確率 ε で「探索（random）」: ランダムに行動を選ぶ
      - 確率 1-ε で「活用（greedy）」: 現在のQ関数が最大となる行動を選ぶ

    これにより、
      - 初期段階では探索によって多様な経験を集め
      - 学習が進むにつれて Q が良くなり、活用が効く
    という形で、探索と最適化を両立させる。

    本実装の依存:
      - q_network.main_network.predict_on_batch(state) が
        Q(s, a) を action 次元分（= len(actions_list)）返すことを前提にしている。
      - つまり、この方策は「離散行動（actions_list）」の DQN 系に対応している。
    """

    def __init__(self, q_network, epsilon):
        """
        引数:
            q_network:
                Qネットワークを持つオブジェクト。
                少なくとも q_network.main_network を通して Q値を推定できる必要がある。

            epsilon: float
                探索率 ε。
                例: 0.1 なら 10% の確率でランダム行動、90% の確率で greedy 行動。

        注意:
            εを固定で持つ設計になっているが、実務では
              - 線形減衰（epsilon decay）
              - 指数減衰
              - ステップ数に応じたスケジューリング
            を入れることが多い。
        """
        self.q_network = q_network
        self.epsilon = epsilon

    def get_action(self, state, actions_list):
        """
        現在状態 state に対して ε-greedy で行動を1つ選択する。

        引数:
            state: array-like
                環境から得た現在状態（観測）。
                例: Pendulum なら (3,) のベクトルなど。

            actions_list: list
                離散行動の候補リスト。
                例: [-1, 1] のように2値に離散化している場合など。

        戻り値:
            (action, epsilon, q_values)
                action:
                    選択した行動（actions_list の要素）
                epsilon:
                    現在の ε（ここでは固定値）
                q_values:
                    greedy 選択をした場合は Q(s,·) のベクトル
                    random 選択をした場合は None（計算していないため）

        実装上の流れ:
          1) 一様乱数 u ~ Uniform(0,1) を生成
          2) u < ε なら探索（ランダム行動）
          3) それ以外なら活用（Q値が最大の行動を選ぶ）
        """

        # ------------------------------------------------------------
        # 1) 探索するか（random）/ 活用するか（greedy）を確率で決める
        # ------------------------------------------------------------
        # np.random.uniform() は [0,1) の一様乱数を返す
        # これが ε より小さければ探索（ランダム行動）にする
        is_random_action = np.random.uniform() < self.epsilon

        if is_random_action:
            # --------------------------------------------------------
            # 2-a) 探索（random）
            # --------------------------------------------------------
            # ランダム行動のときは、Q値の計算をしない（計算コスト削減）
            # ただしログのためにQ値を残したいならここで推定してもよい。
            q_values = None

            # actions_list から一様に行動を選ぶ
            action = np.random.choice(actions_list)

        else:
            # --------------------------------------------------------
            # 2-b) 活用（greedy）
            # --------------------------------------------------------
            # ネットワークに状態を入力して Q(s,·) を推定し、
            # 最大の Q を与える行動を選ぶ。

            # Keras の predict_on_batch は「バッチ入力」を前提にするため、
            # (state_dim,) を (1, state_dim) に reshape している
            state = np.reshape(state, (1, len(state)))

            # Q(s,·) を推定
            # 返り値は shape=(1, action_len) を想定し、[0] で1サンプル分にする
            q_values = self.q_network.main_network.predict_on_batch(state)[0]

            # greedy: argmax で最大 Q の行動インデックスを得て、actions_list に変換
            action = actions_list[np.argmax(q_values)]

        # ------------------------------------------------------------
        # 3) 行動と（固定の）εと、必要ならQ値を返す
        # ------------------------------------------------------------
        return action, self.epsilon, q_values
