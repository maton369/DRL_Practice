import csv
from datetime import datetime

import numpy as np


def now_str(str_format="%Y%m%d%H%M"):
    """
    現在時刻を文字列として返すユーティリティ関数。

    典型的な用途:
    - ログファイル名にタイムスタンプを付ける（衝突回避）
    - 学習実験の出力ディレクトリを時刻で切る
    - CSV などの記録に「いつ実行したか」を残す

    引数:
        str_format: str
            datetime.strftime に渡すフォーマット文字列。
            デフォルト '%Y%m%d%H%M' は「年/月/日/時/分」をゼロ埋めで並べる形式で、
            例: 202601070845 のような「ソートしやすい時刻文字列」になる。

    戻り値:
        str: 指定フォーマットに整形された現在時刻

    注意:
    - datetime.now() はローカルタイム（OS設定のタイムゾーン）に依存する。
      研究用途でタイムゾーンの混乱を避けたい場合は datetime.utcnow() を使うなども検討する。
    """
    return datetime.now().strftime(str_format)


def idx2mask(idx, max_size):
    """
    インデックス（またはインデックス集合）から 0/1 のマスクベクトルを生成する関数。

    何をしているか:
    - 長さ max_size のベクトル mask を 0 で初期化する
    - idx で指定された位置を 1.0 にする

    典型的な用途（強化学習・機械学習でよくある）:
    - 行動の one-hot 表現（離散行動 a を one-hot にする）
    - 特定の要素だけを有効にするフィルタ（選択マスク）
    - 訓練データのサンプリング結果をベクトル化する

    引数:
        idx:
            - int の場合:
                例: idx=3, max_size=5 -> [0,0,0,1,0]
            - list / np.ndarray などの複数インデックスの場合:
                例: idx=[1,4], max_size=6 -> [0,1,0,0,1,0]
            NumPy のインデクシング規則に従って指定できる。

        max_size: int
            マスクベクトルの長さ。

    戻り値:
        np.ndarray (shape=(max_size,))
            指定された位置が 1.0、それ以外が 0.0 のベクトル。

    注意（落とし穴）:
    - idx が範囲外だと IndexError になる。
    - mask の dtype はデフォルトで float64（np.zeros の既定）になる。
      もし float32 にしたいなら np.zeros(max_size, dtype=np.float32) のように指定する。
    - idx が負の値だと Python/NumPy の仕様で末尾から数える挙動になる（意図したものか要注意）。
    """
    mask = np.zeros(max_size)
    mask[idx] = 1.0
    return mask


class RecordHistory:
    """
    実験ログ・学習履歴などを CSV に追記していくための小さなヘルパークラス。

    ねらい:
    - ヘッダ（列名）を統一して CSV を生成する
    - 1行分の履歴（辞書または配列）を簡単に追記できるようにする

    想定ユースケース（強化学習/機械学習）:
    - エポックごとの loss, accuracy, reward などを CSV に保存
    - ハイパーパラメータ探索の結果を逐次追記
    - 学習中の評価指標を時系列で残し、後から pandas / Excel で解析する

    設計方針:
    - header を「CSVの列順の定義」として保持する
    - add_histry(history) は「辞書から header の順に値を取り出して書く」
      という役割にする（列順のブレを防ぐ）

    注意:
    - クラス名/メソッド名にタイポがある（add_histry）。
      実運用では add_history に直した方がよい。
    """

    def __init__(self, csv_path, header):
        """
        引数:
            csv_path: str
                出力する CSV ファイルのパス。
                例: "runs/202601070845/history.csv"

            header: list[str]
                CSV のヘッダ（列名）を表すリスト。
                この順番が、そのまま CSV の列の順番になる。

                例:
                    header = ["step", "loss", "accuracy"]
                の場合、CSV の各行は step, loss, accuracy の順で書かれる。

        このコンストラクタは “ファイルを作る” ことはしない。
        実際にヘッダ付き CSV を作りたい場合は generate_csv() を呼ぶ。
        """
        self.csv_path = csv_path
        self.header = header

    def generate_csv(self):
        """
        ヘッダ行を持つ CSV ファイルを新規作成する。

        何をしているか:
        - open(..., 'w') でファイルを新規作成（既存ファイルは上書きされる）
        - csv.writer で header を 1 行目に書き込む

        注意:
        - 既に同名ファイルがある場合、内容は消える（上書き）。
        - Windows 環境で “空行が挟まる” 問題を避けるには newline="" を指定するのが定石。
          例: open(self.csv_path, 'w', newline='')
        - 文字化けを避けるには encoding='utf-8'（必要なら utf-8-sig）を指定することも多い。
        """
        with open(self.csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def add_histry(self, history):
        """
        1ステップ分の履歴を CSV に追記する（辞書入力版）。

        引数:
            history: dict
                例:
                    history = {"step": 10, "loss": 0.12, "accuracy": 0.95}
                のように、header に対応するキーを持つ辞書を想定する。

        何をしているか:
        - header の順番で history[key] を取り出してリスト化
        - open(..., 'a') で追記モードで開く
        - csv.writer.writerow(...) で 1 行追記

        注意（落とし穴）:
        - history に header のキーが存在しないと KeyError になる。
          例: header=["step","loss"] なのに history={"step": 1} だと "loss" が無くて落ちる。
        - 逆に history に余分なキーがあっても無視される（header に含まれない列は書かれない）。
        - header が CSV の列順の “唯一の真実” になるため、header の更新には注意が必要。
        """
        # header の順番に値を取り出すことで、列順がブレないログにできる
        history_list = [history[key] for key in self.header]

        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(history_list)

    def add_list(self, array):
        """
        1行分を配列（リスト）として、そのまま CSV に追記する。

        add_histry との使い分け:
        - add_histry: dict 入力（キーで列を指定） -> header の順で並べて書く
        - add_list  : list 入力（既に並び順が揃っている） -> そのまま書く

        引数:
            array: list
                例:
                    [10, 0.12, 0.95]
                のような “1行分の値” を格納した配列。

        注意:
        - ここでは header との長さ一致チェックをしていない。
          array の長さが header と違ってもそのまま書かれるため、
          CSV の列数がずれたログが混ざる可能性がある。
          実務では長さチェックを入れると安全である。
        """
        with open(self.csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(array)
