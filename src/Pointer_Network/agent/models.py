import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder（PyTorch版）
    入力系列をエンコードして、(1) 各時刻の埋め込み列 enc_outputs と (2) 最終状態 enc_state を返す。

    元コード（TF版）の意図:
      - まず BiLSTM を通して入力系列を「文脈付き表現」に変換する
      - その後、あえて LSTM レイヤを一発で使わず、
        LSTMCell を使って time-step ループを明示的に書いている
        （Decoder 側もループなので、構造を揃えたい意図）

    PyTorch版の構成:
      - nn.LSTM(bidirectional=True, batch_first=True) で BiLSTM を実装
      - その出力（双方向なので次元は 2*n_neurons）を、LSTMCell 用に n_neurons に射影してから
        明示ループで LSTMCell に通す（TF版の「enc_rec_cell: LSTMCell(n_neurons)」を再現するため）

    入出力の shape（batch_first=True 前提）:
      inputs:
        shape = [batch_size, seq_length, input_dim]
      enc_outputs:
        shape = [batch_size, seq_length, n_neurons]
      enc_state:
        (h, c) のタプルで、それぞれ shape = [batch_size, n_neurons]
    """

    def __init__(self, input_dim, n_neurons=128, batch_size=4, seq_length=10):
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.batch_size = int(batch_size)
        self.seq_length = int(seq_length)

        # BiLSTM: 出力は 2*n_neurons（forward/backward を concat）
        self.bilstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=self.n_neurons,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # BiLSTM の出力 (2*n_neurons) を LSTMCell の入力 (n_neurons) に合わせる射影
        self.proj = nn.Linear(self.n_neurons * 2, self.n_neurons)

        # 明示ループで使う LSTMCell（TF版の enc_rec_cell 相当）
        self.enc_rec_cell = nn.LSTMCell(self.n_neurons, self.n_neurons)

    def _get_initial_state(self, device, batch_size=None):
        """
        LSTMCell の初期状態 (h0, c0) を返す。
        TF版は get_initial_state を呼んでいたので、それに相当する処理。
        """
        b = self.batch_size if batch_size is None else int(batch_size)
        h0 = torch.zeros(b, self.n_neurons, device=device)
        c0 = torch.zeros(b, self.n_neurons, device=device)
        return h0, c0

    def forward(self, inputs):
        """
        Encoder の順伝播。

        inputs:
          [B, T, D]

        returns:
          enc_outputs: [B, T, n_neurons]
          enc_state:   (h_T, c_T) それぞれ [B, n_neurons]
        """
        device = inputs.device
        B, T, _ = inputs.shape

        # 1) BiLSTM で文脈付き系列表現を得る: [B, T, 2H]
        bi_outputs, _ = self.bilstm(inputs)

        # 2) LSTMCell 入力に合わせて射影: [B, T, H]
        bi_outputs = torch.tanh(self.proj(bi_outputs))

        # 3) 明示ループで LSTMCell を回す
        h, c = self._get_initial_state(device=device, batch_size=B)
        enc_outputs = []

        for t in range(T):
            x_t = bi_outputs[:, t, :]  # [B, H]
            h, c = self.enc_rec_cell(x_t, (h, c))
            enc_outputs.append(h)

        # [T, B, H] -> [B, T, H]
        enc_outputs = torch.stack(enc_outputs, dim=0).transpose(0, 1)

        # 最終状態（TF版は enc_states[-1] を返していた）
        enc_state = (h, c)
        return enc_outputs, enc_state


class ActorDecoder(nn.Module):
    """
    ActorDecoder（PyTorch版）
    Pointer Network 風の「pointing 機構」を用いて、入力系列の位置（インデックス）を逐次選択するデコーダ。

    これは TSP/巡回路生成などで典型的な構造で、
      - Encoder が作る埋め込み列 enc_outputs（各地点の表現）
      - Decoder が作る query（内部状態）
    から、各地点を「次に選ぶべき確率分布」を作る。

    重要なアルゴリズム要素:
      1) マスク（既訪問地点を選べないようにする）
         - visited の位置に -infty を足して logits を潰す
      2) Categorical 分布からサンプルして tour を生成
      3) 同時に log_prob を積み上げ、方策勾配の目的（logπ）に使う

    出力:
      - log_prob: その tour をサンプルした対数尤度（バッチごとにスカラー）
      - tour: 選択したインデックス列（最初の地点を末尾に追加して閉路にする）

    shape:
      enc_outputs: [B, T, H]
      enc_state:   (h, c) それぞれ [B, H]
      tour:        [B, T+1]
      log_prob:    [B]
    """

    def __init__(self, n_neurons=128, batch_size=4, seq_length=10):
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.batch_size = int(batch_size)
        self.seq_length = int(seq_length)

        # マスク用の大きい定数（-infty 相当）
        self.infty = 1.0e8

        # サンプリング seed（必要なら generator を使う）
        self.seed = None

        # Decoder の再帰セル
        self.dec_rec_cell = nn.LSTMCell(self.n_neurons, self.n_neurons)

        # 初期入力（GO ベクトル）
        # TF版は tf.get_variable('GO',[1,H]) を tile していたので、学習可能パラメータとして持つ
        self.go = nn.Parameter(torch.zeros(1, self.n_neurons))

        # Pointing 機構のパラメータ
        # TF版:
        #   enc_term = conv1d(enc_outputs, W_ref)
        #   dec_term = matmul(dec_output, W_out)
        #   scores   = sum(v * tanh(enc_term + dec_term))
        #
        # PyTorch版では、等価に「線形変換 + ブロードキャスト加算」で実装する
        self.W_ref = nn.Linear(self.n_neurons, self.n_neurons, bias=False)  # enc側
        self.W_out = nn.Linear(self.n_neurons, self.n_neurons, bias=False)  # dec側
        self.v = nn.Parameter(torch.zeros(self.n_neurons))

    def set_seed(self, seed):
        self.seed = seed

    def _init_mask(self, batch_size, device):
        """
        既訪問地点マスクを初期化する。
        mask: [B, T] で、訪問済みなら 1.0、未訪問なら 0.0
        """
        return torch.zeros(batch_size, self.seq_length, device=device)

    def _pointing(self, enc_outputs, dec_output, mask):
        """
        Pointing 機構:
          - enc_outputs（各地点の埋め込み）と dec_output（query）から、
            各地点の選択 logits（スコア）を作る。

        enc_outputs: [B, T, H]
        dec_output : [B, H]
        mask      : [B, T]

        return:
          masked_scores: [B, T]
        """
        # Encoder 項: [B, T, H]
        enc_term = self.W_ref(enc_outputs)

        # Decoder 項: [B, 1, H] にして足し合わせ
        dec_term = self.W_out(dec_output).unsqueeze(1)

        # v * tanh(...) を H で内積してスカラー化: [B, T]
        # v は [H] なのでブロードキャストされる
        scores = torch.sum(self.v * torch.tanh(enc_term + dec_term), dim=-1)

        # 既訪問地点のスコアを大きく下げる（実質 -infty）
        masked_scores = scores - self.infty * mask
        return masked_scores

    def forward(self, enc_outputs, enc_state):
        """
        tour を生成し、その log_prob（対数尤度）を返す。

        返す log_prob は、各 step の logπ を足し合わせたもの:
          log_prob = Σ_t log π(a_t | s_t)

        enc_outputs: [B, T, H] ただし T==seq_length を想定
        enc_state:   (h, c) それぞれ [B, H]
        """
        device = enc_outputs.device
        B, T, H = enc_outputs.shape
        assert T == self.seq_length, "enc_outputs の seq_length と decoder の seq_length が不一致です"

        # Decoder の入力系列を output_list として参照できるようにしておく
        # output_list[t] = enc_outputs[:, t, :] を取りたい用途（TF版の tf.gather 相当）
        output_list = enc_outputs  # [B, T, H]

        # マスク初期化
        mask = self._init_mask(batch_size=B, device=device)

        # 初期入力 GO を batch に拡張: [B, H]
        dec_input = self.go.repeat(B, 1)

        # 初期状態は Encoder の最終状態を引き継ぐ（seq2seq の基本形）
        h, c = enc_state

        locations = []
        log_probs = []

        # 逐次デコード（地点選択）
        for _ in range(self.seq_length):
            # LSTMCell 更新
            h, c = self.dec_rec_cell(dec_input, (h, c))

            # Pointing logits（マスク込み）
            logits = self._pointing(enc_outputs, h, mask)  # [B, T]

            # Categorical からサンプル（地点インデックス）
            # TF版の Categorical(logits=masked_scores) と同じ
            dist = torch.distributions.Categorical(logits=logits)

            # サンプリング（seed を厳密に合わせたいなら generator を渡す設計が必要）
            loc = dist.sample()  # [B]
            locations.append(loc)

            # その選択の log_prob
            logp = dist.log_prob(loc)  # [B]
            log_probs.append(logp)

            # マスク更新（選んだ地点を訪問済みにする）
            mask = mask + F.one_hot(loc, num_classes=self.seq_length).float()

            # 次入力の更新:
            # TF版は input = gather(output_list, location)[0] だったが、
            # PyTorch ではバッチごとに異なる index を取る必要があるため gather を使う
            # output_list: [B, T, H]
            # loc: [B] -> [B, 1, 1] にして gather
            idx = loc.view(B, 1, 1).expand(B, 1, H)
            dec_input = output_list.gather(dim=1, index=idx).squeeze(1)  # [B, H]

        # 初期地点を末尾に追加して閉路にする（TF版と同じ）
        first_location = locations[0]
        locations.append(first_location)

        # tour: [B, T+1]
        tour = torch.stack(locations, dim=1)

        # log_prob: [B]（各stepのlogpを合計）
        log_prob = torch.stack(log_probs, dim=0).sum(dim=0)

        return log_prob, tour


class CriticDecoder(nn.Module):
    """
    CriticDecoder（PyTorch版）
    Encoder の出力（系列埋め込み）を attention（glimpsing）で集約し、baseline（価値）を出力する Critic。

    元コード（TF版）の処理:
      - frame = enc_state[0]（LSTMのh）を query として attention を計算
      - enc_outputs に対して glimpsing（注意重み付き和）を取って固定長ベクトルへ
      - そのベクトルを FC に通して baseline を出す

    ここで baseline は tour の評価値（例えば期待報酬の推定）として使われ、
    Actor の advantage を作るために使うことが多い。

    出力:
      baseline: [B, 1]
    """

    def __init__(self, n_neurons=128, batch_size=4, seq_length=10):
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.batch_size = int(batch_size)
        self.seq_length = int(seq_length)

        # glimpsing のパラメータ（TF版の W_ref_g, W_q_g, v_g に対応）
        self.W_ref_g = nn.Linear(self.n_neurons, self.n_neurons, bias=False)
        self.W_q_g = nn.Linear(self.n_neurons, self.n_neurons, bias=False)
        self.v_g = nn.Parameter(torch.zeros(self.n_neurons))

        # 最終の回帰ヘッド（TF版: Dense(relu) -> Dense(linear)）
        self.hidden = nn.Linear(self.n_neurons, self.n_neurons)
        self.out = nn.Linear(self.n_neurons, 1)

    def forward(self, enc_outputs, enc_state):
        """
        enc_outputs: [B, T, H]
        enc_state:   (h, c) それぞれ [B, H]

        return:
          baseline: [B, 1]
        """
        h, c = enc_state

        # query（TF版では frame = enc_state[0]）
        frame = h  # [B, H]

        # attention の logits を計算
        # enc_ref_g: [B, T, H]
        enc_ref_g = self.W_ref_g(enc_outputs)

        # enc_q_g: [B, 1, H]
        enc_q_g = self.W_q_g(frame).unsqueeze(1)

        # scores_g: [B, T]
        scores_g = torch.sum(self.v_g * torch.tanh(enc_ref_g + enc_q_g), dim=-1)

        # attention 重み: [B, T]
        attention_g = F.softmax(scores_g, dim=-1)

        # glimpse（注意重み付き和）:
        # enc_outputs * attention を掛けて T で合計 -> [B, H]
        glimpse = enc_outputs * attention_g.unsqueeze(-1)
        glimpse = glimpse.sum(dim=1)

        # FC で baseline を出す
        hidden = F.relu(self.hidden(glimpse))
        baseline = self.out(hidden)  # [B, 1]
        return baseline