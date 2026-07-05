import numpy as np

class OthelloBoard:
    """
    オセロの盤面とゲームロジックを管理するクラス
    黒石: 1, 白石: -1, 空きマス: 0 として扱う
    """
    
    # 探索する8方向 (行の増分, 列の増分)
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.current_player = 1  # 黒(1)が先手
        self._initialize_board()

    def _initialize_board(self):
        """初期盤面の配置"""
        self.board[3, 3] = -1
        self.board[4, 4] = -1
        self.board[3, 4] = 1
        self.board[4, 3] = 1

    def is_on_board(self, row, col):
        """指定された座標が盤面上にあるか判定"""
        return 0 <= row < 8 and 0 <= col < 8

    def get_flipped_stones(self, row, col, player):
        """
        指定した位置に石を置いた場合、裏返る石のリストを返す
        戻り値: 裏返る石の座標のリスト [(r1, c1), (r2, c2), ...]
        """
        if not self.is_on_board(row, col) or self.board[row, col] != 0:
            return []

        flipped_stones = []
        opponent = -player

        for dr, dc in self.DIRECTIONS:
            r, c = row + dr, col + dc
            temp_flipped = []
            
            # 相手の石が続く限り進む
            while self.is_on_board(r, c) and self.board[r, c] == opponent:
                temp_flipped.append((r, c))
                r += dr
                c += dc
                
            # その先に自分の石があれば、挟んでいるので裏返せる
            if temp_flipped and self.is_on_board(r, c) and self.board[r, c] == player:
                flipped_stones.extend(temp_flipped)

        return flipped_stones

    def get_legal_moves(self, player):
        """現在の手番のプレイヤーが打てる合法手のリストを返す"""
        legal_moves = []
        for r in range(8):
            for c in range(8):
                if self.get_flipped_stones(r, c, player):
                    legal_moves.append((r, c))
        return legal_moves

    def make_move(self, row, col):
        """石を置き、反転処理を行い、手番を交代する"""
        flipped_stones = self.get_flipped_stones(row, col, self.current_player)
        if not flipped_stones:
            raise ValueError(f"Invalid move at ({row}, {col})")

        # 石を置く
        self.board[row, col] = self.current_player
        
        # 石を裏返す
        for r, c in flipped_stones:
            self.board[r, c] = self.current_player

        # 手番を交代する
        self.current_player = -self.current_player

    def pass_turn(self):
        """パス（スキップ）処理：強制的に手番を交代する"""
        self.current_player = -self.current_player

        
    def is_game_over(self):
        """ゲーム終了判定（両者ともに打つ手がない場合）"""
        return len(self.get_legal_moves(1)) == 0 and len(self.get_legal_moves(-1)) == 0

    def get_winner(self):
        """勝者を判定する (1: 黒勝利, -1: 白勝利, 0: 引き分け)"""
        black_score = np.sum(self.board == 1)
        white_score = np.sum(self.board == -1)
        
        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return -1
        else:
            return 0