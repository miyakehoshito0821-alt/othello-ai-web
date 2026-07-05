import tkinter as tk
from tkinter import messagebox
from core.board import OthelloBoard

class OthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello - Phase 1 GUI Debugger")
        
        # ゲームロジックの初期化
        self.game = OthelloBoard()
        
        # マス目の大きさ（ピクセル）
        self.cell_size = 60
        
        # メインキャンバスの作成
        self.canvas = tk.Canvas(
            root, 
            width=self.cell_size * 8, 
            height=self.cell_size * 8, 
            bg="green"
        )
        self.canvas.pack()
        
        # ステータスラベル（現在の手番などを表示）
        self.status_label = tk.Label(root, text="手番: 黒 (プレイヤー)", font=("Arial", 14))
        self.status_label.pack(pady=10)
        
        # クリックイベントのバインド
        self.canvas.bind("<Button-1>", self.on_click)
        
        # 初回描画
        self.draw_board()

    def draw_board(self):
        """盤面と石を再描画する"""
        self.canvas.delete("all")
        
        # 合法手（次に打てる場所）のリストを取得
        legal_moves = self.game.get_legal_moves(self.game.current_player)
        
        for r in range(8):
            for c in range(8):
                # マス目の枠線を描画
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")
                
                # 石の描画
                stone = self.game.board[r, c]
                if stone == 1:  # 黒
                    self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill="black")
                elif stone == -1:  # 白
                    self.canvas.create_oval(x1+5, y1+5, x2-5, y2-5, fill="white")
                
                # デバッグ用：合法手がある場所に薄い円を表示
                if (r, c) in legal_moves:
                    self.canvas.create_oval(x1+20, y1+20, x2-20, y2-20, fill="lightgray", outline="", state="disabled")

    def on_click(self, event):
        """盤面がクリックされた時の処理"""
        # クリックされた座標から行・列を計算
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        # 合法手でなければ無視
        legal_moves = self.game.get_legal_moves(self.game.current_player)
        if (row, col) not in legal_moves:
            return

        # 石を置く
        self.game.make_move(row, col)
        
        # パス判定のループ（両者打てなくなるまで）
        while not self.game.is_game_over():
            next_player_moves = self.game.get_legal_moves(self.game.current_player)
            if len(next_player_moves) > 0:
                # 次のプレイヤーに打てる手がある場合は通常通り交代
                break
            else:
                # 打てる手がない場合はパスし、手番を元に戻す
                player_name = "黒" if self.game.current_player == 1 else "白"
                messagebox.showinfo("パス", f"{player_name}に打てる場所がないためパスします。")
                self.game.current_player = -self.game.current_player
        
        # 画面の更新
        self.draw_board()
        self.update_status()
        
        # ゲーム終了判定
        if self.game.is_game_over():
            self.handle_game_over()

    def update_status(self):
        """手番表示の更新"""
        player_name = "黒" if self.game.current_player == 1 else "白"
        self.status_label.config(text=f"手番: {player_name}")

    def handle_game_over(self):
        """ゲーム終了時の勝敗表示"""
        winner = self.game.get_winner()
        black_count = (self.game.board == 1).sum()
        white_count = (self.game.board == -1).sum()
        
        result_msg = f"黒: {black_count}石\n白: {white_count}石\n\n"
        if winner == 1:
            result_msg += "黒の勝ちです！"
        elif winner == -1:
            result_msg += "白の勝ちです！"
        else:
            result_msg += "引き分けです！"
            
        messagebox.showinfo("ゲーム終了", result_msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = OthelloGUI(root)
    root.mainloop()