import os
import torch
import numpy as np
import time

# プロジェクトのモジュールを読み込む
from core.board import OthelloBoard
from ml.model import OthelloNet
from ml.mcts import MCTS

# ==========================================
# 1. 準備（モデルのロード）
# ==========================================
print("AIのモデルを準備しています...")
device = torch.device("cpu")
ai_model = OthelloNet()
ai_model.load_state_dict(torch.load('ml/othello_model.pth', map_location=device))
ai_model.eval()

# MCTSの準備（今回はテストを早く終わらせるためシミュレーション回数を50回に設定）
mcts = MCTS(ai_model, num_simulations=50)

# ==========================================
# 2. 各AIの思考ロジック
# ==========================================
def get_move_old_ai(game):
    """旧型AI: CNNの直感だけで一番スコアが高い手を選ぶ"""
    legal_moves = game.get_legal_moves(game.current_player)
    if not legal_moves: return None
    
    state = np.copy(game.board) * game.current_player
    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 1, 8, 8)
    
    with torch.no_grad():
        output = ai_model(state_tensor).squeeze()
        
    best_move = None
    best_score = -float('inf')
    for r, c in legal_moves:
        score = output[r * 8 + c].item()
        if score > best_score:
            best_score = score
            best_move = (r, c)
    return best_move

def get_move_new_ai(game):
    """新型AI: MCTSを使って数手先まで読む"""
    legal_moves = game.get_legal_moves(game.current_player)
    if not legal_moves: return None
    if len(legal_moves) == 1: return legal_moves[0]
    
    return mcts.search(game.board, game.current_player)

# ==========================================
# 3. 試合の進行ロジック
# ==========================================
def play_game(black_ai, white_ai):
    game = OthelloBoard()
    
    while not game.is_game_over():
        legal_moves = game.get_legal_moves(game.current_player)
        
        if not legal_moves:
            game.current_player = -game.current_player
            continue
            
        if game.current_player == 1:
            move = black_ai(game)
        else:
            move = white_ai(game)
            
        if move:
            game.make_move(move[0], move[1])
            
    # 勝敗判定
    black_stones = np.sum(game.board == 1)
    white_stones = np.sum(game.board == -1)
    
    if black_stones > white_stones: return 1  # 黒の勝ち
    elif white_stones > black_stones: return -1 # 白の勝ち
    else: return 0 # 引き分け

# ==========================================
# 4. 100番勝負の実行
# ==========================================
if __name__ == "__main__":
    TOTAL_GAMES = 100
    HALF_GAMES = TOTAL_GAMES // 2
    
    new_ai_wins = 0
    old_ai_wins = 0
    draws = 0
    
    print(f"\n🚀 {TOTAL_GAMES}番勝負を開始します！ (画面描画なしの高速モード)")
    start_time = time.time()
    
    # 前半50戦：新型AIが「先手（黒）」
    print("\n--- 前半戦（新型AI: 黒, 旧型AI: 白） ---")
    for i in range(HALF_GAMES):
        result = play_game(black_ai=get_move_new_ai, white_ai=get_move_old_ai)
        if result == 1: new_ai_wins += 1
        elif result == -1: old_ai_wins += 1
        else: draws += 1
        if (i+1) % 10 == 0: print(f"{i+1}試合完了...")

    # 後半50戦：新型AIが「後手（白）」
    print("\n--- 後半戦（旧型AI: 黒, 新型AI: 白） ---")
    for i in range(HALF_GAMES):
        result = play_game(black_ai=get_move_old_ai, white_ai=get_move_new_ai)
        if result == -1: new_ai_wins += 1 # この試合は白が新型AIなので、-1なら新型の勝ち
        elif result == 1: old_ai_wins += 1
        else: draws += 1
        if (i+1) % 10 == 0: print(f"{i+1}試合完了...")

    # ==========================================
    # 5. 結果発表
    # ==========================================
    elapsed_time = time.time() - start_time
    win_rate = (new_ai_wins / TOTAL_GAMES) * 100

    print("\n==========================================")
    print("🏆 100番勝負 最終結果 🏆")
    print("==========================================")
    print(f"新型AI (MCTS) の勝ち: {new_ai_wins} 試合")
    print(f"旧型AI (直感) の勝ち: {old_ai_wins} 試合")
    print(f"引き分け           : {draws} 試合")
    print(f"\n📊 新型AIの勝率: {win_rate:.1f} %")
    print(f"⏱️ 実行時間: {elapsed_time:.1f} 秒")
    print("==========================================")
    
    if win_rate >= 70:
        print("大成功！MCTSの導入によりAIは圧倒的に進化しました！")
    elif win_rate >= 55:
        print("成功！明確な強さの向上が見られます。")
    else:
        print("あれ？もう少しMCTSの評価関数やパラメータを調整する必要がありそうです。")