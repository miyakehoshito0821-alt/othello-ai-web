import os
import sys
import struct
import numpy as np

# プロジェクトのルートディレクトリをパスに追加して core.board を読み込めるようにする
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.board import OthelloBoard

def parse_wthor_file(filepath):
    boards = []
    actions = []
    
    with open(filepath, 'rb') as f:
        header = f.read(16)
        if len(header) < 16:
            return boards, actions
            
        # WThorのヘッダー仕様: 4〜7バイト目に試合数が格納されている
        num_games = struct.unpack('<I', header[4:8])[0]
        print(f"[{os.path.basename(filepath)}] {num_games} 試合のデータを読み込みます...")
        
        valid_games = 0
        for _ in range(num_games):
            game_data = f.read(68)
            if len(game_data) < 68:
                break
                
            # 最初の8バイトは大会や選手IDなど。9バイト目以降が60手分の棋譜
            moves = game_data[8:68]
            game = OthelloBoard()
            
            is_valid_game = True
            for move_byte in moves:
                if move_byte == 0:
                    break # 終局
                    
                # WThor形式の変換: 例えば 'f5' は 65。 10の位が列(a-h -> 1-8)、1の位が行(1-8)
                col = (move_byte // 10) - 1
                row = (move_byte % 10) - 1
                
                # 【シニアの工夫】モデル学習用に現在の盤面を保存
                # AIが「黒でも白でも」同じように盤面を評価できるよう、
                # 現在のプレイヤーの石を常に 1、相手を -1 に正規化します。
                state = game.board * game.current_player
                action_index = row * 8 + col
                
                boards.append(state.copy())
                actions.append(action_index)
                
                # 石を置く
                try:
                    game.make_move(row, col)
                except ValueError:
                    # 万が一不正な手があった場合はこの試合をスキップ
                    is_valid_game = False
                    break
                
                # パス処理 (GUIと同じロジック)
                while not game.is_game_over() and len(game.get_legal_moves(game.current_player)) == 0:
                    game.current_player = -game.current_player
            
            if is_valid_game:
                valid_games += 1
                
    print(f"抽出完了: {valid_games} 試合から {len(boards)} 手分のデータを取得しました。")
    return np.array(boards), np.array(actions)

if __name__ == "__main__":
    wthor_file = "WTH_2024.wtb"
    output_file = "othello_dataset.npz"
    
    if not os.path.exists(wthor_file):
        print(f"エラー: {wthor_file} が見つかりません。ダウンロードして同じフォルダに配置してください。")
        sys.exit(1)
        
    X, y = parse_wthor_file(wthor_file)
    
    # Numpy形式（圧縮付き）で保存
    np.savez_compressed(output_file, boards=X, actions=y)
    print(f"\nデータセットを {output_file} に保存しました！")
    print(f"入力(X)の形状: {X.shape}")
    print(f"正解(y)の形状: {y.shape}")