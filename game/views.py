import os
import sys
import json
import torch
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# ==========================================
# 1. パスの設定とモジュールのインポート
# ==========================================
# プロジェクトのルートディレクトリ（coreやmlがある階層）をパスに追加
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.board import OthelloBoard
from ml.model import OthelloNet
from ml.mcts import MCTS

# ==========================================
# 2. AIモデルのロード（サーバー起動時に1回だけ実行）
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'othello_model.pth')
device = torch.device("cpu")
ai_model = None  # 最初は空にしておく

def get_ai_model():
    """
    AIが必要になった瞬間に初めてロードする（すでにロード済みならそのまま返す）
    """
    global ai_model
    if ai_model is None:
        print("⏳ AIモデルを初めてメモリに読み込んでいます...")
        ai_model = OthelloNet()
        if os.path.exists(MODEL_PATH):
            ai_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            ai_model.eval()
            print("✅ AIモデルのロード完了！")
        else:
            print("⚠️ AIモデルが見つかりません。")
    return ai_model

# ==========================================
# 3. AI推論ロジック（ルールエンジン連携）
# ==========================================
def predict_best_move(board_array, current_player):
    # ここで初めてAIを呼び出す
    model = get_ai_model()
    
    game = OthelloBoard()
    game.board = np.array(board_array)
    game.current_player = current_player
    legal_moves = game.get_legal_moves(current_player)

    if not legal_moves: return None, None
    if len(legal_moves) == 1: return legal_moves[0][0], legal_moves[0][1]

    # MCTSによるシミュレーション
    mcts = MCTS(model, num_simulations=10) # Render無料枠に合わせて回数を調整
    best_move = mcts.search(board_array, current_player)
    
    return best_move[0], best_move[1]

# ==========================================
# 4. フロントエンド用ビュー（HTML表示）
# ==========================================
def index(request):
    """
    トップページ（http://localhost:8000/）にアクセスした際にオセロ画面を返す
    """
    return render(request, 'game/index.html')

# ==========================================
# 5. フロントエンド通信用API
# ==========================================
@csrf_exempt
def get_ai_move(request):
    """
    JavaScriptから盤面データを受け取り、AIの次の手を返すAPI
    """
    if request.method == 'POST':
        try:
            # JavaScriptから送信されたJSONデータを読み込む
            data = json.loads(request.body)
            board_array = data.get('board')
            current_player = data.get('current_player')

            # AIに推論させる
            row, col = predict_best_move(board_array, current_player)

            # パスの場合の処理
            if row is None or col is None:
                return JsonResponse({
                    'status': 'pass',
                    'message': '打てる場所がありません'
                })

            # 正常に手が決まった場合
            return JsonResponse({
                'status': 'success',
                'row': row,
                'col': col
            })
            
        except Exception as e:
            # エラーが発生した場合
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
            
    return JsonResponse({'status': 'error', 'message': 'POSTメソッドのみ許可されています'}, status=405)