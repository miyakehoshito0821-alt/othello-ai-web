import os
import sys
import json
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.board import OthelloBoard
from ml.model import NumpyOthelloNet
from ml.mcts import MCTS

# NumPy版AIの初期化（超軽量なので遅延読み込み不要、即座に起動します）
WEIGHTS_PATH = os.path.join(BASE_DIR, 'ml', 'othello_weights.npz')
ai_model = NumpyOthelloNet(WEIGHTS_PATH)

def predict_best_move(board_array, current_player):
    """NumPy版AIとMCTSを使って最善手を計算する"""
    game = OthelloBoard()
    game.board = np.array(board_array)
    game.current_player = current_player
    legal_moves = game.get_legal_moves(current_player)

    if not legal_moves: return None, None
    if len(legal_moves) == 1: return legal_moves[0][0], legal_moves[0][1]

    # MCTSによるシミュレーション
    # ※応答速度を優先し、検討回数を少なめに設定しています。
    #    50回が速すぎず強さも安定する目安ですが、必要ならさらに調整可能です。
    mcts = MCTS(ai_model, num_simulations=25)
    best_move = mcts.search(board_array, current_player)
    
    return best_move[0], best_move[1]

def index(request):
    return render(request, 'game/index.html')

@csrf_exempt
def get_ai_move(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            board_array = data.get('board')
            current_player = data.get('current_player')

            row, col = predict_best_move(board_array, current_player)

            if row is None or col is None:
                return JsonResponse({'status': 'pass', 'message': '打てる場所がありません'})

            return JsonResponse({'status': 'success', 'row': row, 'col': col})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
            
    return JsonResponse({'status': 'error', 'message': 'POSTのみ許可'}, status=405)