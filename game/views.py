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

# ==========================================
# 2. AIモデルのロード（サーバー起動時に1回だけ実行）
# ==========================================
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'othello_model.pth')
device = torch.device("cpu")
ai_model = OthelloNet()

if os.path.exists(MODEL_PATH):
    ai_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    ai_model.eval()  # 推論モード（学習ストップ）に切り替え
    print("✅ 学習済みAIモデルのロードに成功しました！")
else:
    print("⚠️ AIモデルが見つかりません。")

# ==========================================
# 3. AI推論ロジック（ルールエンジン連携）
# ==========================================
def predict_best_move(board_array, current_player):
    """
    盤面の2次元配列を受け取り、合法手の中からAIが最も良いと判断した手を返す
    """
    # フェーズ1で作ったルールエンジンに現在の状態をセット
    game = OthelloBoard()
    game.board = np.array(board_array)
    game.current_player = current_player
    
    # 合法手（打てる場所）のリストを取得
    legal_moves = game.get_legal_moves(current_player)

    # 万が一打てる場所がない場合（パス）
    if not legal_moves:
        return None, None

    # 盤面を正規化してAIに入力 (自分:1, 相手:-1)
    state = np.array(board_array) * current_player
    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 1, 8, 8)
    
    with torch.no_grad(): # 推論時はメモリ節約のため勾配計算をオフ
        output = ai_model(state_tensor).squeeze() # 64マスのスコア（1次元配列）
        
    # 合法手の中から、AIのスコアが最も高いマスを選ぶ
    best_move = None
    best_score = -float('inf')
    
    for r, c in legal_moves:
        action_index = r * 8 + c
        score = output[action_index].item()
        
        if score > best_score:
            best_score = score
            best_move = (r, c)
            
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