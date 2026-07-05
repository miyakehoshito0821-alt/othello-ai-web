import urllib.request
import json

# DjangoサーバーのAPIのURL
url = 'http://127.0.0.1:8000/api/play/'

# オセロの初期盤面を作成（0: 空き, 1: 黒, -1: 白）
board = [[0] * 8 for _ in range(8)]
board[3][3] = -1
board[4][4] = -1
board[3][4] = 1
board[4][3] = 1

# 送信するデータ（AIは白番だと仮定してリクエストを送る）
payload = {
    'board': board,
    'current_player': -1
}

# リクエストの作成と送信
req = urllib.request.Request(
    url, 
    data=json.dumps(payload).encode('utf-8'), 
    headers={'Content-Type': 'application/json'}
)

print("AIに盤面を送信中...")

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print("✅ 通信成功！AIの回答:")
        print(f"  打つ場所: 行 {result['row']}, 列 {result['col']}")
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")