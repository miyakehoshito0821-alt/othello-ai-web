import numpy as np

class NumpyOthelloNet:
    """
    PyTorchを使わず、NumPyだけでCNNの推論処理を再現した超軽量クラス
    """
    def __init__(self, weights_path):
        # 変換した重みデータを読み込む
        weights = np.load(weights_path)
        
        self.conv1_w = weights['conv1.weight'] # (64, 1, 3, 3)
        self.conv1_b = weights['conv1.bias']   # (64,)
        self.conv2_w = weights['conv2.weight'] # (128, 64, 3, 3)
        self.conv2_b = weights['conv2.bias']   # (128,)
        
        # 全結合層の重み
        self.fc1_w = weights['fc1.weight']     # (256, 8192)
        self.fc1_b = weights['fc1.bias']       # (256,)
        self.fc2_w = weights['fc2.weight']     # (64, 256)
        self.fc2_b = weights['fc2.bias']       # (64,)

    def relu(self, x):
        return np.maximum(0, x)

    def conv2d(self, x, w, b):
        """簡易版2D畳み込み（パディング=1、ストライド=1専用）"""
        out_channels, in_channels, kh, kw = w.shape
        _, _, h, w_in = x.shape
        
        # パディング処理（上下左右に0を1マス足す）
        padded = np.pad(x, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
        out = np.zeros((1, out_channels, h, w_in))
        
        for oc in range(out_channels):
            for r in range(h):
                for c in range(w_in):
                    # 3x3の領域を切り取って掛け算の総和を計算
                    patch = padded[0, :, r:r+3, c:c+3]
                    out[0, oc, r, c] = np.sum(patch * w[oc]) + b[oc]
        return out

    def forward(self, x):
        """盤面データ(1, 1, 8, 8)を入力して64マスのスコアを返す"""
        # 第1畳み込み層 + ReLU
        x = self.conv2d(x, self.conv1_w, self.conv1_b)
        x = self.relu(x)
        
        # 第2畳み込み層 + ReLU
        x = self.conv2d(x, self.conv2_w, self.conv2_b)
        x = self.relu(x)
        
        # フラット化（1次元にする）
        x = x.reshape(1, -1)
        
        # 全結合層1 + ReLU
        x = np.dot(x, self.fc1_w.T) + self.fc1_b
        x = self.relu(x)
        
        # 全結合層2（最終スコア）
        x = np.dot(x, self.fc2_w.T) + self.fc2_b
        return x.flatten()