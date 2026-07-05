import torch
import numpy as np

# 保存したPyTorchモデルを読み込む
state_dict = torch.load('ml/othello_model.pth', map_location=torch.device('cpu'))

# NumPyの配列として辞書型に保存し直す
weights_dict = {}
for key, tensor in state_dict.items():
    weights_dict[key] = tensor.numpy()

# ml/othello_weights.npz として保存
np.savez('ml/othello_weights.npz', **weights_dict)
print("🎉 PyTorchの重みをNumPy形式（ml/othello_weights.npz）に変換しました！")