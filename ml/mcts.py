import os
import sys
import math
import torch
import numpy as np

# プロジェクトのルートをパスに追加して core.board を読み込めるようにする
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.board import OthelloBoard

class Node:
    """MCTSの探索ツリーを構成するノード（盤面状態）"""
    def __init__(self, state, current_player, parent=None, prior_prob=0.0):
        self.state = state
        self.current_player = current_player
        self.parent = parent
        self.children = {} # {アクション: Node}
        self.visit_count = 0
        self.value_sum = 0
        self.prior_prob = prior_prob

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, num_simulations=50):
        self.model = model
        self.num_simulations = num_simulations # 脳内で何回シミュレーションするか（多いほど強く、遅くなる）
        self.c_puct = 1.0 # 探索(未知の手)と知識(直感)のバランス

    def search(self, initial_state, current_player):
        """シミュレーションを実行し、最適な手を返す"""
        root = Node(initial_state, current_player)

        for _ in range(self.num_simulations):
            node = root
            game = OthelloBoard()
            game.board = np.copy(node.state)
            game.current_player = node.current_player

            # 1. Selection (選択): ツリーを末端まで降りていく
            while node.is_expanded():
                action, node = self.select_child(node)
                game.make_move(action[0], action[1])
                # パス処理
                while not game.is_game_over() and not game.get_legal_moves(game.current_player):
                    game.current_player = -game.current_player

            # 2. Expansion & Evaluation (展開と評価)
            legal_moves = game.get_legal_moves(game.current_player)
            if not game.is_game_over():
                # CNNで「直感（各手の確率）」を取得し、子ノードを展開
                policy = self.get_policy(game.board, game.current_player, legal_moves)
                for action, prob in policy.items():
                    node.children[action] = Node(game.board, game.current_player, parent=node, prior_prob=prob)
                
                # 簡易的な価値評価（石の数の差）
                value = self.evaluate_board(game)
            else:
                value = self.evaluate_board(game) # 終局時

            # 3. Backpropagation (伝播): 評価結果をルートまで遡って反映
            self.backpropagate(node, value, game.current_player)

        # 脳内シミュレーションで最も多く訪問（検討）した手を選ぶ
        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_action = action

        return best_action

    def select_child(self, node):
        """PUCTアルゴリズム（AlphaZero方式）で次に調べるノードを選択"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            q_value = child.value()
            u_value = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_policy(self, board, player, legal_moves):
        """CNNモデルから各手の確率（Softmax）を取得"""
        state = np.array(board) * player
        state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 1, 8, 8)
        
        with torch.no_grad():
            output = self.model(state_tensor).squeeze()
        
        policy = {}
        scores = []
        for r, c in legal_moves:
            idx = r * 8 + c
            scores.append(output[idx].item())
        
        # Softmax関数で確率に変換
        scores = np.array(scores)
        scores = np.exp(scores - np.max(scores))
        probs = scores / np.sum(scores)
        
        for i, action in enumerate(legal_moves):
            policy[action] = probs[i]
            
        return policy

    def evaluate_board(self, game):
        """盤面の有利不利を評価（-1.0 〜 1.0）"""
        p1_stones = np.sum(game.board == game.current_player)
        p2_stones = np.sum(game.board == -game.current_player)
        if p1_stones + p2_stones == 0: return 0
        return (p1_stones - p2_stones) / (p1_stones + p2_stones)

    def backpropagate(self, node, value, leaf_player):
        """シミュレーション結果の反映"""
        while node is not None:
            reward = value if node.current_player == leaf_player else -value
            node.visit_count += 1
            node.value_sum += reward
            node = node.parent