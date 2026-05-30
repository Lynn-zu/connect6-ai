# app.py - 修复版：对接训练生成的模型文件
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import numpy as np
import random
from typing import Tuple, List, Dict, Optional
import time
import os
import subprocess
import sys
import threading


# 检查TensorFlow可用性
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} 可用")
except ImportError as e:
    print(f"TensorFlow不可用: {e}")
    TF_AVAILABLE = False

# ===================== 核心配置（与训练脚本完全对齐） =====================
# 复制训练脚本的核心配置，保证一致性
BOARD_SIZE = 15
INPUT_CHANNELS = 17  # 输入特征通道数
RES_BLOCKS = 6  # 残差块数量（与训练一致）
FILTERS = 64  # 卷积核数量（与训练一致）
L2_REG = 1e-4  # L2正则化系数（与训练一致）
LEARNING_RATE =1e-3   # 学习率（与训练一致）

# 模型路径（与训练脚本完全一致）
MODEL_DIR = "models"
MODEL_NAME = "connect6.model.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
OLD_MODEL_PATH = os.path.join(MODEL_DIR, "connect6_mcts_model.h5")

# 创建模型目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('data/games', exist_ok=True)

app = Flask(__name__, template_folder='templates')
CORS(app)

model_cache_lock = threading.Lock()
cached_model = None
cached_model_error = ''
cached_model_mtime = None


def get_cached_model():
    global cached_model, cached_model_error, cached_model_mtime
    if not TF_AVAILABLE:
        cached_model_error = 'TensorFlow不可用'
        return None
    if not os.path.exists(MODEL_PATH):
        cached_model = None
        cached_model_mtime = None
        cached_model_error = f'模型文件不存在: {MODEL_PATH}'
        return None
    current_mtime = os.path.getmtime(MODEL_PATH)
    with model_cache_lock:
        if cached_model is not None and cached_model_mtime == current_mtime:
            return cached_model
        try:
            from tensorflow.keras.models import load_model
            cached_model = load_model(MODEL_PATH, compile=False)
            cached_model_mtime = current_mtime
            cached_model_error = ''
            return cached_model
        except Exception as e:
            cached_model = None
            cached_model_mtime = None
            cached_model_error = str(e)
            return None

class PolicyValueNet:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        inputs = keras.Input((BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS))
        x = layers.Conv2D(
            FILTERS, 3, padding="same",
            kernel_regularizer=regularizers.l2(L2_REG)
        )(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        for _ in range(RES_BLOCKS):
            x = self.res_block(x)

        # policy head
        p = layers.Conv2D(2, 1)(x)
        p = layers.Flatten()(p)
        policy = layers.Dense(
            BOARD_SIZE * BOARD_SIZE,
            activation="softmax",
            name="policy",
            dtype="float32"
        )(p)

        # value head
        v = layers.Conv2D(1, 1)(x)
        v = layers.Flatten()(v)
        v = layers.Dense(128, activation="relu")(v)
        value = layers.Dense(
            1, activation="tanh",
            name="value",
            dtype="float32"
        )(v)

        model = keras.Model(inputs, [policy, value])
        return model

    def predict(self, board, player):
        x = board_to_features(board, player)[None]
        p, v = self.model.predict(x, verbose=0)
        return p[0].reshape(BOARD_SIZE, BOARD_SIZE), float(v[0][0])

    def board_to_features(board, player):
        f = np.zeros((BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS), dtype=np.float32)
        f[:, :, 0] = (board == player)
        f[:, :, 1] = (board == 3 - player)
        return f


    def res_block(self, x):
        shortcut = x
        x = layers.Conv2D(FILTERS, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(FILTERS, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        return layers.ReLU()(x)


# ===================== 六子棋棋盘类 =====================
class Connect6Board:
    """六子棋棋盘类"""
    def __init__(self, size=BOARD_SIZE):  # 使用训练的棋盘大小
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # 1=黑, 2=白
        self.game_over = False
        self.winner = None
        self.last_moves = []
        self.move_history = []
        self.is_first_turn = True

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.last_moves = []
        self.move_history = []
        self.is_first_turn = True

    def is_valid_move(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        return self.board[row, col] == 0

    def make_move(self, moves: List[Tuple[int, int]]) -> bool:
        if self.game_over:
            return False

        # 六子棋规则：首手1子，之后每手2子
        expected = 1 if (self.is_first_turn and self.current_player == 1) else 2
        if len(moves) != expected:
            return False

        for r, c in moves:
            if not self.is_valid_move(r, c):
                return False

        # 执行落子
        self.move_history.append((moves.copy(), self.current_player))
        self.last_moves = moves.copy()
        for r, c in moves:
            self.board[r, c] = self.current_player

        # 检查胜利
        for r, c in moves:
            if self.check_win(r, c):
                self.game_over = True
                self.winner = self.current_player
                return True

        # 检查平局
        if np.all(self.board != 0):
            self.game_over = True
            self.winner = 0
            return True

        # 切换玩家
        self.current_player = 3 - self.current_player
        self.is_first_turn = False
        return True

    def check_win(self, row: int, col: int) -> bool:
        """检查是否连成6子"""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正向检查
            x, y = row, col
            for _ in range(5):
                x += dx; y += dy
                if 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == player:
                    count += 1
                else:
                    break
            # 反向检查
            x, y = row, col
            for _ in range(5):
                x -= dx; y -= dy
                if 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == player:
                    count += 1
                else:
                    break
            if count >= 6:
                return True
        return False

    def get_available_moves(self):
        mv = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    mv.append((i, j))
        return mv

    def get_expected_move_count(self):
        return 1 if (self.is_first_turn and self.current_player == 1) else 2

    def pop_last_move(self):
        if not self.move_history:
            return None
        last_moves, player = self.move_history.pop()
        for r, c in last_moves:
            self.board[r, c] = 0
        # 恢复状态
        self.current_player = player
        self.game_over = False
        self.winner = None
        if len(self.move_history) == 0:
            self.is_first_turn = True
        else:
            self.is_first_turn = False
        self.last_moves = self.move_history[-1][0] if self.move_history else []
        return (last_moves, player)

    def undo_move(self):
        return self.pop_last_move() is not None

    def replay_move(self, record):
        moves, player = record
        if self.game_over:
            return False
        for r, c in moves:
            if not self.is_valid_move(r, c):
                return False
        self.move_history.append((moves.copy(), player))
        self.last_moves = moves.copy()
        self.current_player = player
        for r, c in moves:
            self.board[r, c] = player
        for r, c in moves:
            if self.check_win(r, c):
                self.game_over = True
                self.winner = player
                return True
        if np.all(self.board != 0):
            self.game_over = True
            self.winner = 0
            return True
        self.current_player = 3 - player
        self.is_first_turn = False
        return True

    def to_dict(self):
        return {
            'size': self.size,
            'board': self.board.tolist(),
            'current_player': int(self.current_player),
            'game_over': bool(self.game_over),
            'winner': int(self.winner) if self.winner is not None else None,
            'last_moves': self.last_moves,
            'move_history': self.move_history,
            'is_first_turn': bool(self.is_first_turn),
            'expected_move_count': self.get_expected_move_count()
        }

# ===================== Minimax AI（保留原逻辑） =====================
class Connect6MinimaxAI:
    """Minimax AI for Connect6"""
    def __init__(self, player: int, level: str = "medium"):
        self.player = player
        self.opponent = 3 - player
        self.level = level
        self.last_search_time = 0
        self.last_candidates = []

        # 根据难度设置搜索深度
        if level == "easy":
            self.max_depth = 1
            self.candidate_limit = 5
        elif level == "hard":
            self.max_depth = 3
            self.candidate_limit = 12
        else:  # medium
            self.max_depth = 2
            self.candidate_limit = 8

    def evaluate_position(self, board_array, row: int, col: int, player: int) -> int:
        if board_array[row, col] != 0:
            return 0
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            attack_score = self.evaluate_direction(board_array, row, col, dx, dy, player)
            score += attack_score
            defense_score = self.evaluate_direction(board_array, row, col, dx, dy, 3 - player)
            score += defense_score * 0.8
        return int(score)

    def evaluate_direction(self, board_array, r, c, dx, dy, player):
        size = board_array.shape[0]
        score = 0
        temp_board = board_array.copy()
        temp_board[r, c] = player
        forward = 0
        for step in range(1, 6):
            x, y = r + dx * step, c + dy * step
            if 0 <= x < size and 0 <= y < size and temp_board[x, y] == player:
                forward += 1
            else:
                break
        backward = 0
        for step in range(1, 6):
            x, y = r - dx * step, c - dy * step
            if 0 <= x < size and 0 <= y < size and temp_board[x, y] == player:
                backward += 1
            else:
                break
        total = forward + backward + 1
        if total >= 6:
            score += 10000
        elif total == 5:
            score += 5000
        elif total == 4:
            score += 1000
        elif total == 3:
            score += 300
        elif total == 2:
            score += 100
        elif total == 1:
            score += 10
        return score

    def get_candidate_moves(self, board: Connect6Board, player: Optional[int] = None):
        player = player or board.current_player
        board_array = np.array(board.board)
        valid_moves = board.get_available_moves()
        if not valid_moves:
            self.last_decision = 'model_error'
            return []
        scored_moves = []
        for r, c in valid_moves:
            attack = self.evaluate_position(board_array, r, c, player)
            defense = self.evaluate_position(board_array, r, c, 3 - player)
            center = board.size // 2
            center_bonus = max(0, board.size - (abs(r - center) + abs(c - center))) * 0.5
            score = attack + defense * 0.9 + center_bonus
            scored_moves.append((score, (r, c)))
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        candidates = [m for _, m in scored_moves[:self.candidate_limit]]
        self.last_candidates = candidates.copy()
        return candidates

    def minimax(self, board: Connect6Board, depth: int, alpha: float, beta: float, maximizing: bool):
        if depth == 0 or board.game_over:
            return self.evaluate_board(board)
        expected = board.get_expected_move_count()
        current_player = board.current_player
        candidates = self.get_candidate_moves(board, current_player)
        if not candidates:
            return self.evaluate_board(board)

        if maximizing:
            max_eval = -float('inf')
            if expected == 1:
                for (r, c) in candidates:
                    if not board.is_valid_move(r, c):
                        continue
                    saved_state = self.save_board_state(board)
                    board.board[r, c] = current_player
                    board.move_history.append(([(r, c)], current_player))
                    if board.check_win(r, c):
                        board.game_over = True
                        board.winner = current_player
                    board.current_player = 3 - current_player
                    board.is_first_turn = False
                    eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    self.restore_board_state(board, saved_state)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
            else:
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        r1, c1 = candidates[i]
                        r2, c2 = candidates[j]
                        if not (board.is_valid_move(r1, c1) and board.is_valid_move(r2, c2)):
                            continue
                        saved_state = self.save_board_state(board)
                        moves = [(r1, c1), (r2, c2)]
                        for r, c in moves:
                            board.board[r, c] = current_player
                        board.move_history.append((moves.copy(), current_player))
                        win_found = False
                        for r, c in moves:
                            if board.check_win(r, c):
                                board.game_over = True
                                board.winner = current_player
                                win_found = True
                                break
                        board.current_player = 3 - current_player
                        board.is_first_turn = False
                        eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                        max_eval = max(max_eval, eval_score)
                        self.restore_board_state(board, saved_state)
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            if expected == 1:
                for (r, c) in candidates:
                    if not board.is_valid_move(r, c):
                        continue
                    saved_state = self.save_board_state(board)
                    board.board[r, c] = current_player
                    board.move_history.append(([(r, c)], current_player))
                    if board.check_win(r, c):
                        board.game_over = True
                        board.winner = current_player
                    board.current_player = 3 - current_player
                    board.is_first_turn = False
                    eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    self.restore_board_state(board, saved_state)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
            else:
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        r1, c1 = candidates[i]
                        r2, c2 = candidates[j]
                        if not (board.is_valid_move(r1, c1) and board.is_valid_move(r2, c2)):
                            continue
                        saved_state = self.save_board_state(board)
                        moves = [(r1, c1), (r2, c2)]
                        for r, c in moves:
                            board.board[r, c] = current_player
                        board.move_history.append((moves.copy(), current_player))
                        win_found = False
                        for r, c in moves:
                            if board.check_win(r, c):
                                board.game_over = True
                                board.winner = current_player
                                win_found = True
                                break
                        board.current_player = 3 - current_player
                        board.is_first_turn = False
                        eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                        min_eval = min(min_eval, eval_score)
                        self.restore_board_state(board, saved_state)
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
                    if beta <= alpha:
                        break
            return min_eval

    def save_board_state(self, board: Connect6Board):
        return {
            'board': board.board.copy(),
            'current_player': board.current_player,
            'game_over': board.game_over,
            'winner': board.winner,
            'move_history': list(board.move_history),
            'is_first_turn': board.is_first_turn,
            'last_moves': list(board.last_moves)
        }

    def restore_board_state(self, board: Connect6Board, state):
        board.board = state['board']
        board.current_player = state['current_player']
        board.game_over = state['game_over']
        board.winner = state['winner']
        board.move_history = state['move_history']
        board.is_first_turn = state['is_first_turn']
        board.last_moves = state['last_moves']

    def evaluate_board(self, board: Connect6Board) -> float:
        if board.game_over:
            if board.winner == self.player:
                return 100000
            elif board.winner == self.opponent:
                return -100000
            else:
                return 0
        score = 0
        board_array = np.array(board.board)
        for i in range(board.size):
            for j in range(board.size):
                if board_array[i, j] == 0:
                    score += self.evaluate_position(board_array, i, j, self.player)
                    score -= self.evaluate_position(board_array, i, j, self.opponent) * 0.95
        return score

    def get_best_moves(self, board: Connect6Board) -> List[Tuple[int, int]]:
        start_time = time.time()
        winning_moves = self.check_immediate_win(board)
        if winning_moves:
            self.last_search_time = int((time.time() - start_time) * 1000)
            return winning_moves
        blocking_moves = self.check_immediate_threat(board)
        if blocking_moves:
            self.last_search_time = int((time.time() - start_time) * 1000)
            return self.complete_move(board, blocking_moves)
        candidates = self.get_candidate_moves(board, self.player)
        if not candidates:
            self.last_search_time = int((time.time() - start_time) * 1000)
            return []
        expected = board.get_expected_move_count()
        if expected == 1 and len(board.move_history) == 0:
            center = board.size // 2
            if board.is_valid_move(center, center):
                self.last_search_time = int((time.time() - start_time) * 1000)
                return [(center, center)]
        best_score = -float('inf')
        best_moves = []
        if expected == 1:
            for (r, c) in candidates:
                if not board.is_valid_move(r, c):
                    continue
                saved_state = self.save_board_state(board)
                board.board[r, c] = self.player
                board.move_history.append(([(r, c)], self.player))
                if board.check_win(r, c):
                    board.game_over = True
                    board.winner = self.player
                board.current_player = 3 - self.player
                board.is_first_turn = False
                score = self.minimax(board, self.max_depth - 1, -float('inf'), float('inf'), False)
                self.restore_board_state(board, saved_state)
                if score > best_score:
                    best_score = score
                    best_moves = [(r, c)]
        else:
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    r1, c1 = candidates[i]
                    r2, c2 = candidates[j]
                    if not (board.is_valid_move(r1, c1) and board.is_valid_move(r2, c2)):
                        continue
                    saved_state = self.save_board_state(board)
                    moves = [(r1, c1), (r2, c2)]
                    for r, c in moves:
                        board.board[r, c] = self.player
                    board.move_history.append((moves.copy(), self.player))
                    win_found = False
                    for r, c in moves:
                        if board.check_win(r, c):
                            board.game_over = True
                            board.winner = self.player
                            win_found = True
                            break
                    board.current_player = 3 - self.player
                    board.is_first_turn = False
                    score = self.minimax(board, self.max_depth - 1, -float('inf'), float('inf'), False)
                    self.restore_board_state(board, saved_state)
                    if score > best_score:
                        best_score = score
                        best_moves = moves
        self.last_search_time = int((time.time() - start_time) * 1000)
        if not best_moves:
            best_moves = self.heuristic_fallback(board)
        return best_moves

    def check_immediate_win(self, board: Connect6Board) -> List[Tuple[int, int]]:
        expected = board.get_expected_move_count()
        valid_moves = board.get_available_moves()
        if expected == 1:
            for r, c in valid_moves:
                board.board[r, c] = self.player
                if board.check_win(r, c):
                    board.board[r, c] = 0
                    return [(r, c)]
                board.board[r, c] = 0
        else:
            single_winning_moves = []
            for r, c in valid_moves:
                board.board[r, c] = self.player
                if board.check_win(r, c):
                    single_winning_moves.append((r, c))
                board.board[r, c] = 0
            if single_winning_moves:
                return single_winning_moves[:expected]

            for i in range(len(valid_moves)):
                for j in range(i + 1, len(valid_moves)):
                    r1, c1 = valid_moves[i]
                    r2, c2 = valid_moves[j]
                    board.board[r1, c1] = self.player
                    board.board[r2, c2] = self.player
                    if board.check_win(r1, c1) or board.check_win(r2, c2):
                        board.board[r1, c1] = 0
                        board.board[r2, c2] = 0
                        return [(r1, c1), (r2, c2)]
                    board.board[r1, c1] = 0
                    board.board[r2, c2] = 0
        return []

    def check_immediate_win_for_player(self, board: Connect6Board, player: int, expected: Optional[int] = None) -> List[Tuple[int, int]]:
        expected = expected or board.get_expected_move_count()
        valid_moves = board.get_available_moves()
        if expected == 1:
            for r, c in valid_moves:
                board.board[r, c] = player
                if board.check_win(r, c):
                    board.board[r, c] = 0
                    return [(r, c)]
                board.board[r, c] = 0
        else:
            single_winning_moves = []
            for r, c in valid_moves:
                board.board[r, c] = player
                if board.check_win(r, c):
                    single_winning_moves.append((r, c))
                board.board[r, c] = 0
            if single_winning_moves:
                return single_winning_moves[:expected]

            for i in range(len(valid_moves)):
                for j in range(i + 1, len(valid_moves)):
                    r1, c1 = valid_moves[i]
                    r2, c2 = valid_moves[j]
                    board.board[r1, c1] = player
                    board.board[r2, c2] = player
                    if board.check_win(r1, c1) or board.check_win(r2, c2):
                        board.board[r1, c1] = 0
                        board.board[r2, c2] = 0
                        return [(r1, c1), (r2, c2)]
                    board.board[r1, c1] = 0
                    board.board[r2, c2] = 0
        return []

    def check_immediate_threat(self, board: Connect6Board) -> List[Tuple[int, int]]:
        return self.check_immediate_win_for_player(board, self.opponent, board.get_expected_move_count())

    def complete_move(self, board: Connect6Board, preferred: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        expected = board.get_expected_move_count()
        moves = []
        for move in preferred:
            if len(moves) >= expected:
                break
            r, c = move
            if board.is_valid_move(r, c) and move not in moves:
                moves.append(move)
        if len(moves) < expected:
            for move in self.heuristic_fallback(board):
                if len(moves) >= expected:
                    break
                if move not in moves and board.is_valid_move(*move):
                    moves.append(move)
        return moves

    def heuristic_fallback(self, board: Connect6Board) -> List[Tuple[int, int]]:
        expected = board.get_expected_move_count()
        valid_moves = board.get_available_moves()
        if not valid_moves:
            return []
        if expected == 1:
            center = board.size // 2
            if board.is_valid_move(center, center):
                return [(center, center)]
            board_array = np.array(board.board)
            best_score = -1
            best_move = None
            for r, c in valid_moves:
                score = self.evaluate_position(board_array, r, c, self.player)
                if score > best_score:
                    best_score = score
                    best_move = (r, c)
            return [best_move] if best_move else [random.choice(valid_moves)]
        else:
            if len(valid_moves) < 2:
                return valid_moves[:]
            board_array = np.array(board.board)
            scored_moves = []
            for r, c in valid_moves:
                score = self.evaluate_position(board_array, r, c, self.player)
                score += self.evaluate_position(board_array, r, c, self.opponent) * 0.9
                scored_moves.append((score, (r, c)))
            scored_moves.sort(reverse=True, key=lambda x: x[0])
            move1 = scored_moves[0][1]
            move2 = scored_moves[1][1]
            distance = abs(move1[0] - move2[0]) + abs(move1[1] - move2[1])
            if distance <= 2 and len(scored_moves) > 2:
                move2 = scored_moves[2][1]
            return [move1, move2]


# ===================== 强化学习AI（对接训练模型） =====================
class Connect6DeepLearningAI(Connect6MinimaxAI):
    """强化学习六子棋AI（使用训练的模型）"""
    def __init__(self, player: int, board_size: int = BOARD_SIZE):
        self.player = player
        self.opponent = 3 - player
        self.board_size = board_size
        self.level = "deep_learning"
        self.max_depth = 1
        self.candidate_limit = 10
        self.last_search_time = 0
        self.last_candidates = []
        self.last_decision = ''
        self.model_load_error = ''

        # 加载训练好的模型（核心改动）
        self.nn = None
        self.nn = get_cached_model()
        self.model_load_error = cached_model_error
        if self.nn is not None:
            print(f"[OK] 强化学习AI加载训练模型成功 | 棋盘大小: {board_size}")
        else:
            print(f"[ERROR] 加载训练模型失败: {self.model_load_error}")

    def get_best_moves(self, board: Connect6Board):
            """使用训练模型选择最佳动作"""
            start_time = time.time()

            winning_moves = self.check_immediate_win(board)
            if winning_moves:
                self.last_decision = 'rule_immediate_win'
                self.last_search_time = int((time.time() - start_time) * 1000)
                return winning_moves

            blocking_moves = self.check_immediate_threat(board)
            if blocking_moves:
                self.last_decision = 'rule_immediate_block'
                self.last_search_time = int((time.time() - start_time) * 1000)
                return self.complete_move(board, blocking_moves)

            tactical_moves = self.get_tactical_moves(board)
            if tactical_moves:
                self.last_decision = 'rule_tactical'
                self.last_search_time = int((time.time() - start_time) * 1000)
                return tactical_moves

            # 如果模型未加载，使用启发式方法
            if self.nn is None:
                print("[WARN] 模型未加载，使用启发式方法")
                self.last_search_time = int((time.time() - start_time) * 1000)
                self.last_decision = 'fallback_heuristic'
                return self.heuristic_fallback(board)

            # 2. 使用模型预测
            try:
                # 构建模型输入
                board_array = np.array(board.board)

                if board.size != BOARD_SIZE:
                    self.last_decision = 'fallback_board_size'
                    self.last_search_time = int((time.time() - start_time) * 1000)
                    return self.heuristic_fallback(board)

                # 创建特征张量
                features = np.zeros((1, BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS), dtype=np.float32)

                # 填充特征（与训练时保持一致）
                # 通道0: 当前玩家棋子
                features[0, :, :, 0] = (board_array == self.player)
                # 通道1: 对手棋子
                features[0, :, :, 1] = (board_array == (3 - self.player))

                # 预测
                policy, value = self.nn.predict(features, verbose=0)

                # 将策略展平为二维
                policy_2d = policy[0].reshape(BOARD_SIZE, BOARD_SIZE)

                # 3. 从合法动作中选择
                valid_moves = board.get_available_moves()
                expected = board.get_expected_move_count()

                if not valid_moves:
                    self.last_decision = 'no_move'
                    self.last_search_time = int((time.time() - start_time) * 1000)
                    return []

                # 按策略值排序合法动作
                scored_moves = []
                candidates = self.get_model_candidates(board, policy_2d, limit=max(18, self.candidate_limit))
                for r, c in candidates:
                    policy_score = float(policy_2d[r, c])
                    attack_score = self.position_heuristic(board_array, r, c, self.player)
                    defense_score = self.position_heuristic(board_array, r, c, self.opponent)
                    score = self.point_score(board_array, r, c, policy_2d)
                    scored_moves.append((score, (r, c)))
                scored_moves.sort(reverse=True, key=lambda x: x[0])
                self.last_candidates = [m for _, m in scored_moves[:self.candidate_limit]]

                # 根据所需数量选择动作
                if expected == 1:
                    if scored_moves:
                        best_move = [scored_moves[0][1]]
                        self.last_decision = 'model_point_score'
                        self.last_search_time = int((time.time() - start_time) * 1000)
                        return best_move
                else:
                    pair = self.choose_best_pair(board, candidates, policy_2d)
                    if pair:
                        self.last_decision = 'model_pair_score'
                        self.last_search_time = int((time.time() - start_time) * 1000)
                        return pair
                    if len(scored_moves) >= 2:
                        move1 = scored_moves[0][1]
                        move2 = scored_moves[1][1]
                        # 确保两个动作不同
                        if move1 == move2 and len(scored_moves) > 2:
                            move2 = scored_moves[2][1]
                        self.last_decision = 'model_point_pair'
                        self.last_search_time = int((time.time() - start_time) * 1000)
                        return [move1, move2]
                    elif scored_moves:
                        self.last_decision = 'model_point_score'
                        self.last_search_time = int((time.time() - start_time) * 1000)
                        return [scored_moves[0][1]]

            except Exception as e:
                print(f"[ERROR] 模型预测错误: {e}")
                # 出错时使用启发式方法
                return self.heuristic_fallback(board)

            # 默认返回空列表
            self.last_search_time = int((time.time() - start_time) * 1000)
            return []

    def iter_segments(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    end_r = r + dr * 5
                    end_c = c + dc * 5
                    if 0 <= end_r < self.board_size and 0 <= end_c < self.board_size:
                        yield [(r + dr * i, c + dc * i) for i in range(6)]

    def line_completion_points(self, board_array, player, stones_needed):
        points = set()
        for segment in self.iter_segments():
            vals = [board_array[r, c] for r, c in segment]
            if vals.count(player) == 6 - stones_needed and vals.count(0) == stones_needed:
                for r, c in segment:
                    if board_array[r, c] == 0:
                        points.add((r, c))
        return sorted(points)

    def get_tactical_moves(self, board: Connect6Board):
        expected = board.get_expected_move_count()
        board_array = np.array(board.board)

        if expected == 2:
            own_five = self.line_completion_points(board_array, self.player, 1)
            if own_five:
                return self.complete_move(board, own_five)
            opponent_five = self.line_completion_points(board_array, self.opponent, 1)
            if opponent_five:
                return self.complete_move(board, opponent_five)

        own_finish = self.line_completion_points(board_array, self.player, expected)
        if own_finish:
            return self.complete_move(board, own_finish)

        opponent_finish = self.line_completion_points(board_array, self.opponent, expected)
        if opponent_finish:
            return self.complete_move(board, opponent_finish)

        return []

    def get_model_candidates(self, board: Connect6Board, policy_2d, limit=18):
        board_array = np.array(board.board)
        valid = set(board.get_available_moves())
        candidates = []
        seen = set()

        def add(move):
            if move in valid and move not in seen:
                seen.add(move)
                candidates.append(move)

        for stones_needed in (1, 2):
            for move in self.line_completion_points(board_array, self.player, stones_needed):
                add(move)
            for move in self.line_completion_points(board_array, self.opponent, stones_needed):
                add(move)

        stones = np.argwhere(board_array != 0)
        local = []
        for sr, sc in stones:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = int(sr + dr), int(sc + dc)
                    if (r, c) in valid:
                        local.append((self.point_score(board_array, r, c, policy_2d), (r, c)))
        local.sort(reverse=True, key=lambda x: x[0])
        for _, move in local[:limit]:
            add(move)

        flat_order = np.argsort(policy_2d.flatten())[::-1]
        for idx in flat_order:
            add((int(idx // BOARD_SIZE), int(idx % BOARD_SIZE)))
            if len(candidates) >= limit:
                break

        if not candidates:
            for move in self.heuristic_fallback(board):
                add(move)
        return candidates[:limit]

    def point_score(self, board_array, r, c, policy_2d):
        attack_score = self.position_heuristic(board_array, r, c, self.player)
        defense_score = self.position_heuristic(board_array, r, c, self.opponent)
        policy_score = float(policy_2d[r, c])
        neighbor_score = self.neighbor_score(board_array, r, c)
        return policy_score * 0.15 + attack_score * 0.45 + defense_score * 0.55 + neighbor_score * 0.05

    def neighbor_score(self, board_array, r, c):
        count = 0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.board_size and 0 <= cc < self.board_size and board_array[rr, cc] != 0:
                    count += 1
        return min(count / 8.0, 1.0)

    def choose_best_pair(self, board: Connect6Board, candidates, policy_2d):
        valid = [m for m in candidates if board.is_valid_move(*m)]
        if len(valid) < 2:
            return valid[:]
        board_array = np.array(board.board)
        best_score = -float('inf')
        best_pair = None
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                pair = [valid[i], valid[j]]
                score = self.point_score(board_array, pair[0][0], pair[0][1], policy_2d)
                score += self.point_score(board_array, pair[1][0], pair[1][1], policy_2d)
                temp = board_array.copy()
                for r, c in pair:
                    temp[r, c] = self.player
                score += self.board_tactical_score(temp, self.player) * 0.4
                score -= self.board_tactical_score(temp, self.opponent) * 0.55
                distance = abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1])
                if distance <= 1:
                    score -= 0.2
                if score > best_score:
                    best_score = score
                    best_pair = pair
        return best_pair or valid[:2]

    def board_tactical_score(self, board_array, player):
        score = 0.0
        for segment in self.iter_segments():
            vals = [board_array[r, c] for r, c in segment]
            stones = vals.count(player)
            empties = vals.count(0)
            if stones == 6:
                score += 100.0
            elif stones == 5 and empties == 1:
                score += 20.0
            elif stones == 4 and empties == 2:
                score += 6.0
            elif stones == 3 and empties == 3:
                score += 1.5
        return score

    def _board_to_input(self, board):
        """
        将 15x15 棋盘编码成 17 通道的输入 (1, 15, 15, 17)
        保留原 PolicyValueNet 的编码逻辑
        """
        board_input = np.zeros((1, 15, 15, 17), dtype=np.float32)
        # 示例编码：
        # 通道 0: 自己棋子
        board_input[0, :, :, 0] = (board == 1)
        # 通道 1: 对手棋子
        board_input[0, :, :, 1] = (board == 2)
        # 通道 2~16 可以按原训练逻辑填充历史棋盘、重复位置等信息
        # 如果你不在意历史信息，至少保留两个通道
        return board_input

    def _policy_to_moves(self, policy):
        """
        根据模型输出策略选动作
        policy: shape (1, 225)
        返回动作列表 [(x1, y1), (x2, y2), ...]
        """
        policy = policy.flatten()  # (225,)
        # 排序得到概率最高的动作
        idx = np.argsort(-policy)
        moves = [(i // 15, i % 15) for i in idx]
        return moves


    def position_heuristic(self, board_array, r, c, player):
        """位置启发式评估"""
        score = 0
        opponent = 3 - player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        # 攻击评估
        for dx, dy in directions:
            attack_score = self.evaluate_line_potential(board_array, r, c, dx, dy, player)
            score += attack_score
            # 防守评估
            defense_score = self.evaluate_line_potential(board_array, r, c, dx, dy, opponent)
            score += defense_score * 0.8
        return min(score / 100.0, 1.0)

    def evaluate_line_potential(self, board_array, r, c, dx, dy, player):
        """评估线潜力"""
        size = self.board_size
        score = 0
        temp_board = board_array.copy()
        temp_board[r, c] = player
        forward = 0
        for step in range(1, 6):
            x, y = r + dx * step, c + dy * step
            if 0 <= x < size and 0 <= y < size and temp_board[x, y] == player:
                forward += 1
            else:
                break
        backward = 0
        for step in range(1, 6):
            x, y = r - dx * step, c - dy * step
            if 0 <= x < size and 0 <= y < size and temp_board[x, y] == player:
                backward += 1
            else:
                break
        total = forward + backward + 1
        if total >= 6:
            score += 100
        elif total == 5:
            score += 50
        elif total == 4:
            score += 20
        elif total == 3:
            score += 10
        elif total == 2:
            score += 5
        return score

    def select_moves_from_policy(self, policy, board: Connect6Board):
        """从模型策略中选择符合规则的移动"""
        expected = board.get_expected_move_count()
        valid_moves = board.get_available_moves()
        if not valid_moves:
            return []

        # 按策略值排序
        scored_moves = []
        for r, c in valid_moves:
            score = policy[r, c]
            scored_moves.append((score, (r, c)))
        scored_moves.sort(reverse=True, key=lambda x: x[0])

        if expected == 1:
            # 首手1子：选策略值最高的
            return [scored_moves[0][1]] if scored_moves else []
        else:
            # 后续2子：选前两个最高的（且不同）
            if len(scored_moves) >= 2:
                move1 = scored_moves[0][1]
                move2 = scored_moves[1][1]
                if move1 == move2 and len(scored_moves) > 2:
                    move2 = scored_moves[2][1]
                return [move1, move2]
            elif len(scored_moves) == 1:
                return [scored_moves[0][1]]
            else:
                return []

    def check_immediate_win(self, board: Connect6Board) -> List[Tuple[int, int]]:
        """检查立即获胜"""
        expected = board.get_expected_move_count()
        valid_moves = board.get_available_moves()
        if expected == 1:
            for r, c in valid_moves:
                board.board[r, c] = self.player
                if board.check_win(r, c):
                    board.board[r, c] = 0
                    return [(r, c)]
                board.board[r, c] = 0
        else:
            single_winning_moves = []
            for r, c in valid_moves:
                board.board[r, c] = self.player
                if board.check_win(r, c):
                    single_winning_moves.append((r, c))
                board.board[r, c] = 0
            if single_winning_moves:
                return single_winning_moves[:expected]

            for i in range(len(valid_moves)):
                for j in range(i + 1, len(valid_moves)):
                    r1, c1 = valid_moves[i]
                    r2, c2 = valid_moves[j]
                    board.board[r1, c1] = self.player
                    board.board[r2, c2] = self.player
                    if board.check_win(r1, c1) or board.check_win(r2, c2):
                        board.board[r1, c1] = 0
                        board.board[r2, c2] = 0
                        return [(r1, c1), (r2, c2)]
                    board.board[r1, c1] = 0
                    board.board[r2, c2] = 0
        return []


class Connect6PureModelAI(Connect6DeepLearningAI):
    """纯模型AI：只根据模型 policy 选合法点，不使用规则堵棋或启发式评分。"""
    def __init__(self, player: int, board_size: int = BOARD_SIZE):
        super().__init__(player, board_size)
        self.level = "pure_model"

    def get_best_moves(self, board: Connect6Board):
        start_time = time.time()
        if self.nn is None or board.size != BOARD_SIZE:
            self.last_decision = 'no_move'
            self.last_search_time = int((time.time() - start_time) * 1000)
            return []

        try:
            board_array = np.array(board.board)
            features = np.zeros((1, BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS), dtype=np.float32)
            features[0, :, :, 0] = (board_array == self.player)
            features[0, :, :, 1] = (board_array == self.opponent)

            policy, _ = self.nn.predict(features, verbose=0)
            policy_2d = policy[0].reshape(BOARD_SIZE, BOARD_SIZE)
            valid_moves = board.get_available_moves()
            expected = board.get_expected_move_count()
            scored_moves = [
                (float(policy_2d[r, c]), (r, c))
                for r, c in valid_moves
            ]
            scored_moves.sort(reverse=True, key=lambda x: x[0])
            self.last_candidates = [m for _, m in scored_moves[:self.candidate_limit]]
            self.last_search_time = int((time.time() - start_time) * 1000)
            self.last_decision = 'raw_policy'
            return [m for _, m in scored_moves[:expected]]
        except Exception as e:
            self.model_load_error = str(e)
            print(f"[ERROR] 纯模型预测错误: {e}")
            self.last_search_time = int((time.time() - start_time) * 1000)
            return []



# ===================== 全局状态与API =====================
# 全局游戏状态
game_state = {
    'board': None,
    'ai': None,
    'human_color': 1,
    'ai_color': 2,
    'ai_type': 'deep_learning',  # 默认使用强化学习AI（训练模型）
    'ai_level': 'medium',
    'review_mode': False,
    'review_index': 0,
    'redo_stack': []
}

training_lock = threading.Lock()
training_status = {
    'running': False,
    'success': None,
    'message': '尚未开始训练',
    'started_at': None,
    'finished_at': None,
    'num_games': None,
    'epochs': None,
    'iterations': None,
    'returncode': None,
    'stdout_tail': '',
    'stderr_tail': ''
}


def is_training_model_active(ai, board=None):
    """训练模型只支持 15x15 棋盘；其他尺寸会退回启发式。"""
    if not isinstance(ai, Connect6DeepLearningAI):
        return False
    if ai.nn is None:
        return False
    if board is not None and board.size != BOARD_SIZE:
        return False
    return True


def get_ai_runtime(ai, board):
    if isinstance(ai, Connect6PureModelAI):
        return 'pure_model' if is_training_model_active(ai, board) else 'model_unavailable'
    if isinstance(ai, Connect6DeepLearningAI):
        return 'enhanced_model' if is_training_model_active(ai, board) else 'fallback_heuristic'
    return 'minimax'


def _tail(text, limit=4000):
    if not text:
        return ''
    return text[-limit:]


def _run_training(num_games, epochs, iterations):
    global cached_model, cached_model_error, cached_model_mtime
    cmd = [
        sys.executable,
        'train.py',
        '--num_games',
        str(num_games),
        '--epochs',
        str(epochs),
        '--iterations',
        str(iterations)
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        with training_lock:
            training_status.update({
                'running': False,
                'success': result.returncode == 0,
                'message': '模型训练成功' if result.returncode == 0 else '训练失败',
                'finished_at': time.time(),
                'returncode': result.returncode,
                'stdout_tail': _tail(result.stdout),
                'stderr_tail': _tail(result.stderr)
            })
        if result.returncode == 0:
            with model_cache_lock:
                cached_model = None
                cached_model_mtime = None
                cached_model_error = ''
    except subprocess.TimeoutExpired as e:
        with training_lock:
            training_status.update({
                'running': False,
                'success': False,
                'message': '训练超时',
                'finished_at': time.time(),
                'returncode': None,
                'stdout_tail': _tail(e.stdout if isinstance(e.stdout, str) else ''),
                'stderr_tail': _tail(e.stderr if isinstance(e.stderr, str) else '')
            })
    except Exception as e:
        with training_lock:
            training_status.update({
                'running': False,
                'success': False,
                'message': f'训练接口异常: {e}',
                'finished_at': time.time(),
                'returncode': None,
                'stderr_tail': str(e)
            })


def get_game_state():
    board = game_state['board']
    if not board:
        return {'error': '游戏未开始'}
    if game_state['review_mode']:
        state = get_review_state()
    else:
        state = board.to_dict()
    ai = game_state['ai']
    state.update({
        'human_color': game_state['human_color'],
        'ai_color': game_state['ai_color'],
        'ai_type': game_state['ai_type'],
        'ai_level': game_state['ai_level'],
        'review_mode': game_state['review_mode'],
        'review_index': game_state['review_index'],
        'can_undo': bool(board.move_history),
        'can_redo': bool(game_state['redo_stack']),
        'status': get_status_message(board),
        'model_loaded': is_training_model_active(ai, board),
        'model_error': getattr(ai, 'model_load_error', '') if ai else '',
        'ai_runtime': get_ai_runtime(ai, board),
        'ai_decision': getattr(ai, 'last_decision', '') if ai else '',
        'ai_candidates': getattr(ai, 'last_candidates', []) if ai else []
    })
    return state

def get_review_state():
    board = game_state['board']
    review_index = game_state['review_index']
    temp_board = Connect6Board(board.size)
    temp_board.reset()
    move_numbers = {}
    count = 0
    for idx in range(review_index):
        moves, player = board.move_history[idx]
        for i, j in moves:
            temp_board.board[i, j] = player
            count += 1
            move_numbers[f"{i},{j}"] = count
    state = {
        'size': temp_board.size,
        'board': temp_board.board.tolist(),
        'current_player': temp_board.current_player,
        'game_over': temp_board.game_over,
        'winner': temp_board.winner,
        'last_moves': [],
        'move_history': board.move_history[:review_index],
        'is_first_turn': temp_board.is_first_turn,
        'expected_move_count': temp_board.get_expected_move_count(),
        'move_numbers': move_numbers,
        'review_step': f"{review_index}/{len(board.move_history)}"
    }
    return state

def get_status_message(board):
    if game_state['review_mode']:
        return f"复盘模式: {game_state['review_index']}/{len(board.move_history)} 手"
    if board.game_over:
        if board.winner == game_state['human_color']:
            return "恭喜！你赢了！"
        elif board.winner == game_state['ai_color']:
            return "机器人赢了！"
        else:
            return "平局！"
    else:
        player_color = "黑棋" if board.current_player == 1 else "白棋"
        moves_needed = board.get_expected_move_count()
        if board.current_player == game_state['human_color']:
            return f"轮到你落子（{player_color}），需要下{moves_needed}子"
        else:
            if game_state['ai_type'] == 'pure_model':
                ai_type = "强化学习纯模型"
            elif game_state['ai_type'] == 'deep_learning':
                ai_type = "强化学习增强"
            else:
                ai_type = "传统Minimax"
            ai_level = game_state['ai_level']
            return f"{ai_type} AI思考中（{player_color}, 难度:{ai_level}）..."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.json or {}
    board_size = int(data.get('board_size', BOARD_SIZE))  # 默认使用训练的棋盘大小
    ai_level = data.get('ai_level', 'medium')
    ai_type = data.get('ai_type', 'deep_learning')  # 默认强化学习AI
    first_move = data.get('first_move', 'human')
    if ai_type in ('deep_learning', 'pure_model') and board_size != BOARD_SIZE:
        ai_type = 'minimax'

    human_color = 1 if first_move == 'human' else 2
    ai_color = 3 - human_color

    # 创建棋盘
    board = Connect6Board(board_size)
    # 创建AI（优先使用训练模型）
    if ai_type == 'pure_model':
        ai = Connect6PureModelAI(ai_color, board_size)
    elif ai_type == 'deep_learning':
        ai = Connect6DeepLearningAI(ai_color, board_size)
    else:
        ai = Connect6MinimaxAI(ai_color, ai_level)

    # 更新全局状态
    game_state.update({
        'board': board,
        'ai': ai,
        'human_color': human_color,
        'ai_color': ai_color,
        'ai_type': ai_type,
        'ai_level': ai_level,
        'review_mode': False,
        'review_index': 0,
        'redo_stack': []
    })

    # AI先手则自动落子
    if board.current_player == ai_color:
        ai_moves = ai.get_best_moves(board)
        if ai_moves:
            board.make_move(ai_moves)

    resp = get_game_state()
    # 添加AI调试信息
    if hasattr(ai, 'last_search_time'):
        resp['ai_time_ms'] = ai.last_search_time
    if hasattr(ai, 'last_candidates'):
        resp['ai_candidates'] = ai.last_candidates
    # 标识是否使用训练模型
    resp['model_loaded'] = is_training_model_active(ai, board)
    if board.current_player == ai_color and not board.game_over:
        resp['ai_error'] = 'AI未能生成合法落子'

    return jsonify(resp)


@app.route('/api/move', methods=['POST'])
def make_move():
    data = request.json or {}
    moves = data.get('moves', [])
    board = game_state['board']
    if not board:
        return jsonify({'error': '游戏未开始'}), 400

    if game_state['review_mode']:
        game_state['review_mode'] = False
    game_state['redo_stack'] = []

    # 转换移动格式
    moves = [tuple(m) for m in moves]
    # 执行玩家移动
    success = board.make_move(moves)
    if not success:
        return jsonify({'error': '无效的移动'}), 400

    resp = get_game_state()
    # AI回合
    if not board.game_over and board.current_player == game_state['ai_color']:
        ai = game_state['ai']
        ai_moves = ai.get_best_moves(board)
        if ai_moves:
            board.make_move(ai_moves)
        else:
            resp = get_game_state()
            resp['ai_error'] = 'AI未能生成合法落子'
            return jsonify(resp), 500
        resp = get_game_state()

    # AI调试信息
    ai = game_state['ai']
    if hasattr(ai, 'last_search_time'):
        resp['ai_time_ms'] = ai.last_search_time
    if hasattr(ai, 'last_candidates'):
        resp['ai_candidates'] = ai.last_candidates
    resp['model_loaded'] = is_training_model_active(ai, board)

    return jsonify(resp)


@app.route('/api/undo', methods=['POST'])
def undo_move():
    board = game_state['board']
    if not board:
        return jsonify({'error': '游戏未开始'}), 400
    if game_state['review_mode']:
        game_state['review_mode'] = False
    undone = []
    first = board.pop_last_move()
    if first:
        undone.insert(0, first)
        if board.current_player == game_state['ai_color']:
            second = board.pop_last_move()
            if second:
                undone.insert(0, second)
        if undone:
            game_state['redo_stack'].append(undone)
        resp = get_game_state()
        ai = game_state['ai']
        if hasattr(ai, 'last_search_time'):
            resp['ai_time_ms'] = ai.last_search_time
        if hasattr(ai, 'last_candidates'):
            resp['ai_candidates'] = ai.last_candidates
        resp['model_loaded'] = is_training_model_active(ai, board)
        return jsonify(resp)
    else:
        return jsonify({'error': '无法悔棋'}), 400


@app.route('/api/redo', methods=['POST'])
def redo_move():
    board = game_state['board']
    if not board:
        return jsonify({'error': '游戏未开始'}), 400
    if game_state['review_mode']:
        game_state['review_mode'] = False
    if not game_state['redo_stack']:
        return jsonify({'error': '没有可前进的棋步'}), 400

    records = game_state['redo_stack'].pop()
    for record in records:
        if not board.replay_move(record):
            return jsonify({'error': '无法恢复棋步'}), 400

    resp = get_game_state()
    ai = game_state['ai']
    if hasattr(ai, 'last_search_time'):
        resp['ai_time_ms'] = ai.last_search_time
    if hasattr(ai, 'last_candidates'):
        resp['ai_candidates'] = ai.last_candidates
    resp['model_loaded'] = is_training_model_active(ai, board)
    return jsonify(resp)


@app.route('/api/restart', methods=['POST'])
def restart_game():
    board = game_state['board']
    ai = game_state['ai']
    if board and ai:
        board.reset()
        game_state['review_mode'] = False
        game_state['review_index'] = 0
        game_state['redo_stack'] = []
        # AI先手则落子
        if board.current_player == game_state['ai_color']:
            ai_moves = ai.get_best_moves(board)
            if ai_moves:
                board.make_move(ai_moves)
    resp = get_game_state()
    ai = game_state['ai']
    if hasattr(ai, 'last_search_time'):
        resp['ai_time_ms'] = ai.last_search_time
    if hasattr(ai, 'last_candidates'):
        resp['ai_candidates'] = ai.last_candidates
    resp['model_loaded'] = is_training_model_active(ai, board)
    return jsonify(resp)


@app.route('/api/game_state', methods=['GET'])
def get_game_state_route():
    s = get_game_state()
    if 'error' in s:
        return jsonify(s), 400
    # AI调试信息
    ai = game_state['ai']
    if ai and hasattr(ai, 'last_search_time'):
        s['ai_time_ms'] = ai.last_search_time
    if ai and hasattr(ai, 'last_candidates'):
        s['ai_candidates'] = ai.last_candidates
    s['model_loaded'] = is_training_model_active(ai, game_state['board'])
    return jsonify(s)


@app.route('/api/review/toggle', methods=['POST'])
def toggle_review():
    board = game_state['board']
    if not board:
        return jsonify({'error': '游戏未开始'}), 400
    game_state['review_mode'] = not game_state['review_mode']
    if game_state['review_mode']:
        game_state['review_index'] = len(board.move_history)
    else:
        game_state['review_index'] = len(board.move_history)
    resp = get_game_state()
    return jsonify(resp)


@app.route('/api/review/previous', methods=['POST'])
def previous_move():
    board = game_state['board']
    if not board:
        return jsonify({'error': '游戏未开始'}), 400
    if not game_state['review_mode']:
        return jsonify({'error': '不在复盘模式'}), 400
    if game_state['review_index'] > 0:
        game_state['review_index'] -= 1
    resp = get_game_state()
    return jsonify(resp)


@app.route('/api/review/next', methods=['POST'])
def next_move():
    board = game_state['board']
    if not board:
        return jsonify({'error': '游戏未开始'}), 400
    if not game_state['review_mode']:
        return jsonify({'error': '不在复盘模式'}), 400
    if game_state['review_index'] < len(board.move_history):
        game_state['review_index'] += 1
    resp = get_game_state()
    return jsonify(resp)


@app.route('/api/train', methods=['POST'])
@app.route('/api/train_model', methods=['POST'])
def train_model():
    """启动后台训练任务（对接训练脚本）"""
    try:
        if not TF_AVAILABLE:
            return jsonify({
                'success': False,
                'message': 'TensorFlow不可用，无法训练'
            }), 500

        data = request.get_json(silent=True) or {}
        num_games = int(data.get('num_games', 20))
        epochs = int(data.get('epochs', 3))
        iterations = int(data.get('iterations', 1))

        if num_games <= 0 or epochs <= 0 or iterations <= 0:
            return jsonify({
                'success': False,
                'message': '训练参数必须为正整数'
            }), 400

        with training_lock:
            if training_status['running']:
                return jsonify({
                    'success': False,
                    'message': '训练已在运行中',
                    'status': training_status
                }), 409
            training_status.update({
                'running': True,
                'success': None,
                'message': '训练已启动',
                'started_at': time.time(),
                'finished_at': None,
                'num_games': num_games,
                'epochs': epochs,
                'iterations': iterations,
                'returncode': None,
                'stdout_tail': '',
                'stderr_tail': ''
            })

        worker = threading.Thread(
            target=_run_training,
            args=(num_games, epochs, iterations),
            daemon=True
        )
        worker.start()

        return jsonify({
            'success': True,
            'message': '训练已在后台启动',
            'status': training_status
        }), 202

    except Exception as e:
        return jsonify({
            'success': False,
            'message': '训练接口异常',
            'error': str(e)
        }), 500


@app.route('/api/train_status', methods=['GET'])
def get_train_status():
    """查询后台训练状态"""
    with training_lock:
        status = dict(training_status)
    status['model_path'] = MODEL_PATH
    status['model_exists'] = os.path.exists(MODEL_PATH)
    status['model_last_modified'] = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    return jsonify({
        'success': True,
        'status': status
    })


@app.route('/api/download_model', methods=['GET'])
def download_model():
    """下载训练好的模型文件"""
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True)
    elif os.path.exists(OLD_MODEL_PATH):
        return send_file(OLD_MODEL_PATH, as_attachment=True)
    else:
        return jsonify({'error': '模型文件不存在'}), 404


@app.route('/api/model_info', methods=['GET'])
def model_info():
    """获取训练模型信息"""
    if not TF_AVAILABLE:
        return jsonify({'error': 'TensorFlow不可用'}), 500

    try:
        exists = os.path.exists(MODEL_PATH)
        loadable = False
        load_error = ''
        if exists:
            try:
                from tensorflow.keras.models import load_model
                loaded_model = load_model(MODEL_PATH, compile=False)
                loadable = True
                del loaded_model
            except Exception as e:
                load_error = str(e)

        nn = PolicyValueNet()
        info = {
            'model_path': MODEL_PATH,
            'exists': exists,
            'loadable': loadable,
            'load_error': load_error,
            'board_size': BOARD_SIZE,
            'residual_blocks': RES_BLOCKS,
            'conv_filters': FILTERS,
            'input_channels': INPUT_CHANNELS,
            'total_params': nn.model.count_params(),
            'layers': len(nn.model.layers),
            'last_modified': os.path.getmtime(MODEL_PATH) if exists else None
        }
        return jsonify({'success': True, 'model_info': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 80)
    print("六子棋游戏服务器（对接训练模型版）")
    print(f"训练模型路径: {MODEL_PATH}")
    print(f"模型是否存在: {'是' if os.path.exists(MODEL_PATH) else '否'}")
    print(f"TensorFlow: {'可用' if TF_AVAILABLE else '不可用'}")
    print("访问地址: http://localhost:5000")
    print("=" * 80)
    app.run(debug=False, host='0.0.0.0', port=5000)
