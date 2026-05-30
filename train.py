from tqdm import trange, tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras import mixed_precision
import os
import argparse
import shutil
import json
import time
from itertools import combinations
from tensorflow.keras.models import load_model

# =========================
# GPU 配置
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("[OK] GPU ready:", gpus[0])
else:
    print("[WARN] No GPU found")

# =========================
# 超参数
# =========================
BOARD_SIZE = 15
INPUT_CHANNELS = 17
RES_BLOCKS = 6
FILTERS = 64

MCTS_SIMULATIONS = 5
C_PUCT = 1.5
TOP_K = 18
MAX_CANDIDATE_POINTS = 24
EXPLORATION_POINTS = 12

NUM_SELF_PLAY_GAMES = 20
MAX_MOVES = 20

TRAIN_BATCH = 128
EPOCHS_PER_ITER = 3
LEARNING_RATE = 1e-6
L2_REG = 1e-4
NUM_ITERATIONS = 5
POLICY_TEMPERATURE = 1.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

MODEL_PATH = "models/connect6.model.h5"
TRAINING_META_PATH = "models/training_meta.json"
REPLAY_BUFFER_PATH = "data/replay_buffer.npz"
REPLAY_BUFFER_SIZE = 4000
TACTICAL_SAMPLES_PER_ITER = 2048
TACTICAL_EPOCHS_PER_ITER = 3
TEACHER_SAMPLES_PER_RUN = 5000
TEACHER_EPOCHS_PER_RUN = 3
TACTICAL_GATE_PASS_RATE = 0.97
POLICY_LOSS_WEIGHT = 2.0
VALUE_LOSS_WEIGHT = 0.25
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =========================
# 网络
# =========================
class PolicyValueNet:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        inputs = keras.Input((BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS))
        x = layers.Conv2D(FILTERS, 3, padding="same",
                          kernel_regularizer=regularizers.l2(L2_REG))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        for _ in range(RES_BLOCKS):
            x = self.res_block(x)
        # policy head
        p = layers.Conv2D(2, 1)(x)
        p = layers.Flatten()(p)
        policy = layers.Dense(BOARD_SIZE*BOARD_SIZE, activation="softmax",
                              name="policy", dtype="float32")(p)
        # value head
        v = layers.Conv2D(1, 1)(x)
        v = layers.Flatten()(v)
        v = layers.Dense(128, activation="relu")(v)
        value = layers.Dense(1, activation="tanh",
                             name="value", dtype="float32")(v)
        model = keras.Model(inputs, [policy, value])
        model.compile(
            optimizer=keras.optimizers.Adam(LEARNING_RATE),
            loss={"policy": "categorical_crossentropy", "value": "mse"},
            loss_weights={"policy": POLICY_LOSS_WEIGHT, "value": VALUE_LOSS_WEIGHT}
        )
        return model

    def res_block(self, x):
        shortcut = x
        x = layers.Conv2D(FILTERS, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(FILTERS, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        return layers.ReLU()(x)

    def predict(self, board, player):
        x = board_to_features(board, player)[None]
        p, v = self.model.predict(x, verbose=0)
        return p[0].reshape(BOARD_SIZE, BOARD_SIZE), float(v[0][0])

# =========================
# 游戏工具
# =========================
def board_to_features(board, player):
    f = np.zeros((BOARD_SIZE, BOARD_SIZE, INPUT_CHANNELS))
    f[:, :, 0] = (board == player)
    f[:, :, 1] = (board == 3 - player)
    return f

def check_winner(board, player):
    # 检查横向
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 5):
            if np.all(board[r, c:c+6] == player):
                return player
    # 检查纵向
    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - 5):
            if np.all(board[r:r+6, c] == player):
                return player
    # 检查正对角线（左上到右下）
    for r in range(BOARD_SIZE - 5):
        for c in range(BOARD_SIZE - 5):
            if all(board[r+i, c+i] == player for i in range(6)):
                return player
    # 检查反对角线（右上到左下）
    for r in range(BOARD_SIZE - 5):
        for c in range(5, BOARD_SIZE):
            if all(board[r+i, c-i] == player for i in range(6)):
                return player
    return 0


DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]
LINE_SEGMENTS_6 = None


def in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def line_segments(length=6):
    for dr, dc in DIRECTIONS:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                end_r = r + dr * (length - 1)
                end_c = c + dc * (length - 1)
                if in_bounds(end_r, end_c):
                    yield [(r + dr * i, c + dc * i) for i in range(length)]


def tactical_points_for_player(board, player, stones_needed=1):
    global LINE_SEGMENTS_6
    if LINE_SEGMENTS_6 is None:
        LINE_SEGMENTS_6 = list(line_segments(6))
    points = set()
    for segment in LINE_SEGMENTS_6:
        vals = [board[r, c] for r, c in segment]
        if vals.count(player) == 6 - stones_needed and vals.count(0) == stones_needed:
            for r, c in segment:
                if board[r, c] == 0:
                    points.add((r, c))
    return sorted(points)


def policy_from_points(points):
    target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    valid = [(r, c) for r, c in points if in_bounds(r, c)]
    if not valid:
        return target
    prob = 1.0 / len(valid)
    for r, c in valid:
        target[r, c] = prob
    return target


def make_tactical_board(player, threat_player, rng):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    segments = list(line_segments(6))
    segment = segments[int(rng.integers(len(segments)))]
    empty_count = 1 if rng.random() < 0.55 else 2
    empty_indices = set(rng.choice(6, size=empty_count, replace=False).tolist())
    targets = [segment[i] for i in sorted(empty_indices)]
    for i, (r, c) in enumerate(segment):
        if i not in empty_indices:
            board[r, c] = threat_player

    used = set(segment)
    for _ in range(int(rng.integers(2, 8))):
        base_r, base_c = segment[int(rng.integers(6))]
        candidates = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = base_r + dr, base_c + dc
                if in_bounds(r, c) and board[r, c] == 0 and (r, c) not in used:
                    candidates.append((r, c))
        if not candidates:
            continue
        r, c = candidates[int(rng.integers(len(candidates)))]
        board[r, c] = int(rng.choice([1, 2]))
        used.add((r, c))

    return board, targets


def open_run_case(player, threat_player, start, direction, run_len):
    r0, c0 = start
    dr, dc = direction
    stones = [(r0 + dr * i, c0 + dc * i) for i in range(run_len)]
    if not all(in_bounds(r, c) for r, c in stones):
        return None
    targets = [(r0 - dr, c0 - dc), (r0 + dr * run_len, c0 + dc * run_len)]
    targets = [point for point in targets if in_bounds(*point)]
    if not targets:
        return None
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    for r, c in stones:
        board[r, c] = threat_player
    return board, player, targets


def iter_open_run_cases(player, threat_player, run_len):
    for dr, dc in DIRECTIONS:
        for r0 in range(BOARD_SIZE):
            for c0 in range(BOARD_SIZE):
                case = open_run_case(player, threat_player, (r0, c0), (dr, dc), run_len)
                if case is not None:
                    yield case


def make_direct_tactical_cases(max_per_bucket=96, rng=None):
    cases = []
    rng = rng or np.random.default_rng()
    for player in (1, 2):
        for threat_player in (player, 3 - player):
            for run_len in (5, 4, 3, 2):
                bucket = list(iter_open_run_cases(player, threat_player, run_len))
                rng.shuffle(bucket)
                cases.extend(bucket[:max_per_bucket])
    return cases


def make_segment_tactical_cases(max_per_bucket=256, rng=None):
    cases = []
    rng = rng or np.random.default_rng()
    segments = list(line_segments(6))
    for player in (1, 2):
        for threat_player in (player, 3 - player):
            for stones_needed in (1, 2, 3):
                bucket = []
                for segment in segments:
                    for empty_indices in combinations(range(6), stones_needed):
                        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
                        targets = []
                        empty_set = set(empty_indices)
                        for i, (r, c) in enumerate(segment):
                            if i in empty_set:
                                targets.append((r, c))
                            else:
                                board[r, c] = threat_player
                        bucket.append((board, player, targets))
                rng.shuffle(bucket)
                cases.extend(bucket[:max_per_bucket])
    return cases


def add_context_noise(board, protected_points, rng, count=6):
    protected = set(protected_points)
    occupied = set(map(tuple, np.argwhere(board != 0)))
    for _ in range(count):
        stones = list(occupied) or [(BOARD_SIZE // 2, BOARD_SIZE // 2)]
        base_r, base_c = stones[int(rng.integers(len(stones)))]
        candidates = []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r, c = base_r + dr, base_c + dc
                if in_bounds(r, c) and board[r, c] == 0 and (r, c) not in protected:
                    candidates.append((r, c))
        if not candidates:
            continue
        r, c = candidates[int(rng.integers(len(candidates)))]
        board[r, c] = int(rng.choice([1, 2]))
        occupied.add((r, c))


def make_context_tactical_cases(count, rng):
    cases = []
    for i in range(count):
        player = 1 if i % 2 == 0 else 2
        threat_player = player if (i // 2) % 2 == 0 else 3 - player
        dr, dc = DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]
        run_len = int(rng.choice([2, 3, 4, 5], p=[0.25, 0.30, 0.30, 0.15]))
        for _ in range(40):
            r0 = int(rng.integers(BOARD_SIZE))
            c0 = int(rng.integers(BOARD_SIZE))
            case = open_run_case(player, threat_player, (r0, c0), (dr, dc), run_len)
            if case is not None:
                break
        else:
            continue
        board, _, targets = case
        stones = [(r0 + dr * j, c0 + dc * j) for j in range(run_len)]
        add_context_noise(board, set(stones) | set(targets), rng, count=int(rng.integers(3, 9)))
        cases.append((board, player, targets))
    return cases


def generate_tactical_samples(count):
    if count <= 0:
        return None
    rng = np.random.default_rng()
    x_list, yp_list, yv_list = [], [], []
    direct_cases = make_segment_tactical_cases(rng=rng)
    direct_cases.extend(make_direct_tactical_cases(rng=rng))
    direct_cases.extend(make_context_tactical_cases(max(64, count // 4), rng))
    repeats = max(1, count // max(1, len(direct_cases) * 2))
    for _ in range(repeats):
        for board, player, target_points in direct_cases:
            x_list.append(board_to_features(board, player))
            yp_list.append(policy_from_points(target_points).flatten())
            yv_list.append(0.0)
            if len(x_list) >= count:
                break
        if len(x_list) >= count:
            break

    for i in range(count):
        if len(x_list) >= count:
            break
        player = 1 if i % 2 == 0 else 2
        own_win = (i // 2) % 2 == 0
        threat_player = player if own_win else 3 - player
        board, target_points = make_tactical_board(player, threat_player, rng)
        x_list.append(board_to_features(board, player))
        yp_list.append(policy_from_points(target_points).flatten())
        # Tactical reinforcement is for move selection. Keep value neutral so
        # synthetic patterns do not make the value head saturate.
        yv_list.append(0.0)
    return (
        np.array(x_list, dtype=np.float32),
        np.array(yp_list, dtype=np.float32),
        np.array(yv_list, dtype=np.float32),
    )


def available_points(board):
    return [(int(r), int(c)) for r, c in zip(*np.where(board == 0))]


def move_count_for_board(board, player):
    return 1 if player == 1 and np.count_nonzero(board) == 0 else 2


def wins_after_move(board, player, point):
    r, c = point
    if board[r, c] != 0:
        return False
    for dr, dc in DIRECTIONS:
        total = 1
        nr, nc = r + dr, c + dc
        while in_bounds(nr, nc) and board[nr, nc] == player:
            total += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while in_bounds(nr, nc) and board[nr, nc] == player:
            total += 1
            nr -= dr
            nc -= dc
        if total >= 6:
            return True
    return False


def winning_points(board, player, candidates=None):
    points = candidates if candidates is not None else available_points(board)
    return [point for point in points if wins_after_move(board, player, point)]


def local_candidate_points(board, limit=40):
    stones = np.argwhere(board != 0)
    candidates = set()
    if len(stones) == 0:
        center = BOARD_SIZE // 2
        for r in range(center - 2, center + 3):
            for c in range(center - 2, center + 3):
                if in_bounds(r, c):
                    candidates.add((r, c))
    else:
        for sr, sc in stones:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = int(sr + dr), int(sc + dc)
                    if in_bounds(r, c) and board[r, c] == 0:
                        candidates.add((r, c))
    if not candidates:
        candidates = set(available_points(board))
    center = (BOARD_SIZE - 1) / 2
    ranked = sorted(
        candidates,
        key=lambda p: abs(p[0] - center) + abs(p[1] - center)
    )
    return ranked[:limit]


def max_line_len_after(board, player, point):
    r, c = point
    best = 1
    for dr, dc in DIRECTIONS:
        total = 1
        nr, nc = r + dr, c + dc
        while in_bounds(nr, nc) and board[nr, nc] == player:
            total += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while in_bounds(nr, nc) and board[nr, nc] == player:
            total += 1
            nr -= dr
            nc -= dc
        best = max(best, total)
    return best


def teacher_point_score(board, player, point):
    r, c = point
    opponent = 3 - player
    if board[r, c] != 0:
        return -1e9

    own_len = max_line_len_after(board, player, point)
    opp_len = max_line_len_after(board, opponent, point)
    center = (BOARD_SIZE - 1) / 2
    center_score = 10.0 - 0.35 * (abs(r - center) + abs(c - center))
    neighbor_score = 0.0
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                continue
            if board[nr, nc] == player:
                neighbor_score += 1.4 if abs(dr) <= 1 and abs(dc) <= 1 else 0.5
            elif board[nr, nc] == opponent:
                neighbor_score += 1.0 if abs(dr) <= 1 and abs(dc) <= 1 else 0.35

    return (
        center_score
        + neighbor_score
        + (own_len ** 2) * 4.0
        + (opp_len ** 2) * 3.2
    )


def choose_teacher_moves(board, player, expected=None):
    expected = expected or move_count_for_board(board, player)
    opponent = 3 - player
    chosen = []
    candidate_pool = local_candidate_points(board, limit=48)

    def add_ranked(points):
        nonlocal chosen
        ranked = sorted(
            [p for p in points if board[p[0], p[1]] == 0 and p not in chosen],
            key=lambda p: teacher_point_score(board, player, p),
            reverse=True
        )
        for point in ranked:
            chosen.append(point)
            if len(chosen) >= expected:
                return True
        return False

    # 1. Win immediately when possible.
    if add_ranked(winning_points(board, player, candidate_pool)):
        return chosen[:expected]

    # 2. Block immediate opponent wins.
    if add_ranked(winning_points(board, opponent, candidate_pool)):
        return chosen[:expected]

    # 3. In ordinary positions, prefer local, connected, central points.
    add_ranked(candidate_pool)
    if len(chosen) < expected:
        add_ranked(available_points(board))
    return chosen[:expected]


def make_teacher_position(rng, max_plies=10):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    player = 1
    plies = int(rng.integers(3, max_plies + 1))
    for step in range(plies):
        expected = move_count_for_board(board, player)
        if len(available_points(board)) < expected:
            break
        if step == 0:
            center = BOARD_SIZE // 2
            points = [
                (r, c)
                for r in range(center - 2, center + 3)
                for c in range(center - 2, center + 3)
                if in_bounds(r, c)
            ]
            rng.shuffle(points)
            moves = points[:expected]
        else:
            candidates = local_candidate_points(board, limit=32)
            rng.shuffle(candidates)
            moves = candidates[:expected]
        if len(moves) < expected:
            break
        for r, c in moves:
            board[r, c] = player
        if check_winner(board, player) != 0:
            # Return a non-terminal earlier-looking board instead of teaching from finished states.
            for r, c in moves:
                board[r, c] = 0
            break
        player = 3 - player
    return board, player


def generate_teacher_samples(count):
    if count <= 0:
        return None
    rng = np.random.default_rng()
    x_list, yp_list, yv_list = [], [], []
    attempts = 0
    while len(x_list) < count and attempts < count * 20:
        attempts += 1
        board, player = make_teacher_position(rng)
        expected = move_count_for_board(board, player)
        moves = choose_teacher_moves(board, player, expected)
        if len(moves) != expected:
            continue
        target = policy_from_points(moves)
        if target.sum() <= 0:
            continue
        x_list.append(board_to_features(board, player))
        yp_list.append(target.flatten())
        yv_list.append(0.0)

    if len(x_list) < count:
        print(f"[WARN] Only generated {len(x_list)}/{count} teacher samples")
    return (
        np.array(x_list, dtype=np.float32),
        np.array(yp_list, dtype=np.float32),
        np.array(yv_list, dtype=np.float32),
    )

# =========================
# MCTS
# =========================
class Node:
    def __init__(self, board, player, parent=None, prior=1.0):
        self.board = board
        self.player = player
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.P = prior

    def Q(self):
        return 0 if self.N == 0 else self.W / self.N

def candidate_points(policy, board):
    """混合网络高分点、已有棋子周围点和中心点，避免坏模型把搜索锁死。"""
    empty = board == 0
    points = []
    seen = set()

    def add(point):
        r, c = point
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and empty[r, c] and point not in seen:
            seen.add(point)
            points.append(point)

    for point in tactical_points_for_player(board, 1):
        add(point)
    for point in tactical_points_for_player(board, 2):
        add(point)

    flat = policy.flatten()
    for idx in np.argsort(flat)[::-1]:
        add((int(idx // BOARD_SIZE), int(idx % BOARD_SIZE)))
        if len(points) >= TOP_K:
            break

    stones = np.argwhere(board != 0)
    if len(stones) == 0:
        center = BOARD_SIZE // 2
        for r in range(center - 2, center + 3):
            for c in range(center - 2, center + 3):
                add((r, c))
    else:
        local = []
        for sr, sc in stones:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = int(sr + dr), int(sc + dc)
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and empty[r, c]:
                        local.append((float(policy[r, c]), (r, c)))
        local.sort(reverse=True, key=lambda x: x[0])
        for _, point in local[:TOP_K]:
            add(point)

    target_points = min(MAX_CANDIDATE_POINTS, len(points) + EXPLORATION_POINTS)
    if len(points) < target_points:
        empty_points = list(zip(*np.where(empty)))
        np.random.shuffle(empty_points)
        for point in empty_points:
            add((int(point[0]), int(point[1])))
            if len(points) >= target_points:
                break

    return points[:MAX_CANDIDATE_POINTS]


def topk_actions(policy, board, stones_per_move=2):
    """
    根据网络策略选择动作组合。
    每次选 top_k 个概率最高的位置，然后两两组合。
    fallback_flag = True 表示这是随机补充动作
    """
    points = candidate_points(policy, board)
    fallback = False
    if len(points) < stones_per_move:
        # Fallback：随机选择未落子点补齐
        empty_points = list(zip(*np.where(board == 0)))
        if len(empty_points) >= stones_per_move:
            points = np.random.choice(len(empty_points), size=stones_per_move, replace=False)
            points = [empty_points[i] for i in points]
        else:
            # 真的只剩一个空格
            points = empty_points
        fallback = True

    if stones_per_move == 1:
        actions = [[p] for p in points]
    else:
        actions = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                actions.append([points[i], points[j]])
    return actions,fallback


def policy_target_from_visits(pi):
    """把 MCTS 动作访问分布转成棋盘点分布，保证总和为 1。"""
    target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for action, prob in pi.items():
        if prob <= 0:
            continue
        for r, c in action:
            target[r, c] += prob / len(action)
    total = target.sum()
    if total > 0:
        target /= total
    return target


def select_action_from_visits(pi, temperature=POLICY_TEMPERATURE):
    actions = list(pi.keys())
    probs = np.array([pi[a] for a in actions], dtype=np.float64)
    if len(actions) == 0 or probs.sum() <= 0:
        return None
    if temperature <= 1e-6:
        return actions[int(np.argmax(probs))]
    probs = np.power(probs, 1.0 / temperature)
    probs /= probs.sum()
    return actions[int(np.random.choice(len(actions), p=probs))]


def apply_dirichlet_noise_to_children(node):
    if not node.children or DIRICHLET_EPSILON <= 0:
        return
    actions = list(node.children.keys())
    noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
    for action, n in zip(actions, noise):
        child = node.children[action]
        child.P = (1 - DIRICHLET_EPSILON) * child.P + DIRICHLET_EPSILON * float(n)

def extract_policy(root):
    pi = {}
    total = sum(c.N for c in root.children.values())
    if total == 0:
        return pi
    for a,c in root.children.items():
        pi[a] = c.N / total
    return pi

def first_move_topk(board):
    """
    第一手落子限制在棋盘中心九宫格内，且六子棋黑方首手只下一子。
    返回 actions 列表和 fallback 标记
    """
    center = BOARD_SIZE // 2
    # 中心九宫格坐标范围
    r_range = range(center-1, center+2)
    c_range = range(center-1, center+2)

    # 找空位
    empty_points = [(r, c) for r in r_range for c in c_range if board[r, c] == 0]

    if not empty_points:
        return [empty_points], True  # fallback

    return [[p] for p in empty_points], False



def mcts_batch(root, net, batch_size=32, max_leaves=16):
    """
    批量 MCTS 扩展 + 网络预测
    root: MCTS 根节点
    net: PolicyValueNet
    batch_size: 每次批量预测数量
    max_leaves: 每轮最多批量预测的叶节点数
    """
    for _ in trange(MCTS_SIMULATIONS, desc="MCTS sims", leave=False):
        leaves = []
        paths = []

        node = root
        path = [node]

        while node.children:
            # 树选择
            action, node = max(
                node.children.items(),
                # Child Q is stored from the child player's perspective, so the
                # parent should prefer the negated value.
                key=lambda n: -n[1].Q() + C_PUCT * n[1].P * np.sqrt(node.N + 1) / (1 + n[1].N)
            )
            path.append(node)

        # 检查胜负
        winner = check_winner(node.board, 3 - node.player)
        if winner != 0:
            value = 1 if winner == node.player else -1
            for n in reversed(path):
                n.N += 1
                n.W += value
                value = -value
            continue

        leaves.append((node.board, node.player))
        paths.append(path)

        # 限制叶节点数量
        if len(leaves) > max_leaves:
            leaves = leaves[:max_leaves]
            paths = paths[:max_leaves]

        if leaves:
            x = np.array([board_to_features(b, p) for b, p in leaves])
            p_batch, v_batch = net.model.predict(x, verbose=0)

            for i, path in enumerate(paths):
                node = path[-1]
                policy = p_batch[i].reshape(BOARD_SIZE, BOARD_SIZE)
                value = float(v_batch[i][0])

                actions, fallback = topk_actions(policy, node.board, stones_per_move=2)
                for action in actions:
                    prob = np.mean([policy[r, c] for r, c in action])
                    new_board = node.board.copy()
                    for r, c in action:
                        new_board[r, c] = node.player
                    node.children[tuple(map(tuple, action))] = Node(new_board, 3 - node.player, node, prob)
                if node is root:
                    apply_dirichlet_noise_to_children(node)

                for n in reversed(path):
                    n.N += 1
                    n.W += value
                    value = -value

def self_play_batch(net):
    """
    使用批量 MCTS 生成一局游戏数据
    返回: data, winner, total_steps
    """
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), int)
    player = 1
    data = []
    root = Node(board, player)  # 初始根节点

    for step in range(MAX_MOVES):
        # ===============================
        # 第一手：强制中心九宫格
        # ===============================
        if step == 0:
            actions, _ = first_move_topk(board)
            action = actions[np.random.randint(len(actions))]
            fallback = False

            policy_target = np.zeros((BOARD_SIZE, BOARD_SIZE))
            for r, c in action:
                policy_target[r, c] = 1.0

            data.append((board.copy(), player, policy_target, step))

            # 执行动作
            for r, c in action:
                board[r, c] = player

            player = 3 - player
            continue
        else:
            root = Node(board, player)
            # 批量 MCTS
            mcts_batch(root, net)
            pi = extract_policy(root)
            if not pi:
                break
            policy_target = policy_target_from_visits(pi)
            if policy_target.sum() <= 0:
                break
            action = select_action_from_visits(pi)
            if action is None:
                break


        # 记录数据
        data.append((board.copy(), player, policy_target, step))

        # 执行动作
        for r, c in action:
            board[r, c] = player

        # 检查胜负
        winner = check_winner(board, player)
        if winner != 0:
            return data, winner, step + 1

        # 切换玩家 & 更新根节点
        player = 3 - player
        # 尝试把上一步选中的动作对应的子节点作为新根
        root = root.children.get(tuple(map(tuple, action)), Node(board, player))

    return data, 0, len(data)


def compile_model(model, policy_loss_weight=None, value_loss_weight=None):
    policy_loss_weight = POLICY_LOSS_WEIGHT if policy_loss_weight is None else policy_loss_weight
    value_loss_weight = VALUE_LOSS_WEIGHT if value_loss_weight is None else value_loss_weight
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE, clipnorm=1.0),
        loss={"policy": "categorical_crossentropy", "value": "mse"},
        loss_weights={"policy": policy_loss_weight, "value": value_loss_weight}
    )
    return model


def model_has_finite_weights(model):
    for weight in model.weights:
        arr = weight.numpy()
        if not np.all(np.isfinite(arr)):
            print(f"[ERROR] Non-finite weight detected: {weight.name}")
            return False
    return True


def get_policy_diagnostics(model):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    x = board_to_features(board, 1)[None]
    policy, value = model.predict(x, verbose=0)
    p = policy[0]
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(value)):
        return None
    top = int(np.argmax(p))
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "top_row": int(top // BOARD_SIZE),
        "top_col": int(top % BOARD_SIZE),
        "top_p": float(p[top]),
        "entropy": entropy,
        "value": float(value[0][0])
    }


def print_policy_diagnostics(model, label):
    diag = get_policy_diagnostics(model)
    if diag is None:
        print(f"{label} policy: non-finite prediction detected")
        return False
    print(
        f"{label} policy: top=({diag['top_row']},{diag['top_col']}), "
        f"top_p={diag['top_p']:.6f}, entropy={diag['entropy']:.4f}, value={diag['value']:.4f}"
    )
    return True


def tactical_validation_cases():
    cases = []
    starts = {
        (0, 1): (7, 5),
        (1, 0): (5, 7),
        (1, 1): (5, 5),
        (1, -1): (5, 9),
    }
    for player in (1, 2):
        for threat_player in (player, 3 - player):
            for direction, start in starts.items():
                for run_len in (5, 4, 3):
                    case = open_run_case(player, threat_player, start, direction, run_len)
                    if case is not None:
                        cases.append(case)

            # Broken threats inside a six-point segment, e.g. XX_XXX or X_XX_X.
            segments = [
                [(7, c) for c in range(4, 10)],
                [(r, 7) for r in range(4, 10)],
                [(4 + i, 4 + i) for i in range(6)],
                [(4 + i, 10 - i) for i in range(6)],
            ]
            for segment in segments:
                for stones_needed in (1, 2, 3):
                    for empty_indices in combinations(range(6), stones_needed):
                        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
                        targets = []
                        empty_set = set(empty_indices)
                        for i, (r, c) in enumerate(segment):
                            if i in empty_set:
                                targets.append((r, c))
                            else:
                                board[r, c] = threat_player
                        cases.append((board, player, targets))
    return cases


def validate_tactical_response(model, label="Tactical validation"):
    failures = []
    cases = tactical_validation_cases()
    for idx, (board, player, targets) in enumerate(cases, start=1):
        x = board_to_features(board, player)[None].astype(np.float32)
        policy, _ = model.predict(x, verbose=0)
        policy_2d = policy[0].reshape(BOARD_SIZE, BOARD_SIZE)
        valid_moves = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r, c] == 0]
        ranked = sorted(valid_moves, key=lambda p: float(policy_2d[p[0], p[1]]), reverse=True)
        chosen = ranked[:min(2, len(ranked))]
        stones_needed = len(targets)
        threat_players = sorted(set(int(v) for v in board.flatten() if v != 0))
        threat_player = threat_players[0] if threat_players else player
        acceptable = set(tactical_points_for_player(board, threat_player, stones_needed=stones_needed))
        if not acceptable:
            acceptable = set(targets)
        target_hits = len(acceptable & set(chosen))
        required_hits = 1 if stones_needed == 1 else min(2, len(acceptable))
        if target_hits < required_hits:
            failures.append((idx, player, sorted(acceptable), chosen, ranked[:8]))

    passed = len(cases) - len(failures)
    pass_rate = passed / max(1, len(cases))
    print(f"{label}: passed {passed}/{len(cases)} critical three/four/five-line cases")
    for idx, player, targets, chosen, top8 in failures[:8]:
        print(
            f"  fail case={idx}, player={player}, targets={targets}, "
            f"chosen={chosen}, top8={top8}"
        )
    if failures and pass_rate >= TACTICAL_GATE_PASS_RATE:
        print(f"{label}: accepted with pass_rate={pass_rate:.4f} >= {TACTICAL_GATE_PASS_RATE:.4f}")
        return True
    return not failures


def training_arrays_are_finite(*arrays):
    for arr in arrays:
        if not np.all(np.isfinite(arr)):
            return False
    return True


def update_replay_buffer(x_new, yp_new, yv_new):
    if os.path.exists(REPLAY_BUFFER_PATH):
        try:
            old = np.load(REPLAY_BUFFER_PATH)
            x = np.concatenate([old["x"], x_new], axis=0)
            yp = np.concatenate([old["yp"], yp_new], axis=0)
            yv = np.concatenate([old["yv"], yv_new], axis=0)
        except Exception:
            x, yp, yv = x_new, yp_new, yv_new
    else:
        x, yp, yv = x_new, yp_new, yv_new

    if len(x) > REPLAY_BUFFER_SIZE:
        idx = np.random.choice(len(x), size=REPLAY_BUFFER_SIZE, replace=False)
        x, yp, yv = x[idx], yp[idx], yv[idx]

    np.savez_compressed(REPLAY_BUFFER_PATH, x=x, yp=yp, yv=yv)
    return x, yp, yv


def load_training_meta(base_games=69):
    if os.path.exists(TRAINING_META_PATH):
        try:
            with open(TRAINING_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    meta.setdefault("base_games", int(base_games))
    meta.setdefault("total_self_play_games", int(base_games))
    meta.setdefault("successful_runs", 0)
    meta.setdefault("history", [])
    return meta


def save_training_meta(meta):
    with open(TRAINING_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# =========================
# 训练循环
# =========================
def train(num_games=None, epochs=None, iterations=None, base_games=69):
    num_games = num_games or NUM_SELF_PLAY_GAMES
    epochs = epochs or EPOCHS_PER_ITER
    iterations = iterations or NUM_ITERATIONS
    meta = load_training_meta(base_games)

    if os.path.exists(MODEL_PATH):
        net = object.__new__(PolicyValueNet)
        net.model = load_model(MODEL_PATH, compile=False)
        compile_model(net.model)
        print(f"[OK] 已加载完整模型: {MODEL_PATH}")
    else:
        net = PolicyValueNet()
        compile_model(net.model)

    print_policy_diagnostics(net.model, "Before training")

    for it in trange(iterations, desc="Training iterations"):
        X_list, Yp_list, Yv_list = [], [], []
        winners = []

        # 生成多局 self-play 游戏
        for g in trange(num_games, desc="Self-play games", leave=False):
            game_data, winner, total_steps = self_play_batch(net)  # 批量 self-play
            winners.append(winner)
            print(f"Game {g+1}: winner={winner}, steps={total_steps}")

            for board, player, pi, step in game_data:
                # 特征
                X_list.append(board_to_features(board, player))
                Yp_list.append(pi.flatten())

                # 渐进式价值标签
                if winner == 0:
                    value = 0.0
                else:
                    value = 1.0 if player == winner else -1.0
                Yv_list.append(value)

        # 转 numpy
        X_array = np.array(X_list, dtype=np.float32)
        Yp_array = np.array(Yp_list, dtype=np.float32)
        Yv_array = np.array(Yv_list, dtype=np.float32)

        if not training_arrays_are_finite(X_array, Yp_array, Yv_array):
            print("[ERROR] Training data contains NaN/Inf. Stop before fit.")
            return

        X_array, Yp_array, Yv_array = update_replay_buffer(X_array, Yp_array, Yv_array)
        if not training_arrays_are_finite(X_array, Yp_array, Yv_array):
            print("[ERROR] Replay buffer contains NaN/Inf. Stop before fit.")
            return
        print(f"  replay samples: {len(X_array)}")

        print(f"\nIteration {it+1} value stats:")
        print(f"  mean: {Yv_array.mean():.4f}, std: {Yv_array.std():.4f}, min: {Yv_array.min():.4f}, max: {Yv_array.max():.4f}")
        print(f"  policy target sum: min={Yp_array.sum(axis=1).min():.4f}, max={Yp_array.sum(axis=1).max():.4f}")

        if os.path.exists(MODEL_PATH):
            backup_path = f"{MODEL_PATH}.pre_iter_{it+1}.bak"
            shutil.copy2(MODEL_PATH, backup_path)
            print(f"[OK] Backup saved: {backup_path}")

        # 训练模型
        net.model.fit(
            X_array,
            {"policy": Yp_array, "value": Yv_array},
            batch_size=TRAIN_BATCH,
            epochs=epochs,
            verbose=1,
            callbacks=[keras.callbacks.TerminateOnNaN()]
        )

        tactical = generate_tactical_samples(TACTICAL_SAMPLES_PER_ITER)
        if tactical is not None:
            tx, typ, tyv = tactical
            print(f"  tactical reinforcement samples: {len(tx)}, epochs: {TACTICAL_EPOCHS_PER_ITER}")
            net.model.fit(
                tx,
                {"policy": typ, "value": tyv},
                batch_size=TRAIN_BATCH,
                epochs=TACTICAL_EPOCHS_PER_ITER,
                verbose=1,
                callbacks=[keras.callbacks.TerminateOnNaN()]
            )

        if not model_has_finite_weights(net.model):
            print("[ERROR] Model became NaN/Inf after fit. Refuse to overwrite main model.")
            return

        if not print_policy_diagnostics(net.model, f"After iteration {it+1}"):
            print("[ERROR] Model prediction is NaN/Inf after fit. Refuse to overwrite main model.")
            return

        # 保存模型
        if not validate_tactical_response(net.model, f"After iteration {it+1} tactical gate"):
            print("[ERROR] Pure policy still misses critical four/five-line responses. Refuse to overwrite main model.")
            return

        net.model.save(MODEL_PATH)
        diag = get_policy_diagnostics(net.model) or {}
        meta["total_self_play_games"] = int(meta.get("total_self_play_games", base_games)) + int(num_games)
        meta["successful_runs"] = int(meta.get("successful_runs", 0)) + 1
        meta["last_train_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["last_model_path"] = MODEL_PATH
        meta["last_run"] = {
            "num_games": int(num_games),
            "epochs": int(epochs),
            "iteration": int(it + 1),
            "iterations_requested": int(iterations),
            "mcts_simulations": int(MCTS_SIMULATIONS),
            "max_moves": int(MAX_MOVES),
            "top_k": int(TOP_K),
            "learning_rate": float(LEARNING_RATE),
            "tactical_samples": int(TACTICAL_SAMPLES_PER_ITER),
            "tactical_epochs": int(TACTICAL_EPOCHS_PER_ITER),
            "policy_loss_weight": float(POLICY_LOSS_WEIGHT),
            "value_loss_weight": float(VALUE_LOSS_WEIGHT),
            "winners": {
                "black": int(sum(1 for w in winners if w == 1)),
                "white": int(sum(1 for w in winners if w == 2)),
                "draw": int(sum(1 for w in winners if w == 0))
            },
            "diagnostics": diag
        }
        history_item = dict(meta["last_run"])
        history_item["time"] = meta["last_train_time"]
        meta.setdefault("history", []).append(history_item)
        meta["history"] = meta["history"][-50:]
        save_training_meta(meta)
        print(f"[OK] Training meta saved: total_self_play_games={meta['total_self_play_games']}")
        print(f"[OK] Iteration {it+1} complete, model saved.")


def tactical_only_train(samples=None, epochs=None, reset_model=False):
    samples = samples or TACTICAL_SAMPLES_PER_ITER
    epochs = epochs or TACTICAL_EPOCHS_PER_ITER

    if reset_model:
        model = PolicyValueNet().model
        compile_model(model, policy_loss_weight=4.0, value_loss_weight=0.0)
        print("[OK] Created fresh model for tactical reinforcement")
    elif os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        compile_model(model, policy_loss_weight=4.0, value_loss_weight=0.0)
        print(f"[OK] Loaded model for tactical reinforcement: {MODEL_PATH}")
    else:
        model = PolicyValueNet().model
        compile_model(model, policy_loss_weight=4.0, value_loss_weight=0.0)

    if os.path.exists(MODEL_PATH):
        backup_path = f"{MODEL_PATH}.pre_tactical.bak"
        shutil.copy2(MODEL_PATH, backup_path)
        print(f"[OK] Backup saved: {backup_path}")

    x, yp, yv = generate_tactical_samples(samples)
    print(f"[CONFIG] tactical_only samples={samples}, epochs={epochs}, learning_rate={LEARNING_RATE}")
    model.fit(
        x,
        {"policy": yp, "value": yv},
        batch_size=TRAIN_BATCH,
        epochs=epochs,
        verbose=1,
        callbacks=[keras.callbacks.TerminateOnNaN()]
    )

    if not model_has_finite_weights(model):
        print("[ERROR] Model became NaN/Inf after tactical reinforcement. Refuse to save.")
        return
    if not print_policy_diagnostics(model, "After tactical-only reinforcement"):
        print("[ERROR] Model prediction is NaN/Inf after tactical reinforcement. Refuse to save.")
        return

    if not validate_tactical_response(model, "After tactical-only reinforcement tactical gate"):
        print("[ERROR] Pure policy still misses critical four/five-line responses. Refuse to save.")
        return

    model.save(MODEL_PATH)
    print(f"[OK] Tactical-only model saved: {MODEL_PATH}")


def teacher_train(samples=None, epochs=None, tactical_samples=None, tactical_epochs=None):
    samples = samples or TEACHER_SAMPLES_PER_RUN
    epochs = epochs or TEACHER_EPOCHS_PER_RUN
    tactical_samples = TACTICAL_SAMPLES_PER_ITER if tactical_samples is None else tactical_samples
    tactical_epochs = TACTICAL_EPOCHS_PER_ITER if tactical_epochs is None else tactical_epochs

    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        compile_model(model, policy_loss_weight=3.0, value_loss_weight=0.0)
        print(f"[OK] Loaded model for teacher imitation: {MODEL_PATH}")
    else:
        model = PolicyValueNet().model
        compile_model(model, policy_loss_weight=3.0, value_loss_weight=0.0)
        print("[OK] Created fresh model for teacher imitation")

    if os.path.exists(MODEL_PATH):
        backup_path = f"{MODEL_PATH}.pre_teacher.bak"
        shutil.copy2(MODEL_PATH, backup_path)
        print(f"[OK] Backup saved: {backup_path}")

    teacher = generate_teacher_samples(samples)
    if teacher is None:
        print("[ERROR] No teacher samples generated.")
        return
    x, yp, yv = teacher
    if len(x) == 0 or not training_arrays_are_finite(x, yp, yv):
        print("[ERROR] Teacher data is empty or contains NaN/Inf.")
        return

    print(
        f"[CONFIG] teacher_train samples={len(x)}, epochs={epochs}, "
        f"learning_rate={LEARNING_RATE}, tactical_samples={tactical_samples}, tactical_epochs={tactical_epochs}"
    )
    print(f"  teacher policy target sum: min={yp.sum(axis=1).min():.4f}, max={yp.sum(axis=1).max():.4f}")
    model.fit(
        x,
        {"policy": yp, "value": yv},
        batch_size=TRAIN_BATCH,
        epochs=epochs,
        verbose=1,
        callbacks=[keras.callbacks.TerminateOnNaN()]
    )

    tactical = generate_tactical_samples(tactical_samples)
    if tactical is not None and tactical_epochs > 0:
        tx, typ, tyv = tactical
        print(f"  tactical reinforcement samples: {len(tx)}, epochs: {tactical_epochs}")
        model.fit(
            tx,
            {"policy": typ, "value": tyv},
            batch_size=TRAIN_BATCH,
            epochs=tactical_epochs,
            verbose=1,
            callbacks=[keras.callbacks.TerminateOnNaN()]
        )

    if not model_has_finite_weights(model):
        print("[ERROR] Model became NaN/Inf after teacher training. Refuse to save.")
        return
    if not print_policy_diagnostics(model, "After teacher imitation"):
        print("[ERROR] Model prediction is NaN/Inf after teacher training. Refuse to save.")
        return
    if not validate_tactical_response(model, "After teacher imitation tactical gate"):
        print("[ERROR] Pure policy still misses critical four/five-line responses. Refuse to save.")
        return

    model.save(MODEL_PATH)
    print(f"[OK] Teacher imitation model saved: {MODEL_PATH}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, default=NUM_SELF_PLAY_GAMES)
    parser.add_argument("--epochs", type=int, default=EPOCHS_PER_ITER)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--mcts_simulations", type=int, default=MCTS_SIMULATIONS)
    parser.add_argument("--max_moves", type=int, default=MAX_MOVES)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--base_games", type=int, default=69)
    parser.add_argument("--temperature", type=float, default=POLICY_TEMPERATURE)
    parser.add_argument("--dirichlet_alpha", type=float, default=DIRICHLET_ALPHA)
    parser.add_argument("--dirichlet_epsilon", type=float, default=DIRICHLET_EPSILON)
    parser.add_argument("--exploration_points", type=int, default=EXPLORATION_POINTS)
    parser.add_argument("--replay_buffer_size", type=int, default=REPLAY_BUFFER_SIZE)
    parser.add_argument("--tactical_samples", type=int, default=TACTICAL_SAMPLES_PER_ITER)
    parser.add_argument("--tactical_epochs", type=int, default=TACTICAL_EPOCHS_PER_ITER)
    parser.add_argument("--tactical_only", action="store_true")
    parser.add_argument("--reset_model", action="store_true")
    parser.add_argument("--teacher_train", action="store_true")
    parser.add_argument("--teacher_samples", type=int, default=TEACHER_SAMPLES_PER_RUN)
    parser.add_argument("--teacher_epochs", type=int, default=TEACHER_EPOCHS_PER_RUN)
    args = parser.parse_args()

    MCTS_SIMULATIONS = args.mcts_simulations
    MAX_MOVES = args.max_moves
    TOP_K = args.top_k
    LEARNING_RATE = args.learning_rate
    POLICY_TEMPERATURE = args.temperature
    DIRICHLET_ALPHA = args.dirichlet_alpha
    DIRICHLET_EPSILON = args.dirichlet_epsilon
    EXPLORATION_POINTS = args.exploration_points
    REPLAY_BUFFER_SIZE = args.replay_buffer_size
    TACTICAL_SAMPLES_PER_ITER = args.tactical_samples
    TACTICAL_EPOCHS_PER_ITER = args.tactical_epochs
    TEACHER_SAMPLES_PER_RUN = args.teacher_samples
    TEACHER_EPOCHS_PER_RUN = args.teacher_epochs
    print(
        f"[CONFIG] num_games={args.num_games}, epochs={args.epochs}, iterations={args.iterations}, "
        f"mcts_simulations={MCTS_SIMULATIONS}, max_moves={MAX_MOVES}, top_k={TOP_K}, "
        f"learning_rate={LEARNING_RATE}, base_games={args.base_games}, "
        f"temperature={POLICY_TEMPERATURE}, dirichlet_epsilon={DIRICHLET_EPSILON}, "
        f"exploration_points={EXPLORATION_POINTS}, replay_buffer_size={REPLAY_BUFFER_SIZE}, "
        f"tactical_samples={TACTICAL_SAMPLES_PER_ITER}, tactical_epochs={TACTICAL_EPOCHS_PER_ITER}, "
        f"teacher_samples={TEACHER_SAMPLES_PER_RUN}, teacher_epochs={TEACHER_EPOCHS_PER_RUN}"
    )

    if args.teacher_train:
        teacher_train(
            samples=TEACHER_SAMPLES_PER_RUN,
            epochs=TEACHER_EPOCHS_PER_RUN,
            tactical_samples=TACTICAL_SAMPLES_PER_ITER,
            tactical_epochs=TACTICAL_EPOCHS_PER_ITER
        )
    elif args.tactical_only:
        tactical_only_train(
            samples=TACTICAL_SAMPLES_PER_ITER,
            epochs=TACTICAL_EPOCHS_PER_ITER,
            reset_model=args.reset_model
        )
    else:
        train(num_games=args.num_games, epochs=args.epochs, iterations=args.iterations, base_games=args.base_games)
