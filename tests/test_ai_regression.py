import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import Connect6Board, Connect6DeepLearningAI, Connect6PureModelAI


def make_board(stones, current=2):
    board = Connect6Board(15)
    board.is_first_turn = False
    board.current_player = current
    for r, c, player in stones:
        board.board[r, c] = player
    return board


def assert_contains_all(name, actual, required):
    missing = [move for move in required if move not in actual]
    if missing:
        raise AssertionError(f"{name}: missing {missing}, actual={actual}")


def run_enhanced():
    ai = Connect6DeepLearningAI(2, 15)

    board = make_board([(7, c, 1) for c in range(5, 10)], current=2)
    assert_contains_all("block horizontal five", ai.get_best_moves(board), [(7, 4), (7, 10)])

    board = make_board([(r, 4, 2) for r in range(4, 9)], current=2)
    assert_contains_all("finish vertical five", ai.get_best_moves(board), [(3, 4), (9, 4)])

    board = make_board([(4 + i, 4 + i, 1) for i in range(5)], current=2)
    assert_contains_all("block diagonal five", ai.get_best_moves(board), [(3, 3), (9, 9)])

    board = make_board([(4 + i, 10 - i, 2) for i in range(5)], current=2)
    assert_contains_all("finish anti diagonal five", ai.get_best_moves(board), [(3, 11), (9, 5)])


def run_raw_model_smoke():
    ai = Connect6PureModelAI(2, 15)
    board = make_board([(7, c, 1) for c in range(5, 10)], current=2)
    moves = ai.get_best_moves(board)
    if ai.last_decision != "raw_policy":
        raise AssertionError(f"pure model should use raw_policy, got {ai.last_decision}")
    if len(moves) != board.get_expected_move_count():
        raise AssertionError(f"pure model returned wrong move count: {moves}")


def main():
    run_enhanced()
    print("OK Connect6DeepLearningAI")
    run_raw_model_smoke()
    print("OK Connect6PureModelAI raw policy")


if __name__ == "__main__":
    main()
