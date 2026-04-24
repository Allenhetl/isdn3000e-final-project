from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from pettingzoo.classic import tictactoe_v3


NO_PLAYER = 255


@dataclass(frozen=True)
class GameSnapshotData:
    board: list[int]
    legal_actions: list[int]
    current_player: int
    terminated: bool
    winner: int


class TicTacToeGame:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.env = tictactoe_v3.env(render_mode=None)
        self.possible_agents: list[str] = []
        self.board: list[int] = [0] * 9
        self.winner = NO_PLAYER
        self.terminated = False
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> GameSnapshotData:
        self.env.reset(seed=self.seed if seed is None else seed)
        self.possible_agents = list(self.env.possible_agents)
        self.board = [0] * 9
        self.winner = NO_PLAYER
        self.terminated = False
        return self.snapshot()

    def snapshot(self) -> GameSnapshotData:
        if self.terminated:
            return GameSnapshotData(
                board=list(self.board),
                legal_actions=[],
                current_player=NO_PLAYER,
                terminated=True,
                winner=self.winner,
            )

        observation, _reward, termination, truncation, _info = self.env.last()
        action_mask = (
            observation["action_mask"] if isinstance(observation, dict) else np.zeros(9)
        )
        legal_actions = [
            int(index) for index, enabled in enumerate(action_mask) if int(enabled) == 1
        ]
        current_player = self.possible_agents.index(self.env.agent_selection)
        self.terminated = bool(termination or truncation)
        return GameSnapshotData(
            board=list(self.board),
            legal_actions=legal_actions,
            current_player=current_player,
            terminated=self.terminated,
            winner=self.winner,
        )

    def step(self, action: int) -> GameSnapshotData:
        current_agent = self.env.agent_selection
        player_id = self.possible_agents.index(current_agent)
        self.env.step(int(action))
        self.board[int(action)] = player_id + 1

        winner = self._check_winner()
        if winner is not None:
            self.winner = winner
            self.terminated = True
        elif all(value != 0 for value in self.board):
            self.winner = NO_PLAYER
            self.terminated = True

        return self.snapshot()

    def _check_winner(self) -> Optional[int]:
        win_lines = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        )
        for a, b, c in win_lines:
            if self.board[a] != 0 and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a] - 1
        return None
