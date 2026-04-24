from __future__ import annotations

from dataclasses import dataclass

from geometry_msgs.msg import Pose

from ttt_game import TicTacToeGame
from ttt_interfaces.msg import GameSnapshot, PieceState, WorkspaceLayout


NO_PLAYER = GameSnapshot.NO_PLAYER


@dataclass
class RegisteredPlayer:
    name: str
    service_name: str


class MatchCoordinator:
    def __init__(self, layout: WorkspaceLayout, seed: int = 42) -> None:
        self.layout = layout
        self.seed = seed
        self.game = TicTacToeGame(seed=seed)
        self.players: list[RegisteredPlayer | None] = [None, None]
        self.status = GameSnapshot.STATUS_WAITING
        self.turn_index = 0
        self.match_id = "isdn3000e-final"
        self.message = "Waiting for player_0 and player_1 registration."
        self.pieces: dict[int, PieceState] = {
            piece.piece_id: piece for piece in layout.initial_pieces
        }
        self.snapshot_data = self.game.reset(seed=seed)

    def register_player(
        self, player_name: str, service_name: str
    ) -> tuple[bool, int, str]:
        if not player_name.strip():
            return False, NO_PLAYER, "Player name cannot be empty."
        if not service_name.strip():
            return False, NO_PLAYER, "Service name cannot be empty."
        for player in self.players:
            if player is not None and player.name == player_name:
                return (
                    False,
                    NO_PLAYER,
                    f"Player '{player_name}' is already registered.",
                )
        try:
            player_id = self.players.index(None)
        except ValueError:
            return False, NO_PLAYER, "Match is full."
        self.players[player_id] = RegisteredPlayer(
            name=player_name.strip(), service_name=service_name.strip()
        )
        if all(player is not None for player in self.players):
            self.message = "Both players registered. Waiting for explicit match start."
        return True, player_id, f"Registered as player_{player_id}."

    def ready_to_start(self) -> bool:
        return all(player is not None for player in self.players)

    def start_match(self) -> None:
        self.game.reset(seed=self.seed)
        self.snapshot_data = self.game.snapshot()
        self.turn_index = 0
        self.status = GameSnapshot.STATUS_PLAYING
        self.message = f"Match started. Current player: {self.current_player_name()}."
        self.pieces = {piece.piece_id: piece for piece in self.layout.initial_pieces}

    def current_player_name(self) -> str:
        if self.snapshot_data.current_player == NO_PLAYER:
            return ""
        player = self.players[self.snapshot_data.current_player]
        return player.name if player is not None else ""

    def current_player(self) -> RegisteredPlayer | None:
        if self.snapshot_data.current_player == NO_PLAYER:
            return None
        return self.players[self.snapshot_data.current_player]

    def snapshot_msg(self) -> GameSnapshot:
        msg = GameSnapshot()
        msg.match_id = self.match_id
        msg.turn_index = self.turn_index
        msg.status = self.status
        msg.current_player = self.snapshot_data.current_player
        msg.winner = self.snapshot_data.winner
        msg.board = list(self.snapshot_data.board)
        legal_mask = [0] * 9
        for cell_id in self.snapshot_data.legal_actions:
            legal_mask[cell_id] = 1
        msg.legal_actions = legal_mask
        msg.pieces = [self._clone_piece(piece) for piece in self.pieces.values()]
        msg.message = self.message
        return msg

    def validate_piece_choice(
        self, player_id: int, piece_id: int, cell_id: int
    ) -> tuple[bool, str]:
        if cell_id not in self.snapshot_data.legal_actions:
            return False, f"Cell {cell_id} is not legal."
        if piece_id not in self.pieces:
            return False, f"Piece {piece_id} does not exist."
        piece = self.pieces[piece_id]
        if piece.owner != player_id:
            return False, f"Piece {piece_id} does not belong to player_{player_id}."
        if not piece.available:
            return False, f"Piece {piece_id} is no longer available."
        return True, ""

    def commit_turn(self, player_id: int, piece_id: int, cell_id: int) -> None:
        piece = self.pieces[piece_id]
        piece.available = False
        piece.location = PieceState.LOCATION_BOARD
        piece.cell_id = cell_id
        piece.pose = self._clone_pose(self.layout.cell_poses[cell_id])

        self.snapshot_data = self.game.step(cell_id)
        self.turn_index += 1
        if self.snapshot_data.terminated:
            self.status = GameSnapshot.STATUS_FINISHED
            if self.snapshot_data.winner == NO_PLAYER:
                self.message = "Match finished in a draw."
            else:
                winner_name = (
                    self.players[self.snapshot_data.winner].name
                    if self.players[self.snapshot_data.winner]
                    else ""
                )
                self.message = f"Match finished. Winner: {winner_name}."
        else:
            self.status = GameSnapshot.STATUS_PLAYING
            self.message = (
                f"Turn accepted. Current player: {self.current_player_name()}."
            )

    def mark_failure(self, message: str) -> None:
        self.status = GameSnapshot.STATUS_FINISHED
        self.message = message

    @staticmethod
    def _clone_piece(piece: PieceState) -> PieceState:
        clone = PieceState()
        clone.piece_id = piece.piece_id
        clone.owner = piece.owner
        clone.available = piece.available
        clone.location = piece.location
        clone.cell_id = piece.cell_id
        clone.pose = MatchCoordinator._clone_pose(piece.pose)
        return clone

    @staticmethod
    def _clone_pose(pose: Pose) -> Pose:
        clone = Pose()
        clone.position.x = pose.position.x
        clone.position.y = pose.position.y
        clone.position.z = pose.position.z
        clone.orientation.x = pose.orientation.x
        clone.orientation.y = pose.orientation.y
        clone.orientation.z = pose.orientation.z
        clone.orientation.w = pose.orientation.w
        return clone
