from abc import ABC, abstractmethod
from subprocess import check_call
from typing import List, Tuple

import networkit as nk

from BoardState import BoardState
from Move import Move
from Player import Player
from Position import Position
from GamePhase import GamePhase


class Board(ABC):
    def __init__(self, board_graph: Tuple[Tuple[int]], pieces_per_player: int, board_mills):
        self.board_size = len(board_graph)
        self.graph = nk.graph.Graph(self.board_size, directed=False) 
        self.pieces_per_player = pieces_per_player
        self.mills = board_mills
        self.initialize_connections(board_graph)


    def initialize_connections(self, board_graph):
        for v,k in enumerate(board_graph):
            for i in k:
                self.graph.addEdge(v, i)

    def get_initial_board_state(self) -> BoardState:
        return BoardState(self.board_size, self.pieces_per_player)
    
    def mill_check(self,position,mill,player):
        if position==mill[0] and player==self.board[mill[1]] and player==self.board[mill[2]]:
            return True
        if position==mill[1] and player==self.board[mill[0]] and player==self.board[mill[2]]:
            return True
        if position==mill[2] and player==self.board[mill[0]] and player==self.board[mill[1]]:
            return True
        return False
        

    def check_if_move_creates_mill(self, state: BoardState, position: int, player: Player) -> bool:
        #return 0 or 1 or 2 or 3
        mills_counter=0
        for mill in self.mills:
            if position in mill:
                if self.mill_check(self,position,mill,player):
                    mills_counter+=1
            
        return mills_counter

    def get_legal_moves(self, state: BoardState,phase:GamePhase) -> List[Position]:
        if phase == GamePhase.PLACEMENT:
            return [i for i in range(self.board_size) if state.is_position_empty(Position(i))]
        
        if phase == GamePhase.MOVEMENT:
            return NotImplemented
        if phase == GamePhase.FLYING:
            return NotImplemented


    def make_move(self, state: BoardState, move: Move, player: Player) -> BoardState:
        return NotImplemented

    def check_if_game_is_over(self, state: BoardState) -> bool:
        return (state.pieces_from_player_currently_on_board[Player.WHITE] <= 2 or state.pieces_from_player_currently_on_board[Player.BLACK] <= 2) and state.pieces_left_to_place_by_player[Player.WHITE] == 0 and state.pieces_left_to_place_by_player[Player.BLACK] == 0

    def get_winner(self, state: BoardState) -> Player:
        return Player.WHITE if state.pieces_from_player_currently_on_board[Player.WHITE] > state.pieces_from_player_currently_on_board[Player.BLACK] else Player.BLACK
    

    