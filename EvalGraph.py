import networkx as nx
from ThreeMensMorrisBoard import ThreeMensMorrisBoard
from GameGraph import GameGraphGenerator
from Board import Board
from BoardState import BoardState
from Player import Player
import networkx as nx
import sys
sys.setrecursionlimit(100000)

graph = nx.read_adjlist("Threegame_graph.txt")
# graph = nx.convert_node_labels_to_integers(graph, label_attribute='old')


evaluated_graph = nx.DiGraph()

values={}

for node in graph.nodes():
    values[int(node)] = False
visited = set()

def evaluate_terminal(node):
    boa = ThreeMensMorrisBoard()
    boa.board_state.from_int(node)
    winner = boa.get_winner(boa.board_state)
    if winner == Player.WHITE:
        return -1
    elif winner == Player.BLACK:
        return 1
    else:
        return 0  # remis albo błąd
    
for node in graph.nodes():
    if evaluate_terminal(int(node)) != 0:
        values[int(node)] = evaluate_terminal(int(node))
        visited.add(node)


def minimax(node, is_maximizing, path):
    node_int = int(node)

    if node in visited:
        return values[node_int]

    if node in path:

        return 0

    path.add(node)
    successors = list(graph.neighbors(node))

    if not successors:
        # Stan końcowy, ale nie był oznaczony jako terminalny (awaria)
        raise ValueError(f"Node {node} has no successors and is not terminal.")

    if is_maximizing:
        best_value = -float('inf')
        for neighbor in successors:
            eval_value = minimax(neighbor, False, path.copy())
            best_value = max(best_value, eval_value)
    else:
        best_value = float('inf')
        for neighbor in successors:
            eval_value = minimax(neighbor, True, path.copy())
            best_value = min(best_value, eval_value)

    values[node_int] = best_value
    visited.add(node)
    path.remove(node)
    return best_value

for node in graph.nodes():
    if values[int(node)] == False:
        minimax(node, True, set())

for edge in graph.edges():
    e1 = 1 if values[int(edge[0])] == 1 else 2 if values[int(edge[0])] == -1 else 0
    e2 = 1 if values[int(edge[1])] == 1 else 2 if values[int(edge[1])] == -1 else 0
    evaluated_graph.add_edge(int(edge[0])+e1, int(edge[1])+e2)


# Zapisz do pliku
nx.write_adjlist(evaluated_graph, "Threegame_graph_evaluated.txt")
