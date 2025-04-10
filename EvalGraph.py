from ThreeMensMorrisBoard import ThreeMensMorrisBoard
from Player import Player
import networkx as nx


# Wczytanie grafu
graph = nx.read_adjlist("Threegame_graph.txt",create_using=nx.DiGraph())

values = {}
for node in graph.nodes():
    values[int(node)] = 0


def evaluate_terminal(node):
    boa = ThreeMensMorrisBoard()
    boa.board_state.from_int(node)
    winner = boa.get_winner(boa.board_state)
    if winner == Player.WHITE:
        return 10
    elif winner == Player.BLACK:
        return -10
    else:
        return 0  # remisy lub nierozstrzygnięte

for node in graph.nodes():
    node_int = int(node)
    terminal_value = evaluate_terminal(node_int)
    if terminal_value != 0:  
        values[node_int] = terminal_value

print("Terminal nodes evaluated:",len([val for val in values.values() if val == 10 or val == -10]))
print("evaluating non terminal nodes")
graph_reversed = graph.reverse(copy=True)
for _ in range(20):
    for node in graph_reversed.nodes():
        if values[int(node)] != 0:
            for neighbor in graph_reversed.neighbors(node):
                if int(neighbor)%9==3: # gracz 1
                    values[int(neighbor)] = max(values[int(child)] for child in graph.neighbors(neighbor)
                                                if values[int(child)] != 0)
                elif int(neighbor)%9==6: #gracz 2
                    values[int(neighbor)] = min(values[int(child)] for child in graph.neighbors(neighbor)
                                                if values[int(child)] != 0)
    


graph = graph_reversed.reverse(copy=True)
evaluated_graph = nx.DiGraph()
for edge in graph.edges():
    e1 = 1 if values[int(edge[0])] == 10 else 2 if values[int(edge[0])] == -10 else 0
    e2 = 1 if values[int(edge[1])] == 10 else 2 if values[int(edge[1])] == -10 else 0
    evaluated_graph.add_edge(int(edge[0]) + e1, int(edge[1]) + e2)

print(len([val for val in values.values() if val == 10]), "winning nodes")
print(len([val for val in values.values() if val == -10]), "losing nodes")
print(len([val for val in values.values() if val == 0]), "neutral nodes")
nx.write_adjlist(evaluated_graph, "Threegame_graph_evaluated.txt")
print("Graph saved to Threegame_graph_evaluated.txt")
