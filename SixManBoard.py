from Board import Board
import networkit as nk
from networkit import vizbridges
import plotly.io as pio
import networkx as nx
import matplotlib.pyplot as plt

from Maps import SIX_MEN_GRAPH,SIX_MEN_MILLS

class SixMensMorrisBoard(Board):  
    BOARD_SIZE = 16
    PIECES_PER_PLAYER = 6

    def __init__(self):
        graph=SIX_MEN_GRAPH
        mills=SIX_MEN_MILLS
        super().__init__(board_graph=graph, pieces_per_player=self.PIECES_PER_PLAYER, board_mills=mills)

 

    def print_map_3d(self):
        btwn = nk.centrality.Betweenness(self.graph)
        btwn.run()
        widget = nk.vizbridges.widgetFromGraph(self.graph, dimension=nk.vizbridges.Dimension.Two, nodeScores=btwn.scores())
        pio.write_html(widget, "graph_visualization.html",)
        print("Mapa zapisana w pliku graph_visualization.html")


    def print_map_2d(self):
        print("0-------1-------2")
        print("|       |       |")
        print("|   3---4---5   |")
        print("|   |       |   |") 
        print("6---7       8---9")
        print("|   |       |   |")
        print("|   10--11--12  |")
        print("|       |       |")
        print("13------14------15")
        
        G_nx = nx.Graph()
        for u, v in self.graph.iterEdges():
            G_nx.add_edge(u, v)
        plt.figure(figsize=(6,6))
        nx.draw(G_nx, with_labels=True, node_color="lightblue", edge_color="gray", node_size=700, font_size=12)
        plt.title("Graph Visualization (2D)")
        plt.show()

map = SixMensMorrisBoard()
map.print_map_2d()