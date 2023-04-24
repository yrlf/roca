import networkx as nx
from cdt.causality.graph import CAM
from cdt.data import load_dataset
data, graph = load_dataset("sachs")
obj = CAM()
output = obj.predict(data)
nx.draw_networkx(output, font_size=8)
