# =============================================================================
# # Imports
# =============================================================================
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42 
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pydot
import graphviz
#!pip install pydot
#!pip install graphviz

# =============================================================================
# # Network instances
# =============================================================================
G00 = nx.erdos_renyi_graph(9, 0, seed=None, directed=False)
G10 = nx.erdos_renyi_graph(9, 1, seed=None, directed=False)
G05 = nx.erdos_renyi_graph(9, 0.5, seed=None, directed=False)
GA8 = nx.barabasi_albert_graph(9, 8, seed=None)
GA5 = nx.barabasi_albert_graph(9, 5, seed=None)
GA1 = nx.barabasi_albert_graph(9, 1, seed=None)

fig = plt.figure()
ax = fig.add_subplot(231)
surf = nx.draw(G00, pos = nx.circular_layout(G00),node_size=40,node_color="r")
ax.set_title('ER network ($p=0$)')
ax2 = fig.add_subplot(232)
surf2 = nx.draw(G05, pos = nx.circular_layout(G05),node_size=40,node_color="r")
ax2.set_title('ER network ($p=0.5$)')
ax3 = fig.add_subplot(233)
surf3 = nx.draw(G10, pos = nx.circular_layout(G10),node_size=40,node_color="r")
ax3.set_title('ER network ($p=1$)')
ax4 = fig.add_subplot(234)
surf4 = nx.draw(GA1,node_size=40,node_color="r")
ax4.set_title('AB network ($m=1$)')
ax5 = fig.add_subplot(235)
sur5 = nx.draw(GA5,node_size=40,node_color="r")
ax5.set_title('AB network ($m=5$)')
ax6 = fig.add_subplot(236)
surf6 = nx.draw(GA8,node_size=40,node_color="r")
ax6.set_title('AB network ($m=8$)')

fig.tight_layout() 
plt.savefig("networks_landscape.png", format="png",dpi=300)
plt.show()

# =============================================================================
# # Network features - Computing
# =============================================================================

densities = []
probabilities = []
clusters = []
components = []
connected_graph = []
n = 100
p = 0
while p < 1.01:
  probabilities.append(np.round(p,2))
  G = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
  densities.append(nx.density(G))
  clusters.append(nx.average_clustering(G))
  components.append(nx.number_connected_components(G))
  connected_graph.append(nx.is_connected(G))
  p += 0.01
  
densities_ab = []
probabilities_ab = []
clusters_ab = []
components_ab = []
connected_graph_ab = []
paths_ab = []
n = 100
p = 1
while p < 100:
  probabilities_ab.append(p)
  G = nx.barabasi_albert_graph(n, p, seed=None)
  densities_ab.append(nx.density(G))
  paths_ab.append(nx.average_shortest_path_length(G))
  clusters_ab.append(nx.average_clustering(G))
  components_ab.append(nx.number_connected_components(G))
  connected_graph_ab.append(nx.is_connected(G))
  p += 1

connected_graph_ab = connected_graph_ab * 1 #shows the transition at 0.05, accordingly to ln 100 / 100
components_div_ab = np.divide(components_ab,n)
paths_div_ab = np.divide(paths_ab,max(paths_ab))

connected_graph = connected_graph * 1 #shows the transition at 0.05, accordingly to ln 100 / 100
components_div = np.divide(components,n)

# =============================================================================
# # Network features - Plotting
# =============================================================================

fig, (ax, ax2,ax3)  = plt.subplots(1, 3, sharey=True,figsize=(11,5))
ax.plot(probabilities, densities, label = "Density", color = "black")
ax.plot(probabilities, clusters, label = "Avg. clustering coeff.", color = "blue")
ax.plot(probabilities, components_div, label = "Connected components$^{-2}$", color = "green")
ax.plot(probabilities, connected_graph, label = "Graph connected", color = "red")
ax.set_xlabel('p')
ax.set_ylabel('Value')
ax.set_title('ER network with $p \in [0,1]$')
ax2.plot(probabilities, densities, label = "Density", color = "black")
ax2.plot(probabilities, clusters, label = "Avg. clustering coeff.", color = "blue")
ax2.plot(probabilities, components_div, label = "Connected components$^{-2}$", color = "green")
ax2.plot(probabilities, connected_graph, label = "Graph connected", color = "red")
ax2.set_xlabel('p')
ax2.set_title('ER network with $p \in [0,0.06]$')
ax2.set_xlim(0,0.06)
ax3.plot(probabilities_ab, densities_ab, label = "Density", color = "black")
ax3.plot(probabilities_ab, paths_div_ab, label = "Shortest path", color = "purple")
ax3.plot(probabilities_ab, clusters_ab, label = "Clustering", color = "blue")
ax3.plot(probabilities_ab, components_div_ab, label = "Component size", color = "green")
ax3.plot(probabilities_ab, connected_graph_ab, label = "Graph connected", color = "red")
ax3.set_xlabel('m')
ax3.set_title('AB network with $m \in [0,n-1]$')
handles, labels = ax3.get_legend_handles_labels()
lg = fig.legend(handles, labels,fancybox=False, shadow=False, loc='center',ncol=3, bbox_to_anchor=(0.45, 0.04))
plt.savefig("networks_features.png", format="png",dpi=300)
plt.show()
