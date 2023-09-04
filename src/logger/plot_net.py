import graphviz
from model.edge_counting import EdgeCountingAutoencoder
from fuzzy_logic.fuzzy_operators import T_Norm, T_Conorm

FONT = 'Helvetica'
PENWIDTH = '15'
NODE_COLOR = '#00883A'
#EDGE_COLOR = "#232323"
EDGE_COLOR = "#626468"
NODE_TEXT_COLOR = "#FFFFFF"

T_NORM_NAME = "min"
T_CONORM_NAME = "max"

def plot_net(edge_ae: EdgeCountingAutoencoder, save_path: str):
	layers = edge_ae.layers
	layer_sizes = edge_ae.layer_sizes

	dot = graphviz.Digraph(comment='Neural Network', 
                graph_attr={'ranksep':'1', 'splines':'line', 'rankdir':'LR', 'fontname':FONT},
                node_attr={'fixedsize':'true', 'style':'filled', 'color':'none', 'fillcolor':NODE_COLOR, 
					'shape':'circle', 'width':'0.4', 'height':'0.4'},
                edge_attr={'color': EDGE_COLOR, 'arrowsize':'0.3', 'style': 'invis'})

	# nodes
	for layer_idx in range(len(layer_sizes)):
		with dot.subgraph(name='cluster_'+str(layer_idx)) as c:
			c.attr(color='transparent') 

			for a in range(layer_sizes[layer_idx]):
				if layer_idx == 0:
					c.node('l'+str(layer_idx)+str(a), f'<x<SUB>{(a - 3) % 7}</SUB><SUP>(0)</SUP>>', fontcolor=NODE_TEXT_COLOR, fontsize='12')

				elif layer_idx == len(layer_sizes)-1:
					operator_name = T_CONORM_NAME
					if edge_ae.layers[-1].operator in T_Norm:
						operator_name = T_NORM_NAME
	
					c.node('l'+str(layer_idx)+str(a), operator_name, fontcolor=NODE_TEXT_COLOR, fontsize='12')
				else:
					operator_name = T_CONORM_NAME
					if edge_ae.layers[layer_idx-1].operator in T_Norm:
						operator_name = T_NORM_NAME
	
					c.node('l'+str(layer_idx)+str(a), operator_name, fontcolor=NODE_TEXT_COLOR, fontsize='12')
	# connections
	for layer_idx, layer in enumerate(layers):
		for node_idx, edge_count in enumerate(layer.edge_type_count):
			for connection_idx, connection in enumerate(edge_count):
				if connection[1] > connection[0]:
					dot.edge('l'+str(layer_idx)+str(connection_idx), 'l'+str(layer_idx+1)+str(node_idx), style='solid')
				else:
					dot.edge('l'+str(layer_idx)+str(connection_idx), 'l'+str(layer_idx+1)+str(node_idx))
	
	dot.format = 'PNG'
	dot.render(save_path)
	  