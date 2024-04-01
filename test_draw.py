from graphviz import Digraph
import os

# 设置字体环境变量
os.environ["GRAPHVIZ_DOT"] = r"C:\Program Files\Graphviz\bin\dot.exe"
os.environ["DOT_FONTNAME"] = "SimSun"


def Neural_Network_Graph(input_layer, hidden_layers, output_layer, filename="demo1.gv"):
    g = Digraph('g', filename=filename)  # 定义一个有向图
    n = 0  # 所有结点的数量，用其来作为结点的名字（代号）

    g.graph_attr.update(splines="false", nodesep='1.6',
                        ranksep='15', rankdir="LR")
    # 设置下图的属性: 线类型，结点间隔，每一级的间隔

    # Input Layer
    with g.subgraph(name='cluster_input') as c:
        the_label = 'Input Layer'
        c.attr(color='white')
        for i in range(input_layer):
            n += 1
            c.node(str(n))
            c.attr(label=the_label, rank='same')
            c.node_attr.update(color="#2ecc71", style="filled",
                               fontcolor="#2ecc71", shape="circle")

    last_layer_nodes = input_layer  # 最后一层的结点数量
    nodes_up = input_layer  # 总结点数量

    # Hidden Layers

    hidden_layers_nr = len(hidden_layers)  # 隐藏层层数
    for i in range(hidden_layers_nr):
        with g.subgraph(name="cluster_" + str(i + 1)) as c:
            c.attr(color='white')
            c.attr(rank='same')
            the_label = "Hidden Layer" + str(i + 1)
            c.attr(label=the_label)
            for j in range(hidden_layers[i]):
                n += 1
                c.node(str(n), shape="circle", style="filled",
                       color="#3498db", fontcolor="#3498db")
                for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                    g.edge(str(h), str(n))  # 定义好上一层到下一层的连接线
            last_layer_nodes = hidden_layers[i]
            nodes_up += hidden_layers[i]

    # Output Layer
    with g.subgraph(name='cluster_output') as c:
        c.attr(color='white')
        c.attr(rank='same')
        for i in range(1, output_layer + 1):
            n += 1
            c.node(str(n), shape="circle", style="filled",
                   color="#e74c3c", fontcolor="#e74c3c")
            for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                g.edge(str(h), str(n))
        c.attr(label='Output Layer')
        c.node_attr.update(color="#2ecc71", style="filled",
                           fontcolor="#2ecc71", shape="circle")

    g.attr(arrowShape="none")
    g.edge_attr.update(arrowhead="none", color="#707070")
    g.render(filename, format="png", view=True)




# -------------------------------------------
input_layer = 16  # 输入层的神经元数量
hidden_layers = [32]  # 隐藏层层数和数量
output_layer = 16  # 输出层神经元数量
# -----------------------------------------------

Neural_Network_Graph(input_layer, hidden_layers, output_layer)