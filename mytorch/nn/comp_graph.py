from graphviz import Digraph


class CompGraphVisualizer:
    """
        Parent class for ForwardGraphVisualizer and BackwardGraphVisualizer
    """
    def __init__(self):
        self.nodes, self.edges = set(), set()
    
    def visualize(self, root, rankdir="LR"):
        """
            Builds the computational graph and displays it.
            Args:
                root (Tensor): node to start building the trace.
                rankdir (str): graph organization
        """
        self._build_trace(root)
        graph = self._build_graph(rankdir=rankdir)
        return graph
    
    def _build_trace(self, node):
        """Performs a recursive depth-first search over the computational graph."""
        if node not in self.nodes:
            if node:
                self.nodes.add(node)
            for child in node.children:
                if type(child).__name__ == "Tensor":
                    self.edges.add((child, node))
                    self._build_trace(child)
    
    def _build_graph(self):
        raise NotImplementedError("build_graph() must be implemented for subclasses!")


class ForwardGraphVisualizer(CompGraphVisualizer):
    def __init__(self):
        super().__init__()
    
    def _build_graph(self, rankdir:str='TB'):
        r"""
            Plots the forward graph
            Args:
                rankdir (str): TB (top to bottom graph) | LR (left to right)

            ..note: A node is uniquely identified with the id() function which guarantees
                    a unique number for every Python object.
        """
        assert rankdir in ['LR', 'TB'], f"Unexpected rankdir argument (TB, LR available). Got {rankdir}."
        graph = Digraph(format='png', graph_attr={'rankdir': rankdir},node_attr={'color': 'cornflowerblue', 'style': 'filled', 'shape' : 'oval','fixedsize' :'false'})
        
        for n in self.nodes:
            graph.node(name=str(id(n)), label = f"{n.name}\n{n.shape}" if n.name else f"{n.shape}", fontsize="10pt")
            if n.op:
                graph.node(name=str(id(n)) + n.op, label=n.op,shape = 'plaintext', fontsize="10pt",style = '')
                graph.attr('edge',color = 'red',arrowhead="vee")
                graph.edge(str(id(n)) + n.op, str(id(n)))
        
        for n1, n2 in self.edges:

            if n2.op:
                graph.edge(str(id(n1)), str(id(n2)) + n2.op)
            else:
                graph.edge(str(id(n1)), str(id(n2)))
        
        return graph


class BackwardGraphVisualizer(CompGraphVisualizer):
    def __init__(self):
        super().__init__()
    
    def _build_graph(self, rankdir:str='LR'):
        r"""  
            Plots the backward graph

            Args:
                rankdir (str): TB (top to bottom graph) | LR (left to right)

            ..note: A node is uniquely identified with the id() function which guarantees
                    a unique number for every Python object.
        """
        assert rankdir in ['LR', 'TB'], f"Unexpected rankdir argument (TB, LR available). Got {rankdir}."
        graph = Digraph(format='png', graph_attr={'rankdir': rankdir},node_attr={'color': 'darkolivegreen1', 'style': 'filled', 'shape' : 'oval','fixedsize' :'false'})
        
        for n in self.nodes:
            if not n.grad_fn:
                graph.node(name=str(id(n)), label=f"{n.name}\n{n.shape}\nAccumulateGrad" if n.name else f"{n.shape}\nAccumulateGrad", fontsize="10pt")
            else:
                function_name = n.grad_fn.function_name 
                graph.node(name=str(id(n)), label=f"{n.shape}", fontsize="10pt")
                graph.node(name=str(id(n)) + function_name , label= function_name + " backwrd",shape = 'plaintext',fontsize="10pt", style = '')
                graph.attr('edge',color = 'red',arrowhead="vee",dir="back")
                graph.edge(str(id(n)) + function_name, str(id(n)))

        for n1, n2 in self.edges:
            graph.attr('edge',color = 'red',arrowhead="vee",dir="back")
            if n2.grad_fn:
                graph.edge(str(id(n1)), str(id(n2)) + n2.grad_fn.function_name)
            else:
                graph.edge(str(id(n1)), str(id(n2)))
        
        return graph
        
