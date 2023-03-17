from hmm import Model, Node, print_graph


class Tagger (object):

    model = None
    words = None
    nodes = None

    def __init__(self, m):
        self.model = m

    def reset(self, w):
        self.words = w
        self.nodes = []

    def new_node(self, i, word, pos, prev_nodes):
        index = len(self.nodes)
        node = Node(index, i, word, pos, prev_nodes)
        self.nodes.append(node)
        return node
    
    def build_graph(self):
        """Expect words member to contain a sentence."""

        prev_nodes = []

        # Create left boundry node
        prev_nodes.append(self.new_node(-1, None, None, []))

        # Iterate through each word, use our model to get
        # possible POS and for each POS create a seperate
        # node.
        temp = []
        for i, word in enumerate(self.words):
            pos = self.model.parts(word)
            for p in pos:
                temp.append(self.new_node(i, word, p, prev_nodes))
            prev_nodes.extend(temp)
            temp = []

        self.new_node(len(self.words), None, None, prev_nodes)