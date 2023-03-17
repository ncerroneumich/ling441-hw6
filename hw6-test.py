from hw6 import *
from hmm import *

def main():
    tagger = Tagger(example_model)
    tagger.reset(['dogs', 'bark', 'often'])
    tagger.build_graph()
    print_graph(tagger.nodes)

if __name__ == '__main__':
    main()
