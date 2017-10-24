from graphviz import Digraph
from cmd_args import parser
import torch
from torch.autograd import Variable
import getmodel


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + ', '.join(['%d' % v for v in size]) + ')'

    def add_nodes(variable):
        if variable not in seen:
            if torch.is_tensor(variable):
                dot.node(str(id(variable)), size_to_str(variable.size()), fillcolor='orange')
            elif hasattr(variable, 'variable'):
                u = variable.variable
                node_name = '%s\n %s' % (param_map[id(u)], size_to_str(u.size()))
                dot.node(str(id(variable)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(variable)), str(type(variable).__name__))
            seen.add(variable)
            if hasattr(variable, 'next_functions'):
                for u in variable.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(variable)))
                        add_nodes(u[0])
            if hasattr(variable, 'saved_tensors'):
                for t in variable.saved_tensors:
                    dot.edge(str(id(t)), str(id(variable)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def main():
    global args
    args = parser.parse_args()
    assert args.classes > 1
    print("=> creating model ...")
    print("   ARCH: {}".format(args.arch))
    print("    NET: {}".format(args.net))
    print("Classes: {}".format(args.classes))
    model = getmodel.GetModel(args)
    print(model)
    model = model.cuda()
    inputs = Variable(torch.rand(1, 3, 297, 817).cuda())
    outputs = model(inputs)
    g = make_dot(outputs, model.state_dict())
    g.view()


if __name__ == '__main__':
    main()
