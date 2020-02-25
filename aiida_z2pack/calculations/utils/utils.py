from aiida import orm
from aiida.plugins import CalculationFactory

PwCalculation = CalculationFactory('quantumespresso.pw')

def deep_update(old, new):
    """Update nested dictionaries"""
    for k,v in new.items():
        if isinstance(v, dict) and k in old and isinstance(old[k], dict):
            deep_update(old[k], v)
        else:
            old[k] = v  
    return old


def merge_dict_input_to_root(cls, *input_labels):
    """
    Find input from all parents calculations and merge them.
    The latest input takes precedence over the older ones.
    :param *input_labels: optional args can be either the link_label or a list/tuple of possible link_labels.
                          In the case of a list, the first label should be the one to be refered in the current
                          calculation. All other links are possible names to be merged from parent calculations.
                          e.g.: Z2PackCalculations -> 'pw_parameters', PwCalculations -> 'parameters',
                                to merge the two in a Z2PackCalculation  ('pw_parameters', 'parameters')
    """
    res = {}
    node = cls.inputs.parent_folder.creator
    while True:
        for label in input_labels:
            if not isinstance(label, (tuple,list)):
                label = [label]

            base = label[0]

            if base in res:
                new = res[base]
            elif base in cls.inputs:
                new = getattr(cls.inputs, base).get_dict()
            else:
                new = {}

            old = {}
            for l in label:
                if l in node.inputs:
                    old = getattr(node.inputs, l).get_dict()
                    break

            res[base] = deep_update(old, new)

        try:
            node = get_previous_node(node, orm.CalcJobNode)
        except:
            break

    for label, dct in res.items():
        setattr(cls.inputs, label, orm.Dict(dict=dct))

def get_previous_node(old, node_class):
    """
    Function to get the previous CalcJob in a chain of CalcJobs linked by their RemoteData
    :param old: Starting node from which to walk back on the chain
    :param node_class: Subclass of CalcJob in the chain
    """
    remote = old.get_incoming(node_class=orm.RemoteData).first().node
    new    = remote.get_incoming(node_class=node_class).first().node

    return new

def recursive_get_linked_node(node, label, node_class):
    """
    Find the first parent node with a given link_label, and return the node associated to that link.

    :param node: Starting node from which to walk back on the chain
    :param label: Link_label to be searched at each node of the chain
    :param node_class: Subclass of CalcJob in the chain
    """
    res = None
    while res is None:
        try:
            res = node.get_incoming(link_label_filter=label).first().node
        except:
            node = get_previous_node(node, node_class)

    return res

def get_root_parent(cls, node_class):
    """
    Get the topmost node in a chain of CalcJobs
    """
    parent = cls.inputs.parent_folder
    calc   = parent.get_incoming(node_class=node_class).first().node

    while True:
        try:
            calc = get_previous_node(calc, node_class)
        except:
            break

    return calc