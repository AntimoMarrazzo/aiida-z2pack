from aiida import orm

def merge_dict_inputs_from_parent(cls, parent, parent_class, *input_labels, merge=True):
    def deep_update(old, new):
        for k,v in new.items():
            if isinstance(v, dict) and k in old and isinstance(old[k], dict):
                deep_update(old[k], v)
            else:
                old[k] = v  
        return old

    for label in input_labels:
        if isinstance(label, tuple):
            label_old = label[0]
            label_new = label[1]
        else:
            label_old = label_new = label
        new = {}

        if label_new in cls.inputs:
            new = getattr(cls.inputs, label_new).get_dict()
        
        old = {}
        try:
            old = _recursive_get_linked_node(parent, label_old, _get_previous_node, parent_class)
        except:
            pass
        else:
            old = old.get_dict()

        if merge:
            to_set = deep_update(old, new)
        else:
            if not new and old:
                to_set = old
            else:
                to_set = new

        setattr(cls.inputs, label_new, orm.Dict(dict=to_set))

def get_previous_node(old, node_class):
    remote = old.get_incoming(node_class=orm.RemoteData).first().node
    new    = remote.get_incoming(node_class=node_class).first().node

    return new

def recursive_get_linked_node(node, label, get_previous_node, node_class):
    res = None
    while res is None:
        try:
            res = node.get_incoming(link_label_filter=label).first().node
        except:
            node = get_previous_node(node, node_class)

    return res