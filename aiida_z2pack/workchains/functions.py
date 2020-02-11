import numpy as np
from scipy.spatial import KDTree
from aiida import orm
from aiida.engine import calcfunction

def recipr_base(base):
    return np.linalg.inv(base).T * 2 * np.pi

@calcfunction
def crop_kpoints(structure, kpt_data, centers, radius):
    """
    Crop a given set of k-points `kpt_data` that are within a spherical radius `r` from a set of 
    centers `centers`.

    :param structure: StructureData used to get the cell of the material
    """
    if not isinstance(structure, orm.StructureData):
        raise ValueError("Invalide type {} for parameter `structure`".format(type(structure)))
    if not isinstance(kpt_data, orm.KpointsData):
        raise ValueError("Invalide type {} for parameter `kpt_data`".format(type(kpt_data)))
    if not isinstance(centers, orm.ArrayData):
        raise ValueError("Invalide type {} for parameter `centers`".format(type(centers)))
    if not isinstance(radius, orm.Float):
        raise ValueError("Invalide type {} for parameter `radius`".format(type(radius)))
    centers = centers.get_array('centers')
    if len(centers.shape) != 2 or centers.shape[1] != 3:
        raise ValueError("Invalide shape {} for array `centers`. Expected (*,3)".format(centers.shape))
    
    r         = radius.value
    cell      = np.array(structure.cell)
    recipr    = recipr_base(cell)

    try:
        kpt_cryst = np.array(kpt_data.get_kpoints_mesh(print_list=True))
    except MemoryError:
        return orm.Bool(False)
    kpt_cart  = np.dot(kpt_cryst, recipr)

    c_cryst   = centers
    c_cart    = np.dot(c_cryst, recipr)

    kpt_cart  = KDTree(kpt_cart)
    centers   = KDTree(c_cart)

    query = kpt_cart.query_ball_tree(centers, r=r)

    where = [n for n,l in enumerate(query) if len(l)]

    new       = orm.KpointsData()
    new.set_kpoints(kpt_cryst[where])

    return new

@calcfunction
def copy_array_data(array):
    new = orm.ArrayData()

    for label, data in array.get_iterarrays():
        new.set_array(label, data)

    return new
