import numpy as np
from itertools import product
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
def generate_cubic_grid(structure, centers, distance, dim):
    """Generate cubic grids centered in `centers` spanning 7 point per dimension."""
    if not isinstance(structure, orm.StructureData):
        raise ValueError("Invalide type {} for parameter `structure`".format(type(structure)))
    if not isinstance(centers, orm.ArrayData):
        raise ValueError("Invalide type {} for parameter `centers`".format(type(centers)))
    if not isinstance(distance, orm.Float):
        raise ValueError("Invalide type {} for parameter `distance`".format(type(distance)))

    npoints = 7

    cell     = structure.cell
    centers  = centers.get_array('pinned')
    distance = distance.value / (npoints-1)
    dim      = dim.value
    recipr   = recipr_base(cell)

    centers  = np.dot(centers, recipr)

    l    = np.arange(-npoints//2, (npoints-1)//2 + 1) + ((npoints + 1)%2) * 0.5
    lx   = l
    ly   = l if dim > 1 else [0,]
    lz   = l if dim > 2 else [0,]
    grid = np.array(list(product(lx, ly, lz))) * distance

    res = np.empty((0,3))
    for n,c in enumerate(centers):
        new = c + grid
        if n == 0:
            attach = new
        else:
            old_tree = KDTree(res)
            new_tree = KDTree(new)

            query = new_tree.query_ball_tree(old_tree, r=distance)

            attach = np.array([new[n] for n,q in enumerate(query) if not q])

        if len(attach):
            res = np.vstack((res, attach))

    kpt = orm.KpointsData()
    kpt.set_cell_from_structure(structure)
    kpt.set_kpoints(res, cartesian=True)

    return kpt

@calcfunction
def get_crossing_and_lowgap_points(bands_data, el_info, gap_threshold, last):
    if not isinstance(bands_data, orm.BandsData):
        raise ValueError("Invalide type {} for parameter `bands_data`".format(type(bands_data)))
    if not isinstance(el_info, orm.Dict):
        raise ValueError("Invalide type {} for parameter `el_info`".format(type(el_info)))
    if not isinstance(gap_threshold, orm.Float):
        raise ValueError("Invalide type {} for parameter `gap_threshold`".format(type(gap_threshold)))
    if not isinstance(last, orm.ArrayData):
        raise ValueError("Invalide type {} for parameter `last`".format(type(last)))

    info = el_info.get_dict()
    
    bands   = bands_data.get_bands()
    kpoints = bands_data.get_kpoints()
    cell    = bands_data.cell
    recipr  = recipr_base(cell)
    kpt_c   = np.dot(kpoints, recipr)
    vb      = info['vb']
    cb      = info['cb']
    gap_thr = gap_threshold.value
    gaps    = bands[:,cb] - bands[:,vb]

    if not 'pinned' in last.get_arraynames():
        min_gap = gaps.min()
        pinned_thr = min(min_gap * 2.0, 0.1)

        where_pinned = np.where((gap_thr < gaps) & (gaps <= pinned_thr))
        where_found  = np.where(gaps <= gap_thr)
    else:
        where_found = np.where(gaps <= gap_thr)

        last_pinned = last.get_array('pinned')
        last_pinned = np.dot(last_pinned, recipr)
        wg = np.where(gaps > gap_thr)[0]
        app_g = gaps[wg]

        centers = KDTree(last_pinned)
        kpt     = KDTree(kpt_c[wg])
        dist    = np.sum((kpt_c[0] - kpt_c[1])**2)**.5
        query   = centers.query_ball_tree(kpt, r=dist*1.74)

        where_pinned = []
        for q in query:
            app = app_g[q].argmin()
            where_pinned.append(wg[q[app]])

    res = orm.ArrayData()
    res.set_array('pinned', kpoints[where_pinned])
    res.set_array('found', kpoints[where_found])

    return res

@calcfunction
def get_el_info(params):
    if not isinstance(params, orm.Dict):
        raise ValueError("Invalide type {} for parameter `params`".format(type(params)))

    res = {}
    n_el = params['number_of_electrons']
    spin = params['spin_orbit_calculation']

    res['n_el'] = n_el
    res['cb']   = int(n_el) // (int(not spin) + 1)
    res['vb']   = res['cb'] - 1

    return orm.Dict(dict=res)

@calcfunction
def get_kpoint_grid_dimensionality(kpt_data):
    if not isinstance(kpt_data, orm.KpointsData):
        raise ValueError("Invalide type {} for parameter `kpt_data`".format(type(kpt_data)))

    mesh = kpt_data.get_kpoints_mesh()[0]

    dim = sum([j != 1 for j in mesh])
   
    return orm.Int(dim)


@calcfunction
def merge_crossing_results(**kwargs):
    merge = np.empty((0,3))
    for array in kwargs.values():
        found = array.get_array('found')
        merge = np.vstack((merge, found))

    if len(merge):
        merge = np.unique(merge, axis=0)

    res = orm.ArrayData()
    res.set_array('crossings', merge)
    
    return res

