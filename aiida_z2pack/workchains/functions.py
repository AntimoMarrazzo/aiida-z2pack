"""Collection of calcfunctions used by the workchains."""
from __future__ import absolute_import
import numpy as np
from itertools import product
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering

from aiida import orm
from aiida.engine import calcfunction
from aiida.common.exceptions import InputValidationError
from six.moves import range
from six.moves import zip


def recipr_base(base):
    """Generate reciprocal base basis vectors.

    :param base: np.array of basis vectors as rows.

    :return: np.array of reciprocal basis vector as rows.
    """
    return np.linalg.inv(base).T * 2 * np.pi


def get_gap_array_from_PwCalc(calculation):
    """Get an array containing the difference in energy between valence and conduction bands.

    :param calculation: aiida.orm.CalcJob node of a pw calculation.
    """
    params = calculation.outputs.output_parameters

    n_el = params['number_of_electrons']
    spin = params['spin_orbit_calculation']

    cb = int(n_el) // (int(not spin) + 1)
    vb = cb - 1

    bands_data = calculation.outputs.output_band
    bands = bands_data.get_bands()

    return bands[:, cb] - bands[:, vb]


@calcfunction
def crop_kpoints(structure, kpt_data, centers, radius):
    """Crop a given set of k-points `kpt_data` that are within a spherical radius `r` from a set of centers `centers`.

    :param structure: aiida.orm.StructureData used to get the cell of the material.
    :param kpt_data: aiida.orm.KpointsData to crop.
    :param centers: aiida.orm.ArrayData containing an array named `centers`.
                    Each element of `centers` is used as the center of a spherical cropping.
    :param radius: radius of the sphere cropping.

    :return: aiida.orm.KpointsData node containing the cropped kpoints
    """
    if not isinstance(structure, orm.StructureData):
        raise InputValidationError(
            'Invalide type {} for parameter `structure`'.format(
                type(structure)))
    if not isinstance(kpt_data, orm.KpointsData):
        raise InputValidationError(
            'Invalide type {} for parameter `kpt_data`'.format(type(kpt_data)))
    if not isinstance(centers, orm.ArrayData):
        raise InputValidationError(
            'Invalide type {} for parameter `centers`'.format(type(centers)))
    if not isinstance(radius, orm.Float):
        raise InputValidationError(
            'Invalide type {} for parameter `radius`'.format(type(radius)))
    centers = centers.get_array('centers')
    if len(centers.shape) != 2 or centers.shape[1] != 3:
        raise InputValidationError(
            'Invalide shape {} for array `centers`. Expected (*,3)'.format(
                centers.shape))

    r = radius.value
    cell = np.array(structure.cell)
    recipr = recipr_base(cell)

    try:
        kpt_cryst = np.array(kpt_data.get_kpoints_mesh(print_list=True))
    except MemoryError:
        return orm.Bool(False)
    kpt_cart = np.dot(kpt_cryst, recipr)

    c_cryst = centers
    c_cart = np.dot(c_cryst, recipr)

    kpt_cart = KDTree(kpt_cart)
    centers = KDTree(c_cart)

    query = kpt_cart.query_ball_tree(centers, r=r)

    where = [n for n, l in enumerate(query) if len(l)]

    new = orm.KpointsData()
    new.set_kpoints(kpt_cryst[where])

    return new


@calcfunction
def generate_cubic_grid(structure, centers, distance, dim):
    """Generate a cubic grids centered in `centers` of size `distance` and dimensionality `dim`.

    :param structure: aiida.orm.StructureData node  used to get the cell of the material.
    :param centers: aiida.orm.ArrayData containing an array named `centers`.
                    Each element of `centers` is used to generate a cubic grid around it.
    :param distance: aiida.orm.Float indicating the lateral size of the cubic grid.
    :param dim: aiida.orm.Int determining the dimensionality of the grid.
                e.g.: dim=1 -> 5x1x1   dim = 2 -> 5x5x1   dim = 3 -> 5x5x5

    :return: aiida.orm.KpointsData containing the generated grids.
    """
    if not isinstance(structure, orm.StructureData):
        raise InputValidationError(
            'Invalide type {} for parameter `structure`'.format(
                type(structure)))
    if not isinstance(centers, orm.ArrayData):
        raise InputValidationError(
            'Invalide type {} for parameter `centers`'.format(type(centers)))
    if not isinstance(distance, orm.Float):
        raise InputValidationError(
            'Invalide type {} for parameter `distance`'.format(type(distance)))

    npoints = 5

    centers = centers.get_array('pinned')
    dist = distance.value / (npoints - 1)
    dim = dim.value

    # yapf: disable
    l    = np.arange(-(npoints-1)//2, (npoints-1)//2 + 1) + ((npoints + 1)%2) * 0.5
    lx   = l
    ly   = l if dim > 1 else [0,]
    lz   = l if dim > 2 else [0,]
    grid = np.array(list(product(lx, ly, lz))) * dist

    res = np.empty((0,3))
    for n,c in enumerate(centers):
        new = c + grid
        if n == 0:
            attach = new
        else:
            old_tree = KDTree(res)
            new_tree = KDTree(new)

            query = new_tree.query_ball_tree(old_tree, r=dist*1.74)

            attach = np.array([new[n] for n,q in enumerate(query) if not q])

        if len(attach):
            res = np.vstack((res, attach))

    kpt = orm.KpointsData()
    kpt.set_cell_from_structure(structure)
    kpt.set_kpoints(res, cartesian=True)

    return kpt
    # yapf: enable


@calcfunction
def get_crossing_and_lowgap_points(bands_data, gap_threshold):
    """Extract the low-gap points and crossings from the output of a `bands` calculation."""
    if not isinstance(bands_data, orm.BandsData):
        raise InputValidationError(
            'Invalide type {} for parameter `bands_data`'.format(
                type(bands_data)))
    if not isinstance(gap_threshold, orm.Float):
        raise InputValidationError(
            'Invalide type {} for parameter `gap_threshold`'.format(
                type(gap_threshold)))

    calculation = bands_data.creator
    gaps = get_gap_array_from_PwCalc(calculation)
    kpt_cryst = bands_data.get_kpoints()
    kpt_cart = bands_data.get_kpoints(cartesian=True)
    gap_thr = gap_threshold.value

    try:
        kki = calculation.inputs.kpoints.creator.inputs
        last_pinned = kki.centers.get_array('pinned')
        dist = kki.distance.value
    except:
        dist = 200
        last_pinned = np.array([[0., 0., 0.]])

    centers = KDTree(last_pinned)
    kpt_tree = KDTree(kpt_cart)
    query = centers.query_ball_tree(kpt_tree, r=dist * 1.74 / 2)  #~sqrt(3) / 2

    # Limiting fermi velocity to ~ v_f[graphene] * 3
    # GAP ~< dK * 10 / (#PT - 1)
    pinned_thr = dist * 4.00

    # Limiting number of new points per lowgap center based on distance between points
    lim = max(-5 // np.log10(dist), 1) if dist < 1 else 200
    if dist < 0.01:
        lim = 1
    where_pinned = []
    where_found = []
    for n, q in enumerate(query):
        q = np.array(q, dtype=np.int)

        if len(q) == 0:
            continue

        min_gap = gaps[q].min()

        # Skipping points where the gap didn't move much between iterations
        # _, i = kpt_tree.query(last_pinned[n])
        # prev_min_gap = gaps[i]
        # if min_gap / prev_min_gap > 0.95 and dist < 0.005:
        #     continue

        app = None
        scale = 2.5 if lim > 1 else 1.001
        if dist == 200:
            scale = 0.25 / min_gap
        while app is None or len(app) > lim:
            app = np.where(gaps[q] < min_gap * scale)[0]
            scale *= 0.98
            if scale < 1.0001:
                app = np.where(gaps[q] < min_gap * 1.0001)[0]
                break
        where_found.extend([q[i] for i in app if gaps[q[i]] <= gap_thr])
        where_pinned.extend(
            [q[i] for i in app if gap_thr < gaps[q[i]] < pinned_thr])

    # Removing dupicates and avoid exception for empty list
    where_pinned = np.array(where_pinned, dtype=np.int)
    where_pinned = np.unique(where_pinned)
    where_found = np.array(where_found, dtype=np.int)
    where_found = np.unique(where_found)

    res = orm.ArrayData()
    res.set_array('pinned', kpt_cart[where_pinned])
    res.set_array('found', kpt_cryst[where_found])

    return res


@calcfunction
def get_el_info(params):
    """Extract the information about the number of electron and conduction and valence band indexes from the output of a pw calculation."""
    if not isinstance(params, orm.Dict):
        raise InputValidationError(
            'Invalide type {} for parameter `params`'.format(type(params)))

    res = {}
    n_el = params['number_of_electrons']
    spin = params['spin_orbit_calculation']

    res['n_el'] = n_el
    res['cb'] = int(n_el) // (int(not spin) + 1)
    res['vb'] = res['cb'] - 1

    return orm.Dict(dict=res)


@calcfunction
def get_kpoint_grid_dimensionality(kpt_data):
    """Get the dimensionality of a k-point grid. If failed, assumes 3D."""
    if not isinstance(kpt_data, orm.KpointsData):
        raise InputValidationError(
            'Invalide type {} for parameter `kpt_data`'.format(type(kpt_data)))

    try:
        mesh = kpt_data.get_kpoints_mesh()[0]
    except:
        return orm.Int(3)

    dim = sum([j != 1 for j in mesh])

    return orm.Int(dim)


@calcfunction
def merge_crossing_results(**kwargs):
    """Merge the results of multiple call of `get_crossing_and_lowgap_points`."""
    structure = kwargs.pop('structure')
    cell = structure.cell
    recipr = recipr_base(cell)

    merge = np.empty((0, 3))
    for array in kwargs.values():
        found = array.get_array('found')
        merge = np.vstack((merge, found))

    new = []
    if len(merge):
        merge = np.unique(merge, axis=0)

        if len(merge) > 1:
            merge_cart = np.dot(merge, recipr)
            aggl = AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=0.005,
                                           linkage='average')
            res = aggl.fit(merge_cart)

            for n in np.unique(res.labels_):
                w = np.where(res.labels_ == n)[0]
                new.append(np.average(merge[w], axis=0))
        else:
            new = merge

    new = np.array(new)

    res = orm.ArrayData()
    res.set_array('crossings', new)

    return res


@calcfunction
def merge_chern_results(**kwargs):
    """Merge the results of multiple calls of `Z2packBaseWorkChain`."""
    crossings = kwargs.pop('crossings')
    crossings = crossings.get_array('crossings')

    cherns = []
    for param in kwargs.values():
        cherns.append(round(param['invariant']['Chern'], ndigits=5))

    res = {'crossings': crossings, 'cherns': cherns}

    return orm.Dict(dict=res)


########################################################################################################
@calcfunction
def generate_kpt_cross(structure, kpoints, step):
    """Generate a x,y,z cross around each point.

    :param structure: The StructureData to be used.
    :param kpoints: The original list of kpoints in crystal coordinates.
    :param step: The size of the step for the cross.

    :return: A KpointsData object containing all the kpt generated, including the original ones
    """
    if not isinstance(structure, orm.StructureData):
        raise InputValidationError(
            'Invalide type {} for parameter `structure`'.format(
                type(structure)))
    if not isinstance(kpoints, orm.ArrayData):
        raise InputValidationError(
            'Invalide type {} for parameter `kpoints`'.format(type(kpoints)))
    if not isinstance(step, orm.Float):
        raise InputValidationError(
            'Invalide type {} for parameter `step`'.format(type(step)))

    try:
        kpt_cryst = kpoints.get_array('kpoints')
    except:
        kpt_cryst = kpoints.get_array('crossings')
    try:
        skips = kpoints.get_array('skips')
    except:
        skips = [0] * len(kpt_cryst)

    step = step.value

    cell = structure.cell
    recipr = recipr_base(cell)
    kpts_cart = np.dot(kpt_cryst, recipr)

    # Apply cross shifts to original kpts
    shifts = np.array([
        [step, 0, 0],
        [0, step, 0],
        [0, 0, step],
        [0, 0, 0],
        [-step, 0, 0],
        [0, -step, 0],
        [0, 0, -step],
    ])
    app = np.empty((0, 3))
    for s, k in zip(skips, kpts_cart):
        if s:
            continue
        app = np.vstack((app, k + shifts))

    new_kpt = orm.KpointsData()
    new_kpt.set_cell(cell)
    new_kpt.set_kpoints(app, cartesian=True)

    return new_kpt


@calcfunction
def analyze_kpt_cross(bands_data, gap_threshold):
    """Analyze the result of kpt-cross calculation, returning the list of lowst gap and skippable points."""
    if not isinstance(bands_data, orm.BandsData):
        raise InputValidationError(
            'Invalide type {} for parameter `bands_data`'.format(
                type(bands_data)))
    if not isinstance(gap_threshold, orm.Float):
        raise InputValidationError(
            'Invalide type {} for parameter `gap_threshold`'.format(
                type(gap_threshold)))

    gap_thr = gap_threshold.value
    calculation = bands_data.creator
    gaps = get_gap_array_from_PwCalc(calculation)
    kpt_cryst = bands_data.get_kpoints()
    # kpt_cart = bands_data.get_kpoints(cartesian=True)

    res = orm.ArrayData()

    gaps = np.array(gaps).reshape(-1, 7)

    min_pos = np.argmin(gaps, axis=1)
    min_gap = np.min(gaps, axis=1)

    skips = np.where((min_pos == 3) | (min_gap < gap_thr))[0]
    app_kpt = kpt_cryst.reshape(-1, 7, 3)
    new_kpt = app_kpt[list(range(len(min_pos))), min_pos, :]

    res.set_array('skips', skips)
    res.set_array('kpoints', new_kpt)
    res.set_array('gaps', min_gap)

    return res


@calcfunction
def finilize_cross_results(cross_data, gap_threshold):
    """Analyze the final result of kpt-cross calculation, and return valid crossings."""
    if not isinstance(cross_data, orm.ArrayData):
        raise InputValidationError(
            'Invalide type {} for parameter `cross_data`'.format(
                type(cross_data)))
    if not isinstance(gap_threshold, orm.Float):
        raise InputValidationError(
            'Invalide type {} for parameter `gap_threshold`'.format(
                type(gap_threshold)))

    gap_thr = gap_threshold.value
    kpts = cross_data.get_array('kpoints')
    gaps = cross_data.get_array('gaps')

    w = np.where(gaps <= gap_thr)[0]

    crossings = kpts[w, :]

    res = orm.ArrayData()
    res.set_array('crossings', crossings)

    return res
