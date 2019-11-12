# -*- coding: utf-8 -*-

""" Various auxiliary functions used by the pw and phonons workflows """

import collections
from aiida.orm import DataFactory
from aiida.orm.calculation.inline import make_inline,optional_inline

__copyright__ = u"Copyright (c), This file is part of the AiiDA-EPFL Pro platform. For further information please visit http://www.aiida.net/. All rights reserved"
__license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file."
__version__ = "0.1.0"
__authors__ = "Nicolas Mounet, Andrea Cepellotti, Giovanni Pizzi, Gianluca Prandini."


ParameterData = DataFactory('parameter')

def find_optical_band_gap(bandsdata, number_electrons=None):
    """
    Tries to guess the optical band gap, i.e. the minimum distance between
    the 'lumo' and the 'homo'.
    This method is meant to be used only for electronic bands (not phonons)
    By default, it will try to use the occupations to guess the number of
    electrons, otherwise, it can be provided
    explicitely.
    Also, there is an implicit assumption that the kpoints grid is
    "sufficiently" dense, so that the bandsdata are not missing the
    intersection between valence and conduction band if present.
    Use this function with care!

    :param (float) number_electrons: (optional) number of electrons in the unit cell

    :note: By default, the algorithm uses the occupations array
      to guess the number of electrons and the occupied bands. This is to be
      used with care, because the occupations could be smeared so at a
      non-zero temperature, with the unwanted effect that the conduction bands
      might be occupied in an insulator.
      Prefer to pass the number_of_electrons explicitly

    :return: optical_gap, the minimum distance between homo and lumo in eV
        (it can be negative).
    """
    import numpy
    
    try:
        stored_bands = bandsdata.get_bands()
    except KeyError:
        raise KeyError("Cannot do much of a band analysis without bands")

    if len(stored_bands.shape) == 3:
        # I write the algorithm for the generic case of having both the
        # spin up and spin down array

        # put all spins on one band per kpoint
        bands = numpy.concatenate([_ for _ in stored_bands], axis=1)
    else:
        bands = stored_bands

    num_kpoints = len(bands)

    if number_electrons is None:
        # analysis on occupations to get the number of electrons
        try:
            _, stored_occupations = bandsdata.get_bands(also_occupations=True)
        except KeyError:
            raise KeyError("Cannot determine metallicity if I don't have "
                           "either fermi energy, or occupations")

        # put the occupations in the same order of bands, also in case of multiple bands
        if len(stored_occupations.shape) == 3:
            # I write the algorithm for the generic case of having both the
            # spin up and spin down array

            # put all spins on one band per kpoint
            occupations = numpy.concatenate([_ for _ in stored_occupations], axis=1)
        else:
            occupations = stored_occupations

        # now sort the bands by energy
        # Note: I am sort of assuming that I have an electronic ground state

        # sort the bands by energy, and reorder the occupations accordingly
        # since after joining the two spins, I might have unsorted stuff
        bands, occupations = [numpy.array(y) for y in zip(*[zip(*j) for j in
                                                            [sorted(zip(i[0].tolist(), i[1].tolist()),
                                                                    key=lambda x: x[0])
                                                             for i in zip(bands, occupations)]])]
        number_electrons = int(round(sum([sum(i) for i in occupations]) / num_kpoints))

    else:
        bands = numpy.sort(bands)
        number_electrons = int(number_electrons)

    # find the zero-temperature occupation per band (1 for spin-polarized
    # calculation, 2 otherwise)
    number_electrons_per_band = 4 - len(stored_bands.shape)  # 1 or 2
    # gather the energies of the homo band, for every kpoint
    homo = numpy.array([i[number_electrons / number_electrons_per_band - 1] for i in bands])  # take the nth level
    try:
        # gather the energies of the lumo band, for every kpoint
        lumo = numpy.array([i[number_electrons / number_electrons_per_band] for i in bands])  # take the n+1th level
    except IndexError:
        raise ValueError("To compute the optical band gap, "
                         "need more bands than n_band=number_electrons")

    return numpy.min(lumo-homo)

@make_inline
def get_qpoints_from_kpoints_inline(parameters,kpoints):
    """
    Build qpoints from kpoints knowing the k/q mesh ratio (in parameters)
    """
    KpointsData = DataFactory('array.kpoints')
    koverq = parameters.get_dict()['k_over_q']
    
    kpointsmesh = kpoints.get_kpoints_mesh()
    qpointsmesh = [k/koverq if k > 1 else 1 for k in kpointsmesh[0]]
    qpoints = KpointsData()
    qpoints.set_kpoints_mesh(qpointsmesh)
    
    return {'qpoints': qpoints}

def apply_argsort(array, indices, axis=-1):
    """
    Apply indices from argsort of a multidimensional array,
    to a multidimensional array
    :param array: array for which one wants to apply the indices
    :param indices: array of indices (result of a np.argsort command on
        an array with the same shape as array)
    :param axis: axis used for the sorting
    :return : the array array[indices]
    
    Example:
        indices = array.argsort(axis)
        new_array = apply_argsort(array, indices, axis=axis)
        # 'new_array' is the sorted array w.r.t. the axis 'axis'
    """
    import numpy as np
    
    i = list(np.ogrid[[slice(x) for x in array.shape]])
    i[axis] = indices
    return array[i]

def default_nested_dict():
    """
    Default nested dictionary with undefined depth
    """
    return collections.defaultdict(default_nested_dict)

def update_nested_dict(orig_dict, update_dict):
    """
    Update a nested dictionary with the content of
    another nested dictionary with the same internal structure.
    :param orig_dict: initial dictionary, to be updated
    :param update_dict: dictionary used to update it
    :return: the updated dictionary
    
    .. note:: the function modifies orig_dict itself - it becomes the same
    as the updated dictionary.
    .. note:: the dictionaries can contain ParameterData objects; 
    in update_dict, they are replaced by their dictionary content; in orig_dict,
    they are kept as ParameterData (updated, unstored, and not linked to anything)
    If two ParameterData objects are found - corresponding to the same keys,
    then an inline calculation symbolising the update of the initial 
    dictionary, is launched (for provenance keeping purposes).
    """
    from copy import deepcopy
    
    @optional_inline
    def update_nested_dict_inline(original_parameters, update_parameters):
        """
        Inline function to update a nested ParameterData with the content of
        another nested ParameterData with the same structure
        :param original_parameters: ParameterData with initial dictionary,
            to be updated.
        :param update_parameters: ParameterData with dictionary used to 
            update it.
        :return: a dictionary of the form
            {'updated_parameters': ParameterData with the updated dictionary}
        """
        def update_nested_dict_recursive(orig_dict, update_dict):
            """
            Recursive function to update a nested dictionary with the content of
            another nested dictionary with the same structure
            :param orig_dict: initial dictionary, to be updated
            :param update_dict: dictionary used to update it
            :return orig_dict: the updated dictionary
            """
            the_orig_dict = deepcopy(orig_dict)
            for k, v in update_dict.iteritems():
                if (isinstance(v, collections.Mapping) and
                    k in the_orig_dict and isinstance(the_orig_dict[k], collections.Mapping)):
                    tmp_dict = update_nested_dict_recursive(the_orig_dict[k], v)
                    the_orig_dict[k] = tmp_dict
                else:
                    the_orig_dict[k] = update_dict[k]
            return the_orig_dict
        
        orig_dict = original_parameters.get_dict()
        update_dict = update_parameters.get_dict()
        # update dictionary orig_dict
        the_orig_dict = update_nested_dict_recursive(orig_dict, update_dict)
        
        return {'updated_parameters': ParameterData(dict=the_orig_dict)}
    
    if isinstance(orig_dict,ParameterData) and isinstance(update_dict,ParameterData):
        result_dict = update_nested_dict_inline(original_parameters=orig_dict,
                                                update_parameters=update_dict,
                                                store=True)
        return result_dict['updated_parameters']
    
    is_orig_param = False
    try:
        the_orig_dict = orig_dict.get_dict()
        is_orig_param = True
    except AttributeError:
        the_orig_dict = orig_dict
    try:
        update_dict = update_dict.get_dict()
    except AttributeError:
        pass
    
    for k, v in update_dict.iteritems():
        if ((isinstance(v, collections.Mapping) or isinstance(v, ParameterData))
            and k in the_orig_dict and (isinstance(the_orig_dict[k], collections.Mapping)
                or isinstance(the_orig_dict[k], ParameterData))):
            tmp_dict = update_nested_dict(the_orig_dict[k], v)
            the_orig_dict[k] = tmp_dict
        else:
            the_orig_dict[k] = update_dict[k]
    
    if is_orig_param:
        return ParameterData(dict=the_orig_dict)
    else:
        return the_orig_dict

def set_the_set(calc,the_set):
    """
    Call ``'calc.set_[key]'`` methods using keys from 
    the additional_set dictionary, then store  calc
    :param calc: an unstored calculation
    :param the_set: a dictionary with the keys and values to be set
    :return calc: an unstored calculation
    """
    if not isinstance(the_set,dict):
        raise ValueError("set must be a dictionary, found instead a {}".format(the_set))
    try:
        calc.set(**the_set)
    except ValueError as e:
        raise ValueError("A key in params['calc_set'] does not correspond"
                         " to any calc.set_[key] method\n{}".format(e.message))
    return calc

def validate_keys(my_dict,mandatory_keys):
    """
    check that a dictionary contains all mandatory keywords.
    :param my_dict: dictionary to check and update
    :param list_mandatory_keywords: list of length-3 tuples of the form
    (keyword required, class type, its description). The description is used in case the 
    keyword is missing from mydict (prints an error message including the
    description). If type is None, it will not check the type of the value of the keyword, type can be a tuple.
    """
    # check the mandatory keyword (raise an error if not present)
    for k,the_type,description in mandatory_keys:
        if k not in my_dict:
            raise KeyError("Mandatory key '{}' is required (value: {})".format(description,k))
        if the_type is not None:
            if not isinstance(my_dict[k],the_type):
                raise TypeError("The value of '{}' should be of type(s) {}".format(k, the_type))

def wipe_all_scratch(w, results_to_save):
    """
    Wipe out all the scratch on the remote cluster used by a workflow and all 
    its subworkflows (found recursively)
    
    :param results_to_save: a list of Calculation objects that will be skipped
    :w: the workflow instance to clean
    """
    from aiida.orm.workflow import Workflow
    from aiida.orm.calculation.job import JobCalculation
    
    if not isinstance(w, Workflow):
        raise TypeError("Parameter w should be a workflow")
    try:
        if not all( [ isinstance(_,JobCalculation) for _ in results_to_save ] ):
            raise TypeError("Parameter results_to_save should be a list of calculations")
    except TypeError:
        raise TypeError("Parameter results_to_save should be a list of calculations")
    
    steps = w.dbworkflowinstance.steps.all()  
    this_calcs = JobCalculation.query(workflow_step__in=steps)
    this_wfs = Workflow.query(parent_workflow_step__in=steps)
    
    for c in this_calcs:
        if c.pk not in [_.pk for _ in results_to_save]:
            try:
                c.out.remote_folder._clean()
            except AttributeError:
                # remote folder does not exist (probably submission of calc. failed)
                pass
            except OSError:
                # work directory was already removed
                pass
    for this_wf in this_wfs:
        wipe_all_scratch(this_wf, results_to_save)

def get_pwparameterdata_from_wfparams(wf_params, 
                                      restart_mode='from_scratch',
                                      calculation = 'scf'):
    """
    Generate parameters for a pw calculation
    :param wf_params: the workflow parameters dictionary
    :param restart_mode: 'from_scratch' or 'restart'
    :param calculation: one of 'scf', 'nscf', 'bands', 'relax', 'vc-relax', etc.
    
    .. note:: wf_params['parameters'] can be a dictionary OR a 
    ParameterData object. In the latter case, the output is also a
    ParameterData object and the process is stored inside an inline calc.
    """
    import numpy as np
   
    @optional_inline
    def build_pw_input_parameters_inline(parameters,pw_parameters):
        """
        Build the parameters of a pw calculation.
        :param parameters: some additional parameters (calculation, restart_mode)
        :param pw_parameters: PW input parameters, to be updated with the
        previous parameters.
        :return: a dictionary of the form
            {'output_pw_parameters': ParameterData with full set of parameters}
        """
        params_dict = parameters.get_dict()
        calculation = params_dict['calculation']
        restart_mode = params_dict['restart_mode']
        
        pw_params_dict = pw_parameters.get_dict()
    
        try:
            pw_params_dict['CONTROL']
        except KeyError:
            pw_params_dict['CONTROL'] = {}
        
        pw_params_dict['CONTROL']['calculation'] = calculation
        pw_params_dict['CONTROL']['restart_mode'] = restart_mode
        
        if calculation in ['scf','nscf','bands']:
            for namelist in ['IONS', 'CELL']:
                pw_params_dict.pop(namelist,None)
        
        if (calculation == 'bands' and 
            'diago_full_acc' not in pw_params_dict.get('ELECTRONS',{})):
            pw_params_dict['ELECTRONS']['diago_full_acc'] = True
        
        if ( 'max_seconds' not in pw_params_dict['CONTROL'] or
             pw_params_dict['CONTROL']['max_seconds'] > 
                0.97*params_dict.get('max_wallclock_seconds',np.inf) ):
            # automatically set max_seconds in the intut if it was not set, and if it
            # was, change it if it has been set too large
            try:
                max_sec = int(params_dict['max_wallclock_seconds']*0.97)
                # 3% time less to avoid  the scheduler kills pw before it safely stops
                pw_params_dict['CONTROL']['max_seconds'] = max_sec
            except KeyError:
                pass
        
        return {'output_pw_parameters': ParameterData(dict=pw_params_dict)}
    
    parameters_dict = {'calculation': calculation,
                       'restart_mode': restart_mode}
    if ('calculation_set' in wf_params and 
        'max_wallclock_seconds' in wf_params['calculation_set']):
        parameters_dict['max_wallclock_seconds'] = wf_params['calculation_set']['max_wallclock_seconds']
    parameters = ParameterData(dict=parameters_dict)
    
    pw_parameters = wf_params['parameters']
    store = isinstance(pw_parameters,ParameterData)
    the_pw_parameters = ParameterData(dict=pw_parameters) \
                        if not store else pw_parameters
    result_dict = build_pw_input_parameters_inline(parameters=parameters,
                                            pw_parameters=the_pw_parameters,
                                            store=store)
    
    return result_dict['output_pw_parameters']

def take_out_npools_from_cmdline(cmdline):
    """
    Wipe out any indication about npools from the cmdline settings
    :param cmdline: list of strings with the cmdline options (as specified in
        pw input settings)
    :return : the new cmdline with the options about npools
    """
    return  [e for i,e in enumerate(cmdline) 
             if (e not in ('-npools','-npool','-nk') 
                 and cmdline[i-1] not in ('-npools','-npool','-nk'))]

def get_pw_calculation(wf_params, only_initialization=False, parent_calc=None,
                         parent_remote_folder=None):
    """
    Returns a stored calculation
    """
    # default max number of seconds for a calculation with only_initialization=True
    # (should be largely sufficient)
    default_max_seconds_only_init = 1800
    
    from aiida.orm.code import Code
    
    calculation = wf_params['input']['relaxation_scheme']
    
    if only_initialization:
        wf_params['calculation_set'] = update_nested_dict(
            wf_params.get('calculation_set',{}),
            {'max_wallclock_seconds': default_max_seconds_only_init,
             'resources': {'num_machines': 1}})
    
    if parent_calc is None:
        code = Code.get_from_string(wf_params["codename"])
        calc = code.new_calc()
        calc.use_structure(wf_params["structure"])
        calc.use_pseudos_from_family(wf_params["pseudo_family"])
        if parent_remote_folder is None:
            pw_parameters = get_pwparameterdata_from_wfparams(wf_params, calculation=calculation)
        else:
            # restart from a remote folder (typically, with charge density)
            pw_parameters = get_pwparameterdata_from_wfparams(wf_params, restart_mode='restart',
                                                              calculation=calculation)
            calc._set_parent_remotedata(parent_remote_folder)
    else:
        if calculation in ['bands']:
            calc = parent_calc.create_restart(force_restart=True,use_output_structure=True)
        else:
            calc = parent_calc.create_restart(force_restart=True)
        pw_parameters = get_pwparameterdata_from_wfparams(wf_params, 
                                restart_mode='restart', calculation=calculation)
    
    if 'vdw_table' in wf_params:
        calc.use_vdw_table(wf_params['vdw_table'])
    calc.use_parameters(pw_parameters)
    calc.use_kpoints(wf_params['kpoints'])
    calc = set_the_set(calc,wf_params.get('calculation_set',{}))
    
    # set the settings if present        
    try:
        settings_dict = wf_params['settings']
    except KeyError:
        try:
            settings_dict = parent_calc.inp.settings.get_dict()
            if calc.inp.parameters.get_dict()['CONTROL']['calculation'] == 'bands':
                # for bands calculation we take out the npools specification
                # from the parent settings as the number of kpoints will
                # be different
                cmdline = settings_dict.get('cmdline',[])
                the_cmdline = take_out_npools_from_cmdline(cmdline)
                if the_cmdline:
                    settings_dict['cmdline'] = the_cmdline
        except AttributeError:
            settings_dict = {}

    if calc.inp.parameters.get_dict()['CONTROL']['calculation'] == 'bands':
        settings_dict['also_bands'] = True
    if only_initialization:
        settings_dict['ONLY_INITIALIZATION'] = only_initialization
        _ = settings_dict.pop('also_bands',None)
    if settings_dict:
        settings = ParameterData(dict=settings_dict)
        calc.use_settings(settings)
    
    calc.store_all()
    return calc

@make_inline
def get_bandgap_inline(parameters,bands):
    """
    Get the band-gap from a BandsData object
    """
    from aiida.orm.data.array.bands import find_bandgap
    BandsData = DataFactory('array.bands')
    
    if not isinstance(bands,BandsData):
        raise ValueError("bands should be a BandsData object")
    
    number_electrons = parameters.get_dict()['number_of_electrons']
    
    is_insulator, band_gap = find_bandgap(bands,
                                          number_electrons=number_electrons)
    
    return {'output_parameters': ParameterData(dict={'is_insulator': is_insulator,
                                                     'band_gap': band_gap,
                                                     'band_gap_units': bands.units }) }

def get_wfs_with_parameter(parameter, wf_class='Workflow'):
    """
    Find workflows of a given class, with a given parameter (which must be a
    node)
    :param parameter: an AiiDA node
    :param wf_class: the name of the workflow class
    :return: an AiiDA query set with all workflows that have this parameter
    """
    from aiida.common.datastructures import wf_data_types
    from aiida.orm.workflow import Workflow
    try:
        from aiida.backends.djsite.db import models
    except ImportError:
        from aiida.djsite.db import models
    # Find attributes with this name
    qdata = models.DbWorkflowData.objects.filter(aiida_obj=parameter,
        data_type=wf_data_types.PARAMETER)
    # Find workflows with those attributes
    if wf_class == 'Workflow':
        qwf = Workflow.query(data__in=qdata)
    else:
        qwf = Workflow.query(module_class=wf_class,data__in=qdata)
    #q2 = wf_class.query(data__in=q1)
    # return a Django QuerySet with the resulting class instances
    return qwf.distinct().order_by('ctime')

def take_out_keys_from_dictionary(dictionary,keys):
    """
    Take out some keys from a dictionary.
    :param dictionary: a dictionary
    :param keys:a list of keys to be poped out. A '|' in the key means 
        descending into sub-dictionaries.
    Acts on dictionary itself.
    """
    for key in keys:
        if '|' not in key:
            dictionary.pop(key,None)
        else:
            d = dictionary
            while '|' in key:
                try:
                    d = d[key.split('|')[0]]
                except KeyError:
                    break
                key = "|".join(key.split('|')[1:])
            d.pop(key,None)

def get_pw_wfs_with_parameters(wf_params,also_bands=False,
                ignored_keys=['codename','group_name','band_group_name',
                'calculation_set','settings','band_calculation_set','band_settings',
                'input|automatic_parallelization','input|clean_workdir','input|max_restarts',
                'input|final_scf_remove_use_all_frac',
                'parameters|SYSTEM|use_all_frac',
                'parameters|ELECTRONS|electron_maxstep','parameters|ELECTRONS|mixing_beta',
                'parameters|ELECTRONS|mixing_mode','parameters|ELECTRONS|mixing_ndim',
                'parameters|ELECTRONS|diagonalization',
                'band_input|automatic_parallelization','band_input|clean_workdir',
                'band_parameters_update|ELECTRONS|diagonalization']):
    """
    Find all PwWorkflow already run with the same parameters.
    :param wf_params: a dictionary with all the parameters (can contain
        dictionaries, structure and kpoints)
    :param also_bands: if True, check that a band structure is also in
        the workflow results
    :param ignored_keys: list of keys of wf_params that are ignored in the 
        comparison (a '|' means descending into a sub-dictionary)
    :return: the list of workflows.
    """
    from copy import deepcopy
    from aiida.workflows.user.epfl_theos.dbimporters.utils import objects_are_equal
    the_params = deepcopy(wf_params)
    take_out_keys_from_dictionary(the_params,ignored_keys)
    structure_ref = the_params.pop('structure')
    kpoints_ref = the_params.pop('kpoints',None)
    band_kpoints_ref = the_params.pop('band_kpoints',None)
    input_pw_calc_ref = the_params.pop('pw_calculation',0)
    if kpoints_ref:
        try:
            kpoints_ref = kpoints_ref.get_kpoints_mesh()
        except AttributeError:
            kpoints_ref = kpoints_ref.get_kpoints()
    if band_kpoints_ref:
        try:
            band_kpoints_ref = band_kpoints_ref.get_kpoints_mesh()
        except AttributeError:
            band_kpoints_ref = band_kpoints_ref.get_kpoints()
    if input_pw_calc_ref:
        input_pw_calc_ref = input_pw_calc_ref.pk
    
    wfs_pw = get_wfs_with_parameter(structure_ref,'PwWorkflow')
    wfs = []
    for wf_pw in wfs_pw:
        if (('pw_calculation' in wf_pw.get_results() or 'pw_calculation' in wf_pw.get_parameters())
            and (not also_bands or 'band_structure' in wf_pw.get_results())):
            params = wf_pw.get_parameters()
            take_out_keys_from_dictionary(params,ignored_keys+['structure'])
            kpoints = params.pop('kpoints',None)
            band_kpoints = params.pop('band_kpoints',None)
            input_pw_calc = params.pop('pw_calculation',0)                
            if kpoints:
                try:
                    kpoints = kpoints.get_kpoints_mesh()
                except AttributeError:
                    kpoints = kpoints.get_kpoints()
            if band_kpoints:
                try:
                    band_kpoints = band_kpoints.get_kpoints_mesh()
                except AttributeError:
                    band_kpoints = band_kpoints.get_kpoints()
            if input_pw_calc:
                input_pw_calc = input_pw_calc.pk
            
            if (objects_are_equal(kpoints,kpoints_ref)
                and objects_are_equal(band_kpoints,band_kpoints_ref)
                and objects_are_equal(the_params,params)
                and input_pw_calc_ref==input_pw_calc):
                wfs.append(wf_pw)
    
    return wfs

def get_phonondispersion_wfs_with_parameters(wf_params,also_dispersion=False, also_pw_restart=False,
                ignored_keys=['pw_codename', 'pw_calculation_set','pw_settings',
                'pw_input|automatic_parallelization','pw_input|clean_workdir','pw_input|max_restarts',
                'pw_input|final_scf_remove_use_all_frac','pw_input|finish_with_scf',
                'pw_parameters|SYSTEM|use_all_frac',
                'pw_parameters|ELECTRONS|electron_maxstep','pw_parameters|ELECTRONS|mixing_beta',
                'pw_parameters|ELECTRONS|mixing_mode','pw_parameters|ELECTRONS|diagonalization',
                'pw_parameters|ELECTRONS|conv_thr', # conv_thr is added by me
                'ph_codename', 'ph_calculation_set','ph_settings', 'ph_input|use_qgrid_parallelization',
                'dispersion_matdyn_codename', 'dispersion_q2r_codename', 'dispersion_calculation_set',
                'dispersion_settings', 'dispersion_group_name']):
    """
    Find all PhonondispersionWorkflow already run with the same parameters or, if not found any, 
    find all PwWorkflow for the 'run_pw' step already run with the same parameters.
    :param wf_params: a dictionary with all the parameters (can contain
        dictionaries, structure and kpoints)
    :param also_dispersion: if True, check that a phonon dispersion is also in
        the workflow results
    :param ignored_keys: list of keys of wf_params that are ignored in the 
        comparison (a '|' means descending into a sub-dictionary)
    :return: Dictionary of the form:
        {'Phonondispersion': list of PhonondispersionWorkflow}, if previous Phonondispersion workflows were found
        {'Pw': list of PwWorkflow}, if previous Pw workflows were found
        {}, if no previous workflows were found
    """
    from copy import deepcopy
    from aiida.workflows.user.epfl_theos.dbimporters.utils import objects_are_equal
    the_params = deepcopy(wf_params)
    take_out_keys_from_dictionary(the_params,ignored_keys)
    structure_ref = the_params.pop('structure')
    kpoints_ref = the_params.pop('pw_kpoints',None)
    qpoints_ref = the_params.pop('ph_qpoints',None)
    input_pw_calc_ref = the_params.pop('pw_calculation',0)
    input_ph_calc_ref = the_params.pop('ph_calculation',0)
    input_ph_folder_ref = the_params.pop('ph_folder',0)
    if kpoints_ref:
        try:
            kpoints_ref = kpoints_ref.get_kpoints_mesh()
        except AttributeError:
            kpoints_ref = kpoints_ref.get_kpoints()
    if qpoints_ref:
        try:
            qpoints_ref = qpoints_ref.get_kpoints_mesh()
        except AttributeError:
            qpoints_ref = qpoints_ref.get_kpoints()
    if input_pw_calc_ref:
        input_pw_calc_ref = input_pw_calc_ref.pk
    if input_ph_calc_ref:
        input_ph_calc_ref = input_ph_calc_ref.pk  
    if input_ph_folder_ref:
        input_ph_folder_ref = input_ph_folder_ref.pk
    
    wfs_phondisp = get_wfs_with_parameter(structure_ref,'PhonondispersionWorkflow')
    wfs = []
    for wf_phondisp in wfs_phondisp:
        if (('ph_calculation' in wf_phondisp.get_results() or 'ph_calculation' in wf_phondisp.get_parameters())
            or ('ph_folder' in wf_phondisp.get_results() or 'ph_folder' in wf_phondisp.get_parameters())
            and (not also_dispersion or 'phonon_dispersion' in wf_phondisp.get_results())):
            params = wf_phondisp.get_parameters()
            take_out_keys_from_dictionary(params,ignored_keys+['structure'])
            kpoints = params.pop('pw_kpoints',None)
            qpoints = params.pop('ph_qpoints',None)
            input_pw_calc = params.pop('pw_calculation',0)       
            input_ph_calc = params.pop('ph_calculation',0)
            input_ph_folder = params.pop('ph_folder',0)         
            if kpoints:
                try:
                    kpoints = kpoints.get_kpoints_mesh()
                except AttributeError:
                    kpoints = kpoints.get_kpoints()
            if qpoints:
                try:
                    qpoints = qpoints.get_kpoints_mesh()
                except AttributeError:
                    qpoints = qpoints.get_kpoints()
            if input_pw_calc:
                input_pw_calc = input_pw_calc.pk
            
            if (objects_are_equal(kpoints,kpoints_ref)
                and objects_are_equal(qpoints,qpoints_ref)
                and objects_are_equal(the_params,params)
                and input_pw_calc_ref==input_pw_calc
                and input_ph_calc_ref==input_ph_calc
                and input_ph_folder_ref==input_ph_folder):
                wfs.append(wf_phondisp)
    if wfs:
        return {'Phonondispersion': wfs}
    else:
        if also_pw_restart:
            the_params = deepcopy(wf_params)
            pw_params = {'structure': the_params['structure'], 'pseudo_family': the_params['pseudo_family']}
            for k,v in the_params.iteritems():
                if k.startswith('pw_'):
                    new_k = k[3:] # remove 'pw_'
                    pw_params[new_k] = v
            # pw_params['input'].update({'final_scf_remove_use_all_frac': True})  # always present in the PhonondispersionWf     
            wfs = get_pw_wfs_with_parameters(pw_params,also_bands=False)
            if wfs:
                return {'Pw': wfs} 
            else:
                return {}
        else:
            return {}

def get_starting_magnetization_pw(pw_output):
    """
    From the output of a PW calculation, get the atomic magnetic moment
    per unit charge and build the corresponding restart magnetization
    to be applied to a subsequent calculation.
    :param pw_output: dictionary with the output of a PW calc.
    :return: a dictionary of the form:
        {'starting_magnetization': {specie_name_a: starting mag. for a,
                                    specie_name_b: starting mag. for b}
         'angle1' (optional, for SOC calc.): {specie_name_a: angle1 for a,
                                              specie_name_b: angle1 for b}
         'angle2' (optional, for SOC calc.): {specie_name_a: angle2 for a,
                                              specie_name_b: angle2 for b}
         }
    """
    import numpy as np

    if 'atomic_magnetic_moments' not in pw_output:
        return {}
    
    mag_moments = (np.array(pw_output['atomic_magnetic_moments'])/
                  np.array(pw_output['atomic_charges'])).tolist()
    species_name = pw_output['atomic_species_name']
    start_mag = dict([(kind_name,round(np.average([mom 
                for k,mom in zip(species_name,mag_moments)
                if k==kind_name]),3)) for kind_name in set(species_name)])
    result_dict = {'starting_magnetization': start_mag}
    
    if ('atomic_magnetic_theta' in pw_output and
        'atomic_magnetic_phi' in pw_output):
        theta = pw_output['atomic_magnetic_theta']
        phi = pw_output['atomic_magnetic_phi']
        result_dict['angle1'] = dict([(kind_name,round(np.average([th 
                for k,th in zip(species_name,theta)
                if k==kind_name]),3)) for kind_name in set(species_name)])
        result_dict['angle2'] = dict([(kind_name,round(np.average([ph 
                for k,ph in zip(species_name,phi)
                if k==kind_name]),3)) for kind_name in set(species_name)])        
    
    return result_dict

def get_from_parameterdata_or_dict(params,key,**kwargs):
    """
    Get the value corresponding to a key from an object that can be either
    a ParameterData or a dictionary.
    :param params: a dict or a ParameterData object
    :param key: a key
    :param default: a default value. If not present, and if key is not
        present in params, a KeyError is raised, as in params[key]
    :return: the corresponding value
    """
    if isinstance(params,ParameterData):
        params = params.get_dict()
    
    if 'default' in kwargs:
        return params.get(key,kwargs['default'])
    else:
        return params[key]
    
