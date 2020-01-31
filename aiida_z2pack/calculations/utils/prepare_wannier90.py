from six.moves import zip

import os
import copy
import numpy as np
from aiida import orm
from aiida.common import exceptions
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.calculations import _uppercase_dict, _lowercase_dict



PwCalculation = CalculationFactory('quantumespresso.pw')

def prepare_wannier90(cls, folder):
    # inputdict=inputdict_wannier90
    # def _prepare_for_submission_wannier90(cls,folder, inputdict):        
    """
    Routine, which creates the input and prepares for submission

    :param folder: a aiida.common.folders.Folder subclass where
                       the plugin should put all its files.
    :param inputdict: a dictionary with the input nodes, as they would
            be returned by get_inputdata_dict (without the orm.Code!)
    """

    cls._blocked_keywords = cls._blocked_keywords_wannier90
    cls.inputs.metadata.options.input_filename = cls._INPUT_W90_FILE_NAME
    cls.inputs.metadata.options.output_filename = cls._OUTPUT_W90_FILE_NAME
    cls.inputs.parameters = cls.inputs.wannier90_parameters
    cls.inputs.settings = cls.inputs.wannier90_settings

    # return
    ##################################################################
    # Input validation
    ##################################################################

    # Grabs parent calc information
    # parent_folder = inputdict.pop(cls.get_linkname('parent_folder'),None)
    # if not isinstance(parent_folder, orm.RemoteData):
    #     raise exceptions.InputValidationError("parent_folder is not of type "
    #                                "orm.RemoteData")


    parameters = cls.inputs.parameters

    param_dict = parameters.get_dict()
    param_dict = blocked_keyword_finder(param_dict, cls._blocked_keywords)
    param_dict = _lowercase_dict(param_dict, 'param_dict')
    # check_capitals(param_dict)

    # precode_parameters = cls.inputs.overlap_parameters
    # precode_param_dict = precode_parameters.get_dict()
    # precode_param_dict = blocked_keyword_finder(precode_param_dict,
    #                                     cls._blocked_precode_keywords)
    # check_capitals(precode_param_dict)
    # settings = cls.inputs.settings_overlap
    # settings_dict = _uppercase_dict(settings.get_dict(), dict_name='settings')

    # Tries to get the input structure
    structure = cls.inputs.structure
    # if not isinstance(structure, orm.StructureData):
    #     raise exceptions.InputValidationError("structure is not of type orm.StructureData")

    # This section handles the multicode support
    # main_code = cls.inputs.wannier90_code


    # preproc_code = cls.inputs.overlap_code

    ############################################################
    # End basic check on inputs
    ############################################################

    # Here info from the parent, for file copy settings is found
    # parent_info_dict = {}
    # parent_calc = parent_folder.get_inputs_dict()['remote_folder']
    # parent_inputs = parent_calc.get_inputs_dict()
    # wannier_parent = isinstance(parent_calc, Wannier90Calculation)
    # parent_info_dict.update({'wannier_parent':wannier_parent})
    # if parent_info_dict['wannier_parent']:
    #     # If wannier parent, check if it was INIT_ONY and if precode used
    #     parent_settings = parent_inputs.pop('settings',{})
    #     try:
    #         parent_settings = parent_settings.get_inputs_dict()
    #     except AttributeError:
    #         pass
    #     parent_init_only = parent_settings.pop('INIT_ONLY',
    #                                            cls._DEFAULT_INIT_ONLY)
    #     parent_info_dict.update({'parent_init_only':parent_init_only})
    #     parent_precode = parent_inputs.pop(
    #                         cls.get_linkname('preprocessing_code'),None)
    #     parent_info_dict.update({'parent_precode':bool(parent_precode)})
    # else:
    #     if preproc_code is None:
    #         raise exceptions.InputValidationError("You cannot continue from a non"
    #                                    " wannier calculation without a"
    #                                    " preprocess code")


    # Here info from this calculation, for file copy settings is found
    # init_only = settings_dict.pop('INIT_ONLY', cls._DEFAULT_INIT_ONLY)
    # if init_only and preproc_code is None:
    #     raise exceptions.InputValidationError(
    #         'You cannot have init_only mode set, without providing a preprocessing code')

    # prepare the main input text
    input_file_lines = []
    # from aiida.common.utils import conv_to_fortran_withlists
    for param in param_dict:
        input_file_lines.append(param + ' = ' + conv_to_fortran_withlists(param_dict[param]))

    # take projectionsdict and write to file
    # checks if spins are used, and modifies the opening line
    # projection_list = projections.get_orbitals()
    # spin_use = any([bool(projection.get_orbital_dict()['spin'])
    #                for projection in projection_list])
    # if spin_use:
    #     raise exceptions.InputValidationError("spinors are implemented but not tested"
    #                                " disable this error if you know what "
    #                                "you are doing!")
    #     projector_type = "spinor_projections"
    # else:
    #     projector_type = "projections"
    # input_file_lines.append('Begin {}'.format(projector_type))
    # for projection in projection_list:
    #     orbit_line = _print_wann_line_from_orbital(projection)
    #     input_file_lines.append(orbit_line)
    # input_file_lines.append('End {}'.format(projector_type))

    # convert the structure data
    input_file_lines.append("Begin unit_cell_cart")
    input_file_lines.append('ang')
    for vector in structure.cell:
        input_file_lines.append("{0:18.10f} {1:18.10f} {2:18.10f}".format
                                (*vector))
    input_file_lines.append('End unit_cell_cart')

    input_file_lines.append('Begin atoms_cart')
    input_file_lines.append('ang')
    wann_positions, wann_kind_names = _wann_site_format(structure.sites)
    atoms_cart = zip(wann_kind_names, wann_positions)
    for atom in atoms_cart:
        input_file_lines.append('{}  {}'.format(atom[0],atom[1]))
    input_file_lines.append('End atoms_cart')

    # convert the kpoints_path
    # try:
    #     special_points = kpoints_path.get_special_points()
    # except ModificationNotAllowed:
    #     raise exceptions.InputValidationError('kpoints_path must be kpoints with '
    #                                 'a special kpoint path already set!')

    #  TODO If someone wanted to add custom kpoint_path support do so here

    # input_file_lines.append('Begin Kpoint_Path')
    # for i in range(len(special_points[1])):
    #     point1, point2 = special_points[1][i]
    #     coord1 = special_points[0][point1]
    #     coord2 = special_points[0][point2]
    #     path_line = '{} {} {} {} '.format(point1,*coord1)
    #     path_line += ' {} {} {} {}'.format(point2,*coord2)
    #    input_file_lines.append(path_line)
    # input_file_lines.append('End Kpoint_Path')

    #  convert the kmesh
    # try:
    #     kmesh = kpoints.get_kpoints_mesh()[0]
    # except AttributeError:
    #     raise exceptions.InputValidationError('kpoints should be set with '
    #                                'set_kpoints_mesh, '
    #                                'and not set_kpoints... ')

    # mp_line = 'mp_grid = {},{},{}'.format(*kmesh)
    # input_file_lines.append(mp_line)

    # input_file_lines.append('Begin kpoints')
    # for vector in kpoints.get_kpoints_mesh(print_list=True):
    #     input_file_lines.append("{0:18.10f} {1:18.10f} {2:18.10f}"
    #                             .format(*vector))
    # input_file_lines.append('End kpoints')

    # Prints to file the main input
    input_filename = folder.get_abs_path(cls._INPUT_W90_FILE_NAME)
    with open(input_filename, 'w') as file:
        file.write( "\n".join(input_file_lines) )
        file.write( "\n" )

    # Prints the precode input file
    # if preproc_code is not None:
    #     namelist_dict = {'outdir':'.' + PwCalculation._OUTPUT_SUBFOLDER,
    #                      'prefix':PwCalculation._PREFIX,
    #                      'seedname':cls._SEEDNAME,
    #                      }
    #     for precode_param in precode_param_dict:
    #         namelist_dict.update({precode_param:
    #                                   precode_param_dict[precode_param]})
    #     #  Manually makes sure that .EIG, .MMN are not rewritten
    #     # if  parent_info_dict['wannier_parent']:
    #     #     user_mmn_setting = namelist_dict.pop('write_mmn',None)
    #     #     if user_mmn_setting:
    #     #         raise exceptions.InputValidationError("You attempt to write_mmn for a "
    #     #                                    " calculation which inherited"
    #     #                                    " from a wannier90 calc. This"
    #     #                                    " is not allowed. Either set"
    #     #                                    " write_mmn to false, or use a"
    #     #                                    " non-wannier calc as parent.")
    #     #     namelist_dict.update({'write_mmn':False})
    #     #      Add write_eig = .false. once this is available
    #     #      namelist_dict.update({})
    #     #  checks and adds UNK file
    #     #  writing UNK as a setting is obsolete
    #     #  write_unk = settings_dict.pop('WRITE_UNK',None)
    #     #  if write_unk:
    #     #      namelist_dict.update({'write_unk':True})
    #     p2w_input_dict = {'INPUTPP':namelist_dict}

    #     input_precode_filename = folder.get_abs_path(
    #         cls._INPUT_OVERLAP_FILE_NAME)
    #     with open(input_precode_filename,'w') as infile:
    #         for namelist_name in p2w_input_dict.keys():
    #             infile.write("&{0}\n".format(namelist_name))
    #             # namelist content; set to {} if not present,
    #             #  so that we leave an empty namelist
    #             namelist = p2w_input_dict.get(namelist_name, {})
    #             for k, v in sorted(namelist.items()):
    #                 infile.write(get_input_data_text(k,v))
    #             infile.write("/\n")

    ############################################################
    #  end of writing text input
    ############################################################ 

    #call to _prepare_wannier90
    # write_to_file_w90=_prepare_for_submission_wannier90(cls, folder,inputdict=inputdict) 
    # return write_to_file_w90

def _wann_site_format(structure_sites):
    '''
    Generates site locations and cell dimensions
    in a manner that can be used by the wannier90 input script
    '''
    def list2str(list_item):
        '''
        Converts an input list item into a str
        '''
        list_item = copy.deepcopy(list_item)
        if isinstance(list_item, str):
            return list_item
        else:
            return ' ' + ' '.join([str(_) for _ in list_item]) + ' '
    
    calc_positions = []
    calc_kind_names = []
    for i in range(len(structure_sites)):
        calc_positions.append(list2str(structure_sites[i].position))
        calc_kind_names.append(structure_sites[i].kind_name)
    return calc_positions, calc_kind_names

def blocked_keyword_finder(input_params, blocked_keywords):
    """
    Searches through the input_params for any blocked_keywords and
    forces the default, returns the modified input_params
    """
    import re
    for blocked in blocked_keywords:
        flag = blocked[0]
        defaultvalue = blocked[1]
        # if nl in input_params:
        # The following lines is meant to avoid putting in input the
        # parameters like celldm(*)
        stripped_inparams = [re.sub("[(0-9)]", "", _)
                             for _ in input_params.keys()]
        if flag in stripped_inparams:
            raise exceptions.InputValidationError(
                "You cannot specify explicitly the '{}' flag.".format(flag))
        if defaultvalue is not None:
            # if nl not in input_params:
            #     input_params[nl] = {}
            input_params[flag] = defaultvalue
    return input_params

# def check_capitals(input_params):
#     """
#     Goes through the input_params (which must be a dictionary) and
#     raises an exceptions.InputValidationError if any of the keys are not capitalized
#     """
#     for k in input_params:
#         if k != k.lower():
#             raise exceptions.InputValidationError(
#                 "Please make sure all keys are lower case, {} was not!".format(k))


def conv_to_fortran(val,quote_strings=True):
    """
    :param val: the value to be read and converted to a Fortran-friendly string.
    """
    # Note that bool should come before integer, because a boolean matches also
    # isinstance(...,int)
    if isinstance(val, (bool, np.bool_)):
        if val:
            val_str = '.true.'
        else:
            val_str = '.false.'
    elif isinstance(val, int):
        val_str = "{:d}".format(val)
    elif isinstance(val, float):
        val_str = ("{:18.10e}".format(val)).replace('e', 'd')
    elif isinstance(val, str):
        if quote_strings:
            val_str = "'{!s}'".format(val)
        else:
            val_str = "{!s}".format(val)
    else:
        raise ValueError("Invalid value '{}' of type '{}' passed, accepts only bools, ints, floats and strings".format(val, type(val)))

    return val_str

def conv_to_fortran_withlists(val,quote_strings=True):
    """
    Same as conv_to_fortran but with extra logic to handle lists
    :param val: the value to be read and converted to a Fortran-friendly string.
    """
    # Note that bool should come before integer, because a boolean matches also
    # isinstance(...,int)
    if (isinstance(val, (list, tuple))):
        out_list = []
        for thing in val:
            out_list.append(conv_to_fortran(thing,quote_strings=quote_strings))
        val_str = ", ".join(out_list)
        return val_str
    if (isinstance(val, bool)):
        if val:
            val_str = '.true.'
        else:
            val_str = '.false.'
    elif (isinstance(val, int)):
        val_str = "{:d}".format(val)
    elif (isinstance(val, float)):
        val_str = ("{:18.10e}".format(val)).replace('e', 'd')
    elif (isinstance(val, str)):
        if quote_strings:
            val_str = "'{!s}'".format(val)
        else:
            val_str = "{!s}".format(val)
    else:
        raise ValueError("Invalid value passed, accepts only bools, ints, "
                         "floats and strings")

    return val_str


def get_input_data_text(key, val, mapping=None):
    """
    Given a key and a value, return a string (possibly multiline for arrays)
    with the text to be added to the input file.
    :param key: the flag name
    :param val: the flag value. If it is an array, a line for each element
            is produced, with variable indexing starting from 1.
            Each value is formatted using the conv_to_fortran function.
    :param mapping: Optional parameter, must be provided if val is a dictionary.
            It maps each key of the 'val' dictionary to the corresponding
            list index. For instance, if ``key='magn'``,
            ``val = {'Fe': 0.1, 'O': 0.2}`` and ``mapping = {'Fe': 2, 'O': 1}``,
            this function will return the two lines ``magn(1) = 0.2`` and
            ``magn(2) = 0.1``. This parameter is ignored if 'val'
            is not a dictionary.
    """
    # from aiida.common.utils import conv_to_fortran
    # I don't try to do iterator=iter(val) and catch TypeError because
    # it would also match strings
    # I check first the dictionary, because it would also match hasattr(__iter__)
    if isinstance(val, dict):
        if mapping is None:
            raise ValueError("If 'val' is a dictionary, you must provide also "
                             "the 'mapping' parameter")

        # At difference with the case of a list, at the beginning list_of_strings
        # is a list of 2-tuples where the first element is the idx, and the
        # second is the actual line. This is used at the end to resort everything.
        list_of_strings = []
        for elemk, itemval in val.iteritems():
            try:
                idx = mapping[elemk]
            except KeyError:
                raise ValueError("Unable to find the key '{}' in the mapping "
                                 "dictionary".format(elemk))

            list_of_strings.append((idx, "  {0}({2}) = {1}\n".format(
                key, conv_to_fortran(itemval), idx)))

        # I first have to resort, then to remove the index from the first
        # column, finally to join the strings
        list_of_strings = zip(*sorted(list_of_strings))[1]
        return "".join(list_of_strings)
    elif hasattr(val, '__iter__'):
        # a list/array/tuple of values
        list_of_strings = [
            "  {0}({2}) = {1}\n".format(key, conv_to_fortran(itemval),
                                        idx + 1)
            for idx, itemval in enumerate(val)]
        return "".join(list_of_strings)
    else:
        # single value
        return "  {0} = {1}\n".format(key, conv_to_fortran(val))
