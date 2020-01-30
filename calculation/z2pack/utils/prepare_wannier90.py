from six.moves import zip

from aiida import orm
from aiida.common import exceptions
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.calculations import _uppercase_dict


PwCalculation = CalculationFactory('quantumespresso.pw')

def prepare_wannier90(self, folder):
    # inputdict=inputdict_wannier90
    # def _prepare_for_submission_wannier90(self,folder, inputdict):        
    """
    Routine, which creates the input and prepares for submission

    :param folder: a aiida.common.folders.Folder subclass where
                       the plugin should put all its files.
    :param inputdict: a dictionary with the input nodes, as they would
            be returned by get_inputdata_dict (without the orm.Code!)
    """
    ##################################################################
    # Input validation
    ##################################################################

    # Grabs parent calc information
    # parent_folder = inputdict.pop(self.get_linkname('parent_folder'),None)
    # if not isinstance(parent_folder, orm.RemoteData):
    #     raise exceptions.InputValidationError("parent_folder is not of type "
    #                                "orm.RemoteData")

    # Tries to get the input parameters
    try:
        # parameters = inputdict.pop(self.get_linkname('wannier90_parameters'))
        parameters = self.inputs.wannier90_parameters
    except KeyError:
        raise exceptions.InputValidationError("No parameters specified for this calculation")
    if not isinstance(parameters, orm.Dict):
        raise exceptions.InputValidationError("parameters is not of type orm.Dict")


    param_dict = parameters.get_dict()
    param_dict = blocked_keyword_finder(param_dict, self._blocked_keywords)
    check_capitals(param_dict)

    # Tries to get the precode input paramters
    try:
        # precode_parameters = inputdict.pop(self.get_linkname('overlap_parameters'))
        precode_parameters = self.inputs.wannier90_parameters
    except:
        precode_parameters = orm.Dict(dict={})
    # if not isinstance(precode_parameters, orm.Dict):
    #     raise exceptions.InputValidationError('precode_parameters is not of type orm.Dict')
    precode_param_dict = precode_parameters.get_dict()
    precode_param_dict = blocked_keyword_finder(precode_param_dict,
                                        self._blocked_precode_keywords)
    check_capitals(precode_param_dict)

    # Tries to get the input structure
    try:
        # structure = inputdict.pop(self.get_linkname('structure'))
        structure = self.inputs.structure
    except KeyError:
        raise exceptions.InputValidationError("No structure specified for this calculation")
    # if not isinstance(structure, orm.StructureData):
    #     raise exceptions.InputValidationError("structure is not of type orm.StructureData")

    # Settings can be undefined, and defaults to an empty dictionary
    # settings = inputdict.pop(self.get_linkname('settings'),None)
    try:
        settings = self.inputs.settings
    except KeyError:
        settings_dict = {}
    else:
        settings_dict = _uppercase_dict(settings.get_dict(), dict_name='settings')

    # This section handles the multicode support
    try:
        main_code = self.inputs.wannier90_code
    except:
        raise exceptions.InputValidationError("No input code found!")

    try:
        preproc_code = self.inputs.overlap_code
    except:
        pass
        # raise exceptions.InputValidationError("No input overlap code found!")

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
    #                                            self._DEFAULT_INIT_ONLY)
    #     parent_info_dict.update({'parent_init_only':parent_init_only})
    #     parent_precode = parent_inputs.pop(
    #                         self.get_linkname('preprocessing_code'),None)
    #     parent_info_dict.update({'parent_precode':bool(parent_precode)})
    # else:
    #     if preproc_code is None:
    #         raise exceptions.InputValidationError("You cannot continue from a non"
    #                                    " wannier calculation without a"
    #                                    " preprocess code")


    # Here info from this calculation, for file copy settings is found
    init_only = settings_dict.pop('INIT_ONLY', self._DEFAULT_INIT_ONLY)
    if init_only and preproc_code is None:
        raise exceptions.InputValidationError(
            'You cannot have init_only mode set, without providing a preprocessing code')

    # prepare the main input text
    input_file_lines = []
    from aiida.common.utils import conv_to_fortran_withlists
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
    input_filename = folder.get_abs_path(self._INPUT_W90_FILE_NAME)
    with open(input_filename, 'w') as file:
        file.write( "\n".join(input_file_lines) )
        file.write( "\n" )

    # Prints the precode input file
    if preproc_code is not None:
        namelist_dict = {'outdir':'.' + PwCalculation._OUTPUT_SUBFOLDER,
                         'prefix':PwCalculation._PREFIX,
                         'seedname':self._SEEDNAME,
                         }
        for precode_param in precode_param_dict:
            namelist_dict.update({precode_param:
                                      precode_param_dict[precode_param]})
        #  Manually makes sure that .EIG, .MMN are not rewritten
        # if  parent_info_dict['wannier_parent']:
        #     user_mmn_setting = namelist_dict.pop('write_mmn',None)
        #     if user_mmn_setting:
        #         raise exceptions.InputValidationError("You attempt to write_mmn for a "
        #                                    " calculation which inherited"
        #                                    " from a wannier90 calc. This"
        #                                    " is not allowed. Either set"
        #                                    " write_mmn to false, or use a"
        #                                    " non-wannier calc as parent.")
        #     namelist_dict.update({'write_mmn':False})
        #      Add write_eig = .false. once this is available
        #      namelist_dict.update({})
        #  checks and adds UNK file
        #  writing UNK as a setting is obsolete
        #  write_unk = settings_dict.pop('WRITE_UNK',None)
        #  if write_unk:
        #      namelist_dict.update({'write_unk':True})
        p2w_input_dict = {'INPUTPP':namelist_dict}

        input_precode_filename = folder.get_abs_path(
            self._INPUT_OVERLAP_FILE_NAME)
        with open(input_precode_filename,'w') as infile:
            for namelist_name in p2w_input_dict.keys():
                infile.write("&{0}\n".format(namelist_name))
                # namelist content; set to {} if not present,
                #  so that we leave an empty namelist
                namelist = p2w_input_dict.pop(namelist_name,{})
                for k, v in sorted(namelist.iteritems()):
                    infile.write(get_input_data_text(k,v))
                infile.write("/\n")

    ############################################################
    #  end of writing text input
    ############################################################ 

    #call to _prepare_wannier90
    # write_to_file_w90=_prepare_for_submission_wannier90(self, folder,inputdict=inputdict) 
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
        nl = blocked[0]
        flag = blocked[1]
        defaultvalue = None
        if len(blocked) >= 3:
            defaultvalue = blocked[2]
        if nl in input_params:
            # The following lines is meant to avoid putting in input the
            # parameters like celldm(*)
            stripped_inparams = [re.sub("[(0-9)]", "", _)
                                 for _ in input_params[nl].keys()]
            if flag in stripped_inparams:
                raise exceptions.InputValidationError(
                    "You cannot specify explicitly the '{}' flag in the '{}' input.".format(flag, nl))
            if defaultvalue is not None:
                if nl not in input_params:
                    input_params[nl] = {}
                input_params[nl][flag] = defaultvalue
    return input_params

def check_capitals(input_params):
    """
    Goes through the input_params (which must be a dictionary) and
    raises an exceptions.InputValidationError if any of the keys are not capitalized
    """
    for k in input_params:
        if k != k.lower():
            raise exceptions.InputValidationError(
                "Please make sure all keys are lower case, {} was not!".format(k))

