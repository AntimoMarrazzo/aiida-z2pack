# -*- coding: utf-8 -*-
# import copy, os
# import numpy as np
# from aiida.common.utils import classproperty
# from aiida.common.exceptions import InputValidationError, ModificationNotAllowed
# from aiida.common.datastructures import CalcInfo, CodeInfo, code_run_modes
# from aiida.orm import JobCalculation, DataFactory
# from aiida_quantumespresso.calculations import (
#     _lowercase_dict, _uppercase_dict, get_input_data_text)
# from aiida_quantumespresso.calculations.pw import PwCalculation
# from aiida.orm.code import Code
# from aiida.orm.data.array.kpoints import KpointsData
# from aiida.orm.data.upf import UpfData
# from aiida.orm.data.upf import get_pseudos_from_structure
# from aiida.orm.data.orbital import OrbitalData, OrbitalFactory
# from aiida.orm.data.parameter import ParameterData
# from aiida.orm.data.remote import RemoteData
# from aiida.orm.data.structure import StructureData
# from aiida.transport import Transport

import os
import six
import copy
from six.moves import zip

from aiida import orm
from aiida.engine import CalcJob
# __authors__ = "Antimo Marrazzo and The AiiDA team."
# __copyright__ = u"Copyright (c), This file is part of the AiiDA platform. For further information please visit http://www.aiida.net/.. All rights reserved."
# __license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file"
# __version__ = "0.6.0"

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


class Z2packCalculation(CalcJob):
    """
    Plugin for Z2pack, a code for computing topological invariants.
    See http://z2pack.ethz.ch/ for more details
    """
    _PSEUDO_SUBFOLDER = './pseudo/'
    _OUTPUT_SUBFOLDER = './out/'
    _PREFIX = 'aiida'
    _INPUT_NSCF_FILE_NAME = 'aiida.nscf.in'
    _OUTPUT_NSCF_FILE_NAME = 'aiida.nscf.out'
    _DATAFILE_XML_BASENAME = 'data-file-schema.xml'
    _DATAFILE_XML = 'undefined.xml'
    _Z2pack_folder = './'
    _Z2pack_folder_restart_files=[]


    ## Default PW output parser provided by AiiDA
    # to be defined in the subclass

    _automatic_namelists = {}

    # in restarts, will not copy but use symlinks
    _default_symlink_usage = True

    # in restarts, it will copy from the parent the following
    _restart_copy_from = os.path.join(_OUTPUT_SUBFOLDER, '*')
    _restart_copy_from_z2pack = os.path.join(_Z2pack_folder, '*')

    # in restarts, it will copy the previous folder in the following one
    _restart_copy_to = _OUTPUT_SUBFOLDER
    _restart_copy_to_z2pack = _Z2pack_folder
    # Default verbosity; change in subclasses
    _default_verbosity = 'high'

    def _init_internal_params(self):
        super(Z2packCalculation, self)._init_internal_params()
        self._use_kpoints = False
        self._DEFAULT_OUTPUT_FILE = 'z2pack_aiida.out'
        self._DEFAULT_INPUT_FILE='z2pack_aiida.py'
        self._DEFAULT_OUTPUT_Z2PACK = 'aiida.json'
        self._DEFAULT_OUTPUT_RESULTS_Z2PACK = 'results.json'
        self._INPUT_W90_FILE_NAME = 'aiida.win'
        self._OUTPUT_W90_FILE_NAME = 'aiida.wout'
        self._ERROR_FILE_NAME = 'aiida.werr'
        self._INPUT_OVERLAP_FILE_NAME = 'aiida.pw2wan.in'
        self._OUTPUT_OVERLAP_FILE_NAME = 'aiida.pw2wan.out'
        self._PREFIX = 'aiida'
        self._SEEDNAME = 'aiida'
        self._default_parser = 'z2pack'  
        self._INPUT_SUBFOLDER = "./out/"
        self._ALWAYS_SYM_FILES = ['UNK*', '*.mmn']
        self._RESTART_SYM_FILES = ['*.amn','*.eig']
        self._CHK_FILE = '*.chk'
        self._DEFAULT_INIT_ONLY = False
        self._DEFAULT_WRITE_UNK = False
        self._DEFAULT_MIN_NEIGHBOUR_DISTANCE = 0.01
        self._DEFAULT_NUM_LINES = 11
        self._DEFAULT_ITERATOR = 'range(8, 27, 2)'
        self._DEFAULT_GAP_TOLERANCE = 0.3
        self._DEFAULT_MOVE_TOLERANCE = 0.3
        self._DEFAULT_POS_TOLERANCE = 0.01
        self._blocked_keywords =[['length_unit','ang']]
        self._blocked_precode_keywords = []
        self._automatic_namelists = {
            'scf': ['CONTROL', 'SYSTEM', 'ELECTRONS'],
            'nscf': ['CONTROL', 'SYSTEM', 'ELECTRONS'],
            'bands': ['CONTROL', 'SYSTEM', 'ELECTRONS'],
            'relax': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS'],
            'md': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS'],
            'vc-md': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL'],
            'vc-relax': ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL'],
        }

        # Keywords that cannot be set
        self._blocked_keywords = [('CONTROL', 'pseudo_dir'),  # set later
                                  ('CONTROL', 'outdir'),  # set later
                                  ('CONTROL', 'prefix'),  # set later
                                  ('SYSTEM', 'ibrav'),  # set later
                                  ('SYSTEM', 'celldm'),
                                  ('SYSTEM', 'nat'),  # set later
                                  ('SYSTEM', 'ntyp'),  # set later
                                  ('SYSTEM', 'a'), ('SYSTEM', 'b'), ('SYSTEM', 'c'),
                                  ('SYSTEM', 'cosab'), ('SYSTEM', 'cosac'), ('SYSTEM', 'cosbc'),
        ]

        @classmethod
        def define(cls, spec):
            # yapf: disable
            super(Z2packCalculation, cls).define(spec)
            spec.input('metadata.options.input_filename', valid_type=six.string_types, default=cls._DEFAULT_INPUT_FILE)
            spec.input('metadata.options.output_filename', valid_type=six.string_types, default=cls._DEFAULT_OUTPUT_FILE)
            spec.input('metadata.options.withmpi', valid_type=bool, default=True)  # Override default withmpi=False
            spec.input('structure', valid_type=orm.StructureData,
                help='The input structure.')
            spec.input('parameters', valid_type=orm.Dict,
                help='The input parameters that are to be used to construct the input file.')
            spec.input('settings', valid_type=orm.Dict, required=False,
                help='Optional parameters to affect the way the calculation job and the parsing are performed.')
            spec.input('parent_folder', valid_type=orm.RemoteData, required=False,
                help='An optional working directory of a previously completed calculation to restart from.')
            spec.input('vdw_table', valid_type=orm.SinglefileData, required=False,
                help='Optional van der Waals table contained in a `SinglefileData`.')
            spec.input_namespace('pseudos', valid_type=orm.UpfData, dynamic=True,
                help='A mapping of `UpfData` nodes onto the kind name to which they should apply.')

    @classproperty
    def _use_methods(cls):
        """
        Additional use_* methods for the z2pack calculation class.
        """
        retdict = JobCalculation._use_methods
        retdict.update({
            "structure": {
                'valid_types': StructureData,
                'additional_parameter': None,
                'linkname': 'structure',
                'docstring': "Choose the input structure to use",
            },
            "settings": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'settings',
               'docstring': "Use an additional node for special settings",
               },
            "pseudo": {
                'valid_types': UpfData,
                'additional_parameter': "kind",
                'linkname': cls._get_linkname_pseudo,
                'docstring': ("Use a node for the UPF pseudopotential of one of "
                              "the elements in the structure. You have to pass "
                              "an additional parameter ('kind') specifying the "
                              "name of the structure kind (i.e., the name of "
                              "the species) for which you want to use this "
                              "pseudo. You can pass either a string, or a "
                              "list of strings if more than one kind uses the "
                              "same pseudo"),
                },
            "nscf_parameters": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'nscf_parameters',
               'docstring': ("Use a node that specifies the input parameters "
                             "for the nscf code"),
               },
            "overlap_parameters": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'overlap_parameters',
               'docstring': ("Use a node that specifies the input parameters "
                             "for the overlap code (pw2wannier90)"),
               },
            "wannier90_parameters": {
               'valid_types': ParameterData,
               'additional_parameter': None,
               'linkname': 'wannier90_parameters',
               'docstring': ("Use a node that specifies the input parameters "
                             "for the wannier code"),
               },

            "parent_folder": {
               'valid_types': RemoteData,
               'additional_parameter': None,
               'linkname': 'parent_calc_folder',
               'docstring': ("Use a remote folder as parent folder (for "
                             "restarts and similar"),
               },
            "nscf_code": {
                'valid_types': Code,
                'additional_parameter': None,
                'linkname': 'nscf_code',
                'docstring': ("Use a nscf code for "
                         "starting z2pack"),
               },
            "overlap_code": {
                'valid_types': Code,
                'additional_parameter': None,
                'linkname': 'overlap_code',
                'docstring': ("Use a overlap code (pw2wannier90) for "
                         "starting wannier90"),
               },  
            "wannier90_code": {
                'valid_types': Code,
                'additional_parameter': None,
                'linkname': 'wannier90_code',
                'docstring': ("Use a Wannier90 code"),
               },
                       })

        return retdict
    @classmethod
    def _get_linkname_pseudo_prefix(cls):
        """
        The prefix for the name of the link used for each pseudo before the kind name
        """
        return "pseudo_"

    @classmethod
    def _get_linkname_pseudo(cls, kind):
        """
        The name of the link used for the pseudo for kind 'kind'.
        It appends the pseudo name to the pseudo_prefix, as returned by the
        _get_linkname_pseudo_prefix() method.

        :note: if a list of strings is given, the elements are appended
          in the same order, separated by underscores

        :param kind: a string (or list of strings) for the atomic kind(s) for
            which we want to get the link name
        """
        # If it is a list of strings, and not a single string: join them
        # by underscore
        if isinstance(kind, (tuple, list)):
            suffix_string = "_".join(kind)
        elif isinstance(kind, basestring):
            suffix_string = kind
        else:
            raise TypeError("The parameter 'kind' of _get_linkname_pseudo can "
                            "only be a string or a list of strings")
            return "{}{}".format(cls._get_linkname_pseudo_prefix(), suffix_string)
    def _if_pos(self, fixed):
            """
            Simple function that returns 0 if fixed is True, 1 otherwise.
            Useful to convert from the boolean value of fixed_coords to the value required
            by Quantum Espresso as if_pos.
            """
            if fixed:
                return 0
            else:
                return 1

    @classmethod
    def _get_linkname_pseudo_prefix(cls):
        """
        The prefix for the name of the link used for each pseudo before the kind name
        """
        return "pseudo_"

    @classmethod
    def _get_linkname_pseudo(cls, kind):
        """
        The name of the link used for the pseudo for kind 'kind'.
        It appends the pseudo name to the pseudo_prefix, as returned by the
        _get_linkname_pseudo_prefix() method.

        :note: if a list of strings is given, the elements are appended
          in the same order, separated by underscores

        :param kind: a string (or list of strings) for the atomic kind(s) for
            which we want to get the link name
        """
        # If it is a list of strings, and not a single string: join them
        # by underscore
        if isinstance(kind, (tuple, list)):
            suffix_string = "_".join(kind)
        elif isinstance(kind, basestring):
            suffix_string = kind
        else:
            raise TypeError("The parameter 'kind' of _get_linkname_pseudo can "
                            "only be a string or a list of strings")
        return "{}{}".format(cls._get_linkname_pseudo_prefix(), suffix_string)

    
    def use_pseudos_from_family(self, family_name):
        """
        Set the pseudo to use for all atomic kinds, picking pseudos from the
        family with name family_name.

        :note: The structure must already be set.

        :param family_name: the name of the group containing the pseudos
        """
        from collections import defaultdict

        try:
            structure = self._get_reference_structure()
        except AttributeError:
            raise ValueError("Structure is not set yet! Therefore, the method "
                             "use_pseudos_from_family cannot automatically set "
                             "the pseudos")

        # A dict {kind_name: pseudo_object}
        kind_pseudo_dict = get_pseudos_from_structure(structure, family_name)

        # We have to group the species by pseudo, I use the pseudo PK
        # pseudo_dict will just map PK->pseudo_object
        pseudo_dict = {}
        # Will contain a list of all species of the pseudo with given PK
        pseudo_species = defaultdict(list)

        for kindname, pseudo in kind_pseudo_dict.iteritems():
            pseudo_dict[pseudo.pk] = pseudo
            pseudo_species[pseudo.pk].append(kindname)

        for pseudo_pk in pseudo_dict:
            pseudo = pseudo_dict[pseudo_pk]
            kinds = pseudo_species[pseudo_pk]
            # I set the pseudo for all species, sorting alphabetically
            self.use_pseudo(pseudo, sorted(kinds))

    def _get_reference_structure(self):
        """
        Used to get the reference structure to obtain which 
        pseudopotentials to use from a given family using 
        use_pseudos_from_family. 
        
        :note: this method can be redefined in a given subclass
               to specify which is the reference structure to consider.
        """
        return self.get_inputs_dict()[self.get_linkname('structure')]

    def _set_parent_remotedata(self, remotedata):
        """
        Used to set a parent remotefolder in the restart of ph.
        """
        from aiida.common.exceptions import ValidationError

        if not isinstance(remotedata, RemoteData):
            raise ValueError('remotedata must be a RemoteData')

        # complain if another remotedata is already found
        input_remote = self.get_inputs(type=RemoteData)
        if input_remote:
            raise ValidationError("Cannot set several parent calculation to a "
                                  "{} calculation".format(self.__class__.__name__))

        self.use_parent_folder(remotedata)
       
    def use_parent_calculation(self, calc):
        """
        Set the parent calculation,
        from which it will inherit the outputsubfolder.
        The link will be created from parent RemoteData and NamelistCalculation
        """
        #if not isinstance(calc, PwCalculation):
        #    raise ValueError("Parent calculation must be a Pw ")
        if not isinstance(calc, (PwCalculation,Z2packCalculation)) :
            raise ValueError("Parent calculation must be a Pw or Z2pack ")
        if isinstance(calc, PwCalculation):
            # Test to see if parent PwCalculation is nscf
            par_type = calc.inp.parameters.dict.CONTROL['calculation'].lower()
            if par_type != 'scf':
                raise ValueError("Pw calculation must be scf") 
        try:
            remote_folder = calc.get_outputs_dict()['remote_folder']
        except KeyError:
            raise AttributeError("No remote_folder found in output to the "
                                 "parent calculation set")
        self.use_parent_folder(remote_folder)
        
    def _prepare_for_submission(self,tempfolder, inputdict):        
        """
        Routine, which creates the input and prepares for submission

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """        
            
        def prepare_nscf(self,tempfolder,inputdict_nscf):
            """ 
            This methods generate the input data for the nscf part of the calculation
            """
            inputdict = inputdict_nscf
            def _generate_PWCPinputdata(self,parameters,settings_dict,pseudos,structure,kpoints=None):
                """
                This method creates the content of an input file
                in the PW/CP format.
                :
                """
                from aiida.common.utils import get_unique_filename, get_suggestion
                import re
                local_copy_list_to_append = []
        
                # I put the first-level keys as uppercase (i.e., namelist and card names)
                # and the second-level keys as lowercase
                # (deeper levels are unchanged)

                input_params = _uppercase_dict(parameters.get_dict(),
                                               dict_name='parameters')
                input_params = {k: _lowercase_dict(v, dict_name=k)
                                for k, v in input_params.iteritems()}
        
                # I remove unwanted elements (for the moment, instead, I stop; to change when
                # we setup a reasonable logging)
                for blocked in self._blocked_keywords:
                    nl = blocked[0].upper()
                    flag = blocked[1].lower()
                    defaultvalue = None
                    if len(blocked) >= 3:
                        defaultvalue = blocked[2]
                    if nl in input_params:
                        # The following lines is meant to avoid putting in input the
                        # parameters like celldm(*)
                        stripped_inparams = [re.sub("[(0-9)]", "", _)
                                             for _ in input_params[nl].keys()]
                        if flag in stripped_inparams:
                            raise InputValidationError(
                                "You cannot specify explicitly the '{}' flag in the '{}' "
                                "namelist or card.".format(flag, nl))
                        if defaultvalue is not None:
                            if nl not in input_params:
                                input_params[nl] = {}
                            input_params[nl][flag] = defaultvalue
        
                # Set some variables (look out at the case! NAMELISTS should be uppercase,
                # internal flag names must be lowercase)
                if 'CONTROL' not in input_params:
                    input_params['CONTROL'] = {}
                input_params['CONTROL']['pseudo_dir'] = '.'+self._PSEUDO_SUBFOLDER
                input_params['CONTROL']['outdir'] = '.'+self._OUTPUT_SUBFOLDER
                input_params['CONTROL']['prefix'] = self._PREFIX
        
                input_params['CONTROL']['verbosity'] = input_params['CONTROL'].get(
                    'verbosity', self._default_verbosity)  # Set to high if not specified
        
                # ============ I prepare the input site data =============
                # ------------ CELL_PARAMETERS -----------
                cell_parameters_card = "CELL_PARAMETERS angstrom\n"
                for vector in structure.cell:
                    cell_parameters_card += ("{0:18.10f} {1:18.10f} {2:18.10f}"
                                             "\n".format(*vector))
        
                # ------------- ATOMIC_SPECIES ------------
                atomic_species_card_list = []
        
                # Keep track of the filenames to avoid to overwrite files
                # I use a dictionary where the key is the pseudo PK and the value
                # is the filename I used. In this way, I also use the same filename
                # if more than one kind uses the same pseudo.
                pseudo_filenames = {}
        
                # I keep track of the order of species
                kind_names = []
                # I add the pseudopotential files to the list of files to be copied
                for kind in structure.kinds:
                    # This should not give errors, I already checked before that
                    # the list of keys of pseudos and kinds coincides
                    ps = pseudos[kind.name]
                    if kind.is_alloy() or kind.has_vacancies():
                        raise InputValidationError("Kind '{}' is an alloy or has "
                                                   "vacancies. This is not allowed for pw.x input structures."
                                                   "".format(kind.name))
        
                    try:
                        # It it is the same pseudopotential file, use the same filename
                        filename = pseudo_filenames[ps.pk]
                    except KeyError:
                        # The pseudo was not encountered yet; use a new name and
                        # also add it to the local copy list
                        filename = get_unique_filename(ps.filename,
                                                       pseudo_filenames.values())
                        pseudo_filenames[ps.pk] = filename
                        # I add this pseudo file to the list of files to copy
                        local_copy_list_to_append.append((ps.get_file_abs_path(),
                                                          os.path.join(self._PSEUDO_SUBFOLDER,
                                                                       filename)))
                    kind_names.append(kind.name)
                    atomic_species_card_list.append("{} {} {}\n".format(
                        kind.name.ljust(6), kind.mass, filename))
        
                # I join the lines, but I resort them using the alphabetical order of
                # species, given by the kind_names list. I also store the mapping_species
                # list, with the order of species used in the file
                mapping_species, sorted_atomic_species_card_list = zip(
                    *sorted(zip(kind_names, atomic_species_card_list)))
                # The format of mapping_species required later is a dictionary, whose
                # values are the indices, so I convert to this format
                # Note the (idx+1) to convert to fortran 1-based lists
                mapping_species = {sp_name: (idx + 1) for idx, sp_name
                                   in enumerate(mapping_species)}
                # I add the first line
                sorted_atomic_species_card_list = (["ATOMIC_SPECIES\n"] +
                                                   list(sorted_atomic_species_card_list))
                atomic_species_card = "".join(sorted_atomic_species_card_list)
                # Free memory
                del sorted_atomic_species_card_list
                del atomic_species_card_list
        
                # ------------ ATOMIC_POSITIONS -----------
                atomic_positions_card_list = ["ATOMIC_POSITIONS angstrom\n"]
        
                # Check on validity of FIXED_COORDS
                fixed_coords_strings = []
                fixed_coords = settings_dict.pop('FIXED_COORDS', None)
                if fixed_coords is None:
                    # No fixed_coords specified: I store a list of empty strings
                    fixed_coords_strings = [""] * len(structure.sites)
                else:
                    if len(fixed_coords) != len(structure.sites):
                        raise InputValidationError(
                            "Input structure contains {:d} sites, but "
                            "fixed_coords has length {:d}".format(len(structure.sites),
                                                                  len(fixed_coords)))
        
                    for i, this_atom_fix in enumerate(fixed_coords):
                        if len(this_atom_fix) != 3:
                            raise InputValidationError("fixed_coords({:d}) has not length three"
                                                       "".format(i + 1))
                        for fixed_c in this_atom_fix:
                            if not isinstance(fixed_c, bool):
                                raise InputValidationError("fixed_coords({:d}) has non-boolean "
                                                           "elements".format(i + 1))
        
                        if_pos_values = [self._if_pos(_) for _ in this_atom_fix]
                        fixed_coords_strings.append("  {:d} {:d} {:d}".format(*if_pos_values))
        
                for site, fixed_coords_string in zip(
                        structure.sites, fixed_coords_strings):
                    atomic_positions_card_list.append(
                        "{0} {1:18.10f} {2:18.10f} {3:18.10f} {4}\n".format(
                            site.kind_name.ljust(6), site.position[0], site.position[1],
                            site.position[2], fixed_coords_string))
                atomic_positions_card = "".join(atomic_positions_card_list)
                del atomic_positions_card_list  # Free memory
        
                # I set the variables that must be specified, related to the system
                # Set some variables (look out at the case! NAMELISTS should be
                # uppercase, internal flag names must be lowercase)
                if 'SYSTEM' not in input_params:
                    input_params['SYSTEM'] = {}
                input_params['SYSTEM']['ibrav'] = 0
                input_params['SYSTEM']['nat'] = len(structure.sites)
                input_params['SYSTEM']['ntyp'] = len(structure.kinds)
            
                # ============ I prepare the k-points =============
                if self._use_kpoints:
                    try:
                        mesh, offset = kpoints.get_kpoints_mesh()
                        has_mesh = True
                        force_kpoints_list = settings_dict.pop('FORCE_KPOINTS_LIST', False)
                        if force_kpoints_list:
                            kpoints_list = kpoints.get_kpoints_mesh(print_list=True)
                            num_kpoints = len(kpoints_list)
                            has_mesh = False
                            weights = [1.] * num_kpoints
        
                    except AttributeError:
        
                        try:
                            kpoints_list = kpoints.get_kpoints()
                            num_kpoints = len(kpoints_list)
                            has_mesh = False
                            if num_kpoints == 0:
                                raise InputValidationError("At least one k point must be "
                                                           "provided for non-gamma calculations")
                        except AttributeError:
                            raise InputValidationError("No valid kpoints have been found")
        
                        try:
                            _, weights = kpoints.get_kpoints(also_weights=True)
                        except AttributeError:
                            weights = [1.] * num_kpoints
        
                    gamma_only = settings_dict.pop("GAMMA_ONLY", False)
        
                    if gamma_only:
                        if has_mesh:
                            if tuple(mesh) != (1, 1, 1) or tuple(offset) != (0., 0., 0.):
                                raise InputValidationError(
                                    "If a gamma_only calculation is requested, the "
                                    "kpoint mesh must be (1,1,1),offset=(0.,0.,0.)")
        
                        else:
                            if ( len(kpoints_list) != 1 or
                                         tuple(kpoints_list[0]) != tuple(0., 0., 0.) ):
                                raise InputValidationError(
                                    "If a gamma_only calculation is requested, the "
                                    "kpoints coordinates must only be (0.,0.,0.)")
        
                        kpoints_type = "gamma"
        
                    elif has_mesh:
                        kpoints_type = "automatic"
        
                    else:
                        kpoints_type = "crystal"
        
                    kpoints_card_list = ["K_POINTS {}\n".format(kpoints_type)]
        
                    if kpoints_type == "automatic":
                        if any([(i != 0. and i != 0.5) for i in offset]):
                            raise InputValidationError("offset list must only be made "
                                                       "of 0 or 0.5 floats")
                        the_offset = [0 if i == 0. else 1 for i in offset]
                        the_6_integers = list(mesh) + the_offset
                        kpoints_card_list.append("{:d} {:d} {:d} {:d} {:d} {:d}\n"
                                                 "".format(*the_6_integers))
        
                    elif kpoints_type == "gamma":
                        # nothing to be written in this case
                        pass
                    else:
                        kpoints_card_list.append("{:d}\n".format(num_kpoints))
                        for kpoint, weight in zip(kpoints_list, weights):
                            kpoints_card_list.append(
                                "  {:18.10f} {:18.10f} {:18.10f} {:18.10f}"
                                "\n".format(kpoint[0], kpoint[1], kpoint[2], weight))
        
                    kpoints_card = "".join(kpoints_card_list)
                    del kpoints_card_list
        
                # =================== NAMELISTS AND CARDS ========================
                try:
                    namelists_toprint = settings_dict.pop('NAMELISTS')
                    if not isinstance(namelists_toprint, list):
                        raise InputValidationError(
                            "The 'NAMELISTS' value, if specified in the settings input "
                            "node, must be a list of strings")
                except KeyError:  # list of namelists not specified; do automatic detection
                    try:
                        control_nl = input_params['CONTROL']
                        calculation_type = control_nl['calculation']
                    except KeyError:
                        raise InputValidationError(
                            "No 'calculation' in CONTROL namelist."
                            "It is required for automatic detection of the valid list "
                            "of namelists. Otherwise, specify the list of namelists "
                            "using the NAMELISTS key inside the 'settings' input node")
        
                    try:
                        namelists_toprint = self._automatic_namelists[calculation_type]
                    except KeyError:
                        sugg_string = get_suggestion(calculation_type,
                                                     self._automatic_namelists.keys())
                        raise InputValidationError("Unknown 'calculation' value in "
                                                   "CONTROL namelist {}. Otherwise, specify the list of "
                                                   "namelists using the NAMELISTS inside the 'settings' input "
                                                   "node".format(sugg_string))
        
                inputfile = ""
                for namelist_name in namelists_toprint:
                    inputfile += "&{0}\n".format(namelist_name)
                    # namelist content; set to {} if not present, so that we leave an 
                    # empty namelist
                    namelist = input_params.pop(namelist_name, {})
                    for k, v in sorted(namelist.iteritems()):
                        inputfile += get_input_data_text(k, v, mapping=mapping_species)
                    inputfile += "/\n"
            
                # Write cards now
                inputfile += atomic_species_card
                inputfile += atomic_positions_card
                if self._use_kpoints:
                    inputfile += kpoints_card
                inputfile += cell_parameters_card
                #TODO: write CONSTRAINTS
                #TODO: write OCCUPATIONS
                    
                if input_params:            
                    raise InputValidationError(
                        "The following namelists are specified in input_params, but are "
                        "not valid namelists for the current type of calculation: "
                        "{}".format(",".join(input_params.keys())))
        
                return inputfile, local_copy_list_to_append
        
            def _prepare_for_submission_nscf(self, tempfolder,inputdict):
                """
                This is the routine to be called when you want to create
                the input files and related stuff with a plugin.
        
                :param tempfolder: a aiida.common.folders.Folder subclass where
                                   the plugin should put all its files.
                :param inputdict: a dictionary with the input nodes, as they would
                        be returned by get_inputs_dict (without the Code!)
                """
                local_copy_list = []
                remote_copy_list = []
                remote_symlink_list = []
        
                try:
                    parameters = inputdict.pop(self.get_linkname('nscf_parameters'))
                except KeyError:
                    raise InputValidationError("No parameters specified for this calculation")
                if not isinstance(parameters, ParameterData):
                    raise InputValidationError("parameters is not of type ParameterData")
        
                try:
                    structure = inputdict.pop(self.get_linkname('structure'))
                except KeyError:
                    raise InputValidationError("No structure specified for this calculation")
                if not isinstance(structure, StructureData):
                    raise InputValidationError("structure is not of type StructureData")
        
                kpoints = None
        
                # Settings can be undefined, and defaults to an empty dictionary
                settings = inputdict.pop(self.get_linkname('settings'), None)
                if settings is None:
                    settings_dict = {}
                else:
                    if not isinstance(settings, ParameterData):
                        raise InputValidationError("settings, if specified, must be of "
                                                   "type ParameterData")
                    # Settings converted to uppercase
                    settings_dict = _uppercase_dict(settings.get_dict(),
                                                    dict_name='settings')
        
                pseudos = {}
                # I create here a dictionary that associates each kind name to a pseudo
                for link in inputdict.keys():
                    if link.startswith(self._get_linkname_pseudo_prefix()):
                        kindstring = link[len(self._get_linkname_pseudo_prefix()):]
                        kinds = kindstring.split('_')
                        the_pseudo = inputdict.pop(link)
                        if not isinstance(the_pseudo, UpfData):
                            raise InputValidationError("Pseudo for kind(s) {} is not of "
                                                       "type UpfData".format(",".join(kinds)))
                        for kind in kinds:
                            if kind in pseudos:
                                raise InputValidationError("Pseudo for kind {} passed "
                                                           "more than one time".format(kind))
                            pseudos[kind] = the_pseudo
        
                parent_calc_folder = inputdict.pop(self.get_linkname('parent_folder'), None)
                if parent_calc_folder is not None:
                    if not isinstance(parent_calc_folder, RemoteData):
                        raise InputValidationError("parent_calc_folder, if specified, "
                                                   "must be of type RemoteData")
        
    
                try:
                    code = inputdict.pop(self.get_linkname('nscf_code'))
                except KeyError:
                    raise InputValidationError("No code specified for this calculation")
        
                # Here, there should be no more parameters...
        #        if inputdict:
        #            raise InputValidationError("The following input data nodes are "
        #                                       "unrecognized: {}".format(inputdict.keys()))
        
                # Check structure, get species, check peudos
                kindnames = [k.name for k in structure.kinds]
                if set(kindnames) != set(pseudos.keys()):
                    err_msg = ("Mismatch between the defined pseudos and the list of "
                               "kinds of the structure. Pseudos: {}; kinds: {}".format(
                        ",".join(pseudos.keys()), ",".join(list(kindnames))))
                    raise InputValidationError(err_msg)
        
                ##############################
                # END OF INITIAL INPUT CHECK #
                ##############################
                # I create the subfolder that will contain the pseudopotentials
                tempfolder.get_subfolder(self._PSEUDO_SUBFOLDER, create=True)
                # I create the subfolder with the output data (sometimes Quantum
                # Espresso codes crash if an empty folder is not already there
                tempfolder.get_subfolder(self._OUTPUT_SUBFOLDER, create=True)
        
                # If present, add also the Van der Waals table to the pseudo dir
                # Note that the name of the table is not checked but should be the
                # one expected by QE.
        #        if vdw_table:
        #            local_copy_list.append(
        #                (
        #                vdw_table.get_file_abs_path(),
        #                os.path.join(self._PSEUDO_SUBFOLDER,
        #                    os.path.split(vdw_table.get_file_abs_path())[1])
        #                )
        #                )
                input_filecontent, local_copy_pseudo_list = _generate_PWCPinputdata(self,parameters,settings_dict,pseudos,
                                                                                         structure,kpoints)
                local_copy_list += local_copy_pseudo_list
                #calcinfo = CalcInfo()
                input_filename = tempfolder.get_abs_path(self._INPUT_NSCF_FILE_NAME)
                with open(input_filename, 'w') as infile:
                    infile.write(input_filecontent)
        
                # operations for restart
                symlink = settings_dict.pop('PARENT_FOLDER_SYMLINK', self._default_symlink_usage)  # a boolean
                if symlink:
                    if parent_calc_folder is not None:
                        # I put the symlink to the old parent ./out folder
                        remote_symlink_list.append(
                            (parent_calc_folder.get_computer().uuid,
                             os.path.join(parent_calc_folder.get_remote_path(),
                                          self._restart_copy_from),
                             self._restart_copy_to
                            ))
                else:
                    # copy remote output dir, if specified
                    if parent_calc_folder is not None:
                        remote_copy_list.append(
                            (parent_calc_folder.get_computer().uuid,
                             os.path.join(parent_calc_folder.get_remote_path(),
                                          self._restart_copy_from),
                             self._restart_copy_to
                            ))
      
                #return calcinfo
                return local_copy_list,remote_copy_list,remote_symlink_list
        

            #Call to the _prepare_for_submission_nscf
            #print inputdict
            lc,rc,sy=_prepare_for_submission_nscf(self, tempfolder,inputdict=inputdict)
            return lc,rc,sy
                        
        def prepare_wannier90(self,tempfolder,inputdict_wannier90):
            inputdict=inputdict_wannier90
            def _prepare_for_submission_wannier90(self,tempfolder, inputdict):        
                """
                Routine, which creates the input and prepares for submission
        
                :param tempfolder: a aiida.common.folders.Folder subclass where
                                   the plugin should put all its files.
                :param inputdict: a dictionary with the input nodes, as they would
                        be returned by get_inputdata_dict (without the Code!)
                """
                ##################################################################
                # Input validation
                ##################################################################
        
                # Grabs parent calc information
#                parent_folder = inputdict.pop(self.get_linkname('parent_folder'),None)
#                if not isinstance(parent_folder, RemoteData):
#                    raise InputValidationError("parent_folder is not of type "
#                                               "RemoteData")
        
                # Tries to get the input parameters
                try:
                    parameters = inputdict.pop(self.get_linkname('wannier90_parameters'))
                except KeyError:
                    raise InputValidationError("No parameters specified for "
                                               "this calculation")
                if not isinstance(parameters, ParameterData):
                    raise InputValidationError("parameters is not of "
                                               "type ParameterData")
        
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
                                raise InputValidationError(
                                    "You cannot specify explicitly the '{}' flag in "
                                    "the '{}' input.".format(flag, nl))
                            if defaultvalue is not None:
                                if nl not in input_params:
                                    input_params[nl] = {}
                                input_params[nl][flag] = defaultvalue
                    return input_params
        
                def check_capitals(input_params):
                    """
                    Goes through the input_params (which much be a dictionary) and
                    raises an InputValidationError if any of the keys are not capitalized
                    """
                    for k in input_params:
                        if k != k.lower():
                            raise InputValidationError("Please make sure all keys"
                                                       "are lower case, {} was not!"
                                                       "".format(k))
                param_dict = parameters.get_dict()
                param_dict = blocked_keyword_finder(param_dict, self._blocked_keywords)
                check_capitals(param_dict)
        
                # Tries to get the precode input paramters
                try:
                    precode_parameters = inputdict.pop(self.get_linkname
                                                       ('overlap_parameters'))
                except KeyError:
                    precode_parameters = ParameterData(dict={})
                if not isinstance(precode_parameters,ParameterData):
                    raise InputValidationError('precode_parameters is not '
                                               'of type ParameterData')
                precode_param_dict = precode_parameters.get_dict()
                precode_param_dict = blocked_keyword_finder(precode_param_dict,
                                                    self._blocked_precode_keywords)
                check_capitals(precode_param_dict)
                # Tries to get the input projections
                #try:
                #    projections = inputdict.pop(self.get_linkname('projections'))
                #except KeyError:
                #    raise InputValidationError("No projections specified for "
                #                               "this calculation")
                #if not isinstance(projections, OrbitalData):
                #    raise InputValidationError("projections is not of type "
                #                                 "OrbitalData")
        
                # Tries to get the input kpoints
                #try:
                #    kpoints = inputdict.pop(self.get_linkname('kpoints'))
                #except KeyError:
                #    raise InputValidationError("No kpoints specified for this"
                #                               " calculation")
                #if not isinstance(kpoints, KpointsData):
                #    raise InputValidationError("kpoints is not of type KpointsData")
        
                # Tries to get the input kpath, but is not actually mandatory and will
                #  default to None if not found
#                kpoints_path = inputdict.pop(self.get_linkname('kpoints_path'), None)
#                if not isinstance(kpoints, KpointsData) and kpoints_path is not None:
#                    raise InputValidationError("kpoints_path is not of type "
#                                               "KpointsData")
        
                # Tries to get the input structure
                try:
                    structure = inputdict.pop(self.get_linkname('structure'))
                except KeyError:
                    raise InputValidationError("No structure specified for this "
                                               "calculation")
                if not isinstance(structure, StructureData):
                    raise InputValidationError("structure is not of type "
                                               "StructureData")
        
                # Settings can be undefined, and defaults to an empty dictionary
                settings = inputdict.pop(self.get_linkname('settings'),None)
                if settings is None:
                    settings_dict = {}
                else:
                    if not isinstance(settings,  ParameterData):
                        raise InputValidationError("settings, if specified, must be "
                                                   "of type ParameterData")
                    # Settings converted to uppercase
                    settings_dict = _uppercase_dict(settings.get_dict(),
                                                    dict_name='settings')
        
                # This section handles the multicode support
                main_code = inputdict.pop(self.get_linkname('code'),None)
                if main_code is None:
                    raise InputValidationError("No input code found!")
        
        
                preproc_code =  inputdict.pop(self.get_linkname('overlap_code')
                                              ,None)
                if preproc_code is not None:
                    if not isinstance(preproc_code, Code):
                        raise InputValidationError("preprocessing_code, if specified,"
                                                   "must be of type Code")
        
                ############################################################
                # End basic check on inputs
                ############################################################
        
                # Here info from the parent, for file copy settings is found
    #            parent_info_dict = {}
    #            parent_calc = parent_folder.get_inputs_dict()['remote_folder']
    #            parent_inputs = parent_calc.get_inputs_dict()
    #            wannier_parent = isinstance(parent_calc, Wannier90Calculation)
    #            parent_info_dict.update({'wannier_parent':wannier_parent})
    #            if parent_info_dict['wannier_parent']:
    #                # If wannier parent, check if it was INIT_ONY and if precode used
    #                parent_settings = parent_inputs.pop('settings',{})
    #                try:
    #                    parent_settings = parent_settings.get_inputs_dict()
    #                except AttributeError:
    #                    pass
    #                parent_init_only = parent_settings.pop('INIT_ONLY',
    #                                                       self._DEFAULT_INIT_ONLY)
    #                parent_info_dict.update({'parent_init_only':parent_init_only})
    #                parent_precode = parent_inputs.pop(
    #                                    self.get_linkname('preprocessing_code'),None)
    #                parent_info_dict.update({'parent_precode':bool(parent_precode)})
    #            else:
    #                if preproc_code is None:
    #                    raise InputValidationError("You cannot continue from a non"
    #                                               " wannier calculation without a"
    #                                               " preprocess code")
        
        
                # Here info from this calculation, for file copy settings is found
                init_only = settings_dict.pop('INIT_ONLY', self._DEFAULT_INIT_ONLY)
                if init_only:
                    if preproc_code is None:
                        raise InputValidationError ('You cannot have init_only '
                                                    'mode set, without providing a '
                                                    'preprocessing code')
        
                # prepare the main input text
                input_file_lines = []
                from aiida.common.utils import conv_to_fortran_withlists
                for param in param_dict:
                    input_file_lines.append(param+' = '+conv_to_fortran_withlists(
                        param_dict[param]))
        
                # take projectionsdict and write to file
                # checks if spins are used, and modifies the opening line
    #            projection_list = projections.get_orbitals()
    #            spin_use = any([bool(projection.get_orbital_dict()['spin'])
    #                           for projection in projection_list])
    #            if spin_use:
    #                raise InputValidationError("spinors are implemented but not tested"
    #                                           " disable this error if you know what "
    #                                           "you are doing!")
    #                projector_type = "spinor_projections"
    #            else:
    #                projector_type = "projections"
    #            input_file_lines.append('Begin {}'.format(projector_type))
    #            for projection in projection_list:
    #                orbit_line = _print_wann_line_from_orbital(projection)
    #                input_file_lines.append(orbit_line)
    #            input_file_lines.append('End {}'.format(projector_type))
        
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
                atoms_cart = zip(wann_kind_names,wann_positions)
                for atom in atoms_cart:
                    input_file_lines.append('{}  {}'.format(atom[0],atom[1]))
                input_file_lines.append('End atoms_cart')
        
                # convert the kpoints_path
    #            try:
    #                special_points = kpoints_path.get_special_points()
    #            except ModificationNotAllowed:
    #                raise InputValidationError('kpoints_path must be kpoints with '
                #                                'a special kpoint path already set!')
        
                #TODO If someone wanted to add custom kpoint_path support do so here
        
    #            input_file_lines.append('Begin Kpoint_Path')
    #            for i in range(len(special_points[1])):
    #                point1, point2 = special_points[1][i]
    #                coord1 = special_points[0][point1]
    #                coord2 = special_points[0][point2]
    #                path_line = '{} {} {} {} '.format(point1,*coord1)
    #                path_line += ' {} {} {} {}'.format(point2,*coord2)
    #               input_file_lines.append(path_line)
    #            input_file_lines.append('End Kpoint_Path')
        
                # convert the kmesh
    #            try:
    #                kmesh = kpoints.get_kpoints_mesh()[0]
    #            except AttributeError:
    #                raise InputValidationError('kpoints should be set with '
    #                                           'set_kpoints_mesh, '
    #                                           'and not set_kpoints... ')
        
    #            mp_line = 'mp_grid = {},{},{}'.format(*kmesh)
    #            input_file_lines.append(mp_line)
        
    #            input_file_lines.append('Begin kpoints')
    #            for vector in kpoints.get_kpoints_mesh(print_list=True):
    #                input_file_lines.append("{0:18.10f} {1:18.10f} {2:18.10f}"
    #                                        .format(*vector))
    #            input_file_lines.append('End kpoints')
        
                # Prints to file the main input
                input_filename = tempfolder.get_abs_path(self._INPUT_W90_FILE_NAME)
                with open(input_filename, 'w') as file:
                    file.write( "\n".join(input_file_lines) )
                    file.write( "\n" )
        
                # Prints the precode input file
                if preproc_code is not None:
                    namelist_dict = {'outdir':'.'+PwCalculation._OUTPUT_SUBFOLDER,
                                     'prefix':PwCalculation._PREFIX,
                                     'seedname':self._SEEDNAME,
                                     }
                    for precode_param in precode_param_dict:
                        namelist_dict.update({precode_param:
                                                  precode_param_dict[precode_param]})
                    # Manually makes sure that .EIG, .MMN are not rewritten
    #                if  parent_info_dict['wannier_parent']:
    #                    user_mmn_setting = namelist_dict.pop('write_mmn',None)
    #                    if user_mmn_setting:
    #                        raise InputValidationError("You attempt to write_mmn for a "
    #                                                   " calculation which inherited"
    #                                                   " from a wannier90 calc. This"
    #                                                   " is not allowed. Either set"
    #                                                   " write_mmn to false, or use a"
    #                                                   " non-wannier calc as parent.")
    #                    namelist_dict.update({'write_mmn':False})
                        # Add write_eig = .false. once this is available
                        # namelist_dict.update({})
                    # checks and adds UNK file
                    # writing UNK as a setting is obsolete
                    # write_unk = settings_dict.pop('WRITE_UNK',None)
                    # if write_unk:
                    #     namelist_dict.update({'write_unk':True})
                    p2w_input_dict = {'INPUTPP':namelist_dict}
        
                    input_precode_filename = tempfolder.get_abs_path(
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
            write_to_file_w90=_prepare_for_submission_wannier90(self, tempfolder,inputdict=inputdict) 
            return write_to_file_w90         
        def prepare_z2pack(self,tempfolder,inputdict):
#            input_filename = tempfolder.get_abs_path(self._DEFAULT_INPUT_Z2PACK)
            input_filename = tempfolder.get_abs_path(self._DEFAULT_INPUT_FILE)
            try:
                nscf_code = inputdict.pop(self.get_linkname('nscf_code'))
            except KeyError:
                raise InputValidationError("No nscf code specified for this calculation")
            try:
                overlap_code = inputdict.pop(self.get_linkname('overlap_code'))
            except KeyError:
                raise InputValidationError("No overlap code specified for this calculation")
            try:
                wannier90_code = inputdict.pop(self.get_linkname('wannier90_code'))
            except KeyError:
                raise InputValidationError("No Wannier90 code specified for this calculation")
            
            try:
                settings_dict = inputdict.pop(self.get_linkname('settings'))
                try:
                    mpi_command=settings_dict.get_dict()['mpi_command']
                except KeyError:
                    raise InputValidationError("No mpi_command code specified for this calculation")
                try:
                    npools = settings_dict.get_dict()['npools']
                    if type(npools)!=int:
                        raise InputValidationError("npools must be an integer.")
                    else:
                        pools_cmd = ' -nk '+str(npools) + ' '                           
                except KeyError:
                    pools_cmd = ''
                try:
                    dim_mode = settings_dict.get_dict()['dimension_mode']
                except KeyError:
                    raise InputValidationError("No dimension_mode specified for this calculation")
                try:
                    invariant = settings_dict.get_dict()['invariant']
                except KeyError:
                    raise InputValidationError("No invariant specified for this calculation")
                try:
                    pos_tol = settings_dict.get_dict()['pos_tol']
                except KeyError:
                    pos_tol = self._DEFAULT_POS_TOLERANCE
                try:
                    gap_tol = settings_dict.get_dict()['gap_tol']
                except KeyError:
                    gap_tol = self._DEFAULT_GAP_TOLERANCE
                try:
                    move_tol = settings_dict.get_dict()['move_tol']
                except KeyError:
                    move_tol = self._DEFAULT_MOVE_TOLERANCE
                try:
                    num_lines = settings_dict.get_dict()['num_lines']
                except KeyError:
                    num_lines = self._DEFAULT_NUM_LINES
                try:
                    min_neighbour_dist=settings_dict.get_dict()['min_neighbour_dist']
                except KeyError:
                    min_neighbour_dist= self._DEFAULT_MIN_NEIGHBOUR_DISTANCE
                try:
                    iterator=settings_dict.get_dict()['iterator']
                except KeyError:
                    iterator= self._DEFAULT_ITERATOR
                try:
                    restart_mode=settings_dict.get_dict()['restart']
                except KeyError:
                    restart_mode=False
                if(dim_mode=='3D'):
                    try:
                        surface = settings_dict.get_dict()['surface']
                    except KeyError:
                        raise InputValidationError("A surface must be specified for dim_mode==3D ")
                try:
                    prepend_code=settings_dict.get_dict()['prepend_code']
                except KeyError:
                    prepend_code=''
            except KeyError:
                raise InputValidationError("No settings code specified for this calculation")          
            
            input_file_lines=[]
            input_file_lines.append('#!/usr/bin/env python3')
            input_file_lines.append('import sys')
            #input_file_lines.append('sys.path.append('+"'"+'./Z2pack'+"'"+')')
            input_file_lines.append('import z2pack')
            input_file_lines.append('import os')
            input_file_lines.append('import shutil')
            input_file_lines.append('import subprocess')
            input_file_lines.append('import xml.etree.ElementTree as ET')
            input_file_lines.append('import json')
            nscf_cmd = ' '+mpi_command+' ' +nscf_code.get_execname()
            overlap_cmd =' '+mpi_command+' ' +overlap_code.get_execname()
            #wannier90_cmd =  wannier90_code.get_execname() #Serial w90
            wannier90_cmd = ' '+mpi_command+' ' + wannier90_code.get_execname() #Parallel w90
            
            z2cmd = ("('"+wannier90_cmd + ' aiida '+' -pp;' +"'"+  "+"+'\n'+
                     "'"+nscf_cmd + pools_cmd + '< ' +self._INPUT_NSCF_FILE_NAME + '>& pw.log;' +"'"+ "+" +'\n'+
                     "'"+overlap_cmd + '< ' + self._INPUT_OVERLAP_FILE_NAME+' >& pw2wan.log;' + "')")
            input_file_lines.append('z2cmd =' +  z2cmd)
            input_files = [ self._INPUT_NSCF_FILE_NAME, self._INPUT_OVERLAP_FILE_NAME,self._INPUT_W90_FILE_NAME]
            input_file_lines.append('input_files='+str(input_files))
            input_file_lines.append('system=z2pack.fp.System(input_files=input_files,')
            input_file_lines.append('\t kpt_fct=[z2pack.fp.kpoint.qe, z2pack.fp.kpoint.wannier90_full],')
#            input_file_lines.append('\t kpt_fct=[z2pack.fp.kpoint.qe, z2pack.fp.kpoint.wannier90],')
            input_file_lines.append('\t'+'kpt_path='+str([ self._INPUT_NSCF_FILE_NAME, self._INPUT_W90_FILE_NAME])+',')
            input_file_lines.append('\t command=z2cmd'+',')
            input_file_lines.append("\t executable='/bin/bash'"+',')
            input_file_lines.append("\t mmn_path='aiida.mmn')")            
            input_file_lines.append("gap_check={}")
            input_file_lines.append("move_check={}")
            input_file_lines.append("pos_check={}")
            input_file_lines.append("res_dict={'convergence_report':{'GapCheck':{},"
                                                                     "'MoveCheck':{},"
                                                                     "'PosCheck':{}"
                                                                     "}"
                                                ",'invariant':{}}")
            if(prepend_code!=''):
                input_file_lines.append("\t"+prepend_code)
            if (dim_mode=='2D' or dim_mode=='3D'):       
                input_file_lines.append('result = z2pack.surface.run(')
                input_file_lines.append('\t system=system,')
                if(dim_mode=='2D'):
                    if (invariant=='Z2'):
                        input_file_lines.append('\t surface=lambda t1,t2: [t1/2, t2, 0],')
                    elif (invariant=='Chern'):
                        input_file_lines.append('\t surface=lambda t1,t2: [t1, t2, 0],')
                elif(dim_mode=='3D'):
                    input_file_lines.append('\t surface='+surface+',')
                input_file_lines.append('\t pos_tol='+str(pos_tol)+',')
                input_file_lines.append('\t gap_tol='+str(gap_tol)+',')
                input_file_lines.append('\t move_tol='+str(move_tol)+',')
                input_file_lines.append('\t num_lines='+str(num_lines)+',')
                input_file_lines.append('\t min_neighbour_dist='+ str(min_neighbour_dist)+',')
                input_file_lines.append('\t iterator='+ str(iterator)+',')
                input_file_lines.append('\t save_file='+"'"+self._DEFAULT_OUTPUT_Z2PACK +"'"+',')
                if restart_mode:
                    input_file_lines.append('\t load=True')
                input_file_lines.append('\t )')
                if (invariant=='Z2'):
                    input_file_lines.append('Z2=z2pack.invariant.z2(result)')
                    input_file_lines.append("res_dict['invariant'].update({'Z2':Z2})")
                elif (invariant=='Chern'):
                    input_file_lines.append('Chern=z2pack.invariant.chern(result)')
                    input_file_lines.append("res_dict['invariant'].update({'Chern':Chern})")
            else:
                raise TypeError("Only dimension_mode 2D and 3D"
                            "are currently implemented.")
            
            input_file_lines.append("gap_check['PASSED']="
                                    "result.convergence_report['surface']['GapCheck']['PASSED']")
            input_file_lines.append("gap_check['FAILED']="
                                    "result.convergence_report['surface']['GapCheck']['FAILED']")
            input_file_lines.append("move_check['PASSED']="
                                    "result.convergence_report['surface']['MoveCheck']['PASSED']")
            input_file_lines.append("move_check['FAILED']="
                                    "result.convergence_report['surface']['MoveCheck']['FAILED']")
            input_file_lines.append("pos_check['PASSED']="
                                    "result.convergence_report['line']['PosCheck']['PASSED']")
            input_file_lines.append("pos_check['FAILED']="
                                    "result.convergence_report['line']['PosCheck']['FAILED']")
            input_file_lines.append("pos_check['MISSING']="
                                    "result.convergence_report['line']['PosCheck']['MISSING']")

            input_file_lines.append("res_dict['convergence_report']['GapCheck'].update(gap_check)")
            input_file_lines.append("res_dict['convergence_report']['MoveCheck'].update(move_check)")
            input_file_lines.append("res_dict['convergence_report']['PosCheck'].update(pos_check)")  
            
            input_file_lines.append("with open('results.json', 'w') as fp:")
            input_file_lines.append("\t json.dump(res_dict, fp)")
            with open(input_filename, 'w') as file_input:
                file_input.write( "\n".join(input_file_lines) )
                file_input.write( "\n" )
                
        from copy import deepcopy
        inputdict_nscf=inputdict
        inputdict_wannier90=deepcopy(inputdict)
        inputdict_z2pack=deepcopy(inputdict)
        inputdict_final=deepcopy(inputdict)
        inputdict_settings=deepcopy(inputdict)
        try:
            settings_dict = inputdict_settings.pop(self.get_linkname('settings'))
            try:
                restart_mode=settings_dict.get_dict()['restart']
            except KeyError:
                restart_mode=False
        except KeyError:
            restart_mode=False
            raise InputValidationError("No settings specified for this calculation")

        if restart_mode is False:
            local_copy_list,remote_copy_list,remote_symlink_list=prepare_nscf(self,tempfolder,inputdict_nscf)
            prepare_wannier90(self,tempfolder,inputdict_wannier90)
        else:
            parent_calc_folder = inputdict_settings.pop(self.get_linkname('parent_folder'), None)
            local_copy_list=[]
            remote_symlink_list=[]
            remote_copy_list=[]
                    
            _internal_retrieve_list = [self._INPUT_NSCF_FILE_NAME,
                                   self._INPUT_W90_FILE_NAME,
                                   self._DEFAULT_OUTPUT_Z2PACK,
                                   self._INPUT_OVERLAP_FILE_NAME,
                                   ]
            if parent_calc_folder is not None:
                for file_copy in _internal_retrieve_list:
                    remote_copy_list.append(
                        (parent_calc_folder.get_computer().uuid,
                         os.path.join(parent_calc_folder.get_remote_path(),
                         file_copy),
                    self._restart_copy_to_z2pack
                    ))
                remote_copy_list.append(
                        (parent_calc_folder.get_computer().uuid,
                         os.path.join(parent_calc_folder.get_remote_path(),
                         './out'),
                    './'
                    ))
        #   if parent_calc_folder is not None:
        #           remote_copy_list.append(
        #           (parent_calc_folder.get_computer().uuid,
        #           os.path.join(parent_calc_folder.get_remote_path(),
        #                        self._restart_copy_from),
        #           self._restart_copy_to
        #           ))                    
        #    if parent_calc_folder is not None:
        #          # I put the symlink to the old parent ./out folder
        #        remote_symlink_list.append(
        #            (parent_calc_folder.get_computer().uuid,
        #            os.path.join(parent_calc_folder.get_remote_path(),
        #                        self._restart_copy_from),
        #            self._restart_copy_to
        #            ))          
        prepare_z2pack(self,tempfolder,inputdict_z2pack)        
        ############################################
        # set Calcinfo
        ############################################
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.remote_symlink_list = remote_symlink_list
        
        c1 = CodeInfo()
        c1.withmpi = False # No mpi
        c1.cmdline_params = ['>',self._DEFAULT_OUTPUT_FILE]
        main_code = inputdict_final.pop(self.get_linkname('code'),None)
        c1.code_uuid = main_code.uuid
        codes_info=[c1]
        calcinfo.codes_info = codes_info
        calcinfo.codes_run_mode = code_run_modes.SERIAL
        
        # Retrieve files
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self._DEFAULT_OUTPUT_Z2PACK)
        calcinfo.retrieve_list.append(self._DEFAULT_OUTPUT_FILE)
        calcinfo.retrieve_list.append(self._DEFAULT_OUTPUT_RESULTS_Z2PACK)
        
        return calcinfo
        
        
        
        
        
        
        
