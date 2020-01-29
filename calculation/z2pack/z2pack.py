# -*- coding: utf-8 -*-
# import copy, os
# import numpy as np
# from aiida.common.utils import classproperty
# from aiida.common.exceptions import exceptions.InputValidationError, ModificationNotAllowed
# from aiida.common.datastructures import CalcInfo, orm.CodeInfo, code_run_modes
# from aiida.orm import JobCalculation, DataFactory
# from aiida_quantumespresso.calculations import (
#     _lowercase_dict, _uppercase_dict, get_input_data_text)
# from aiida_quantumespresso.calculations.pw import PwCalculation
# from aiida.orm.code import orm.Code
# from aiida.orm.data.array.kpoints import KpointsData
# from aiida.orm.data.upf import orm.UpfData
# from aiida.orm.data.upf import get_pseudos_from_structure
# from aiida.orm.data.orbital import OrbitalData, OrbitalFactory
# from aiida.orm.data.parameter import orm.Dict
# from aiida.orm.data.remote import orm.RemoteData
# from aiida.orm.data.structure import orm.StructureData
# from aiida.transport import Transport

import os
import six
import copy

from aiida import orm
# from aiida.engine import CalcJob
from aiida.plugins import CalculationFactory
from aiida.common import datastructures, exceptions
from aiida.orm.nodes.data.upf import get_pseudos_from_structure

from .utils import prepare_nscf, prepare_wannier90, prepare_z2pack

PwCalculation = CalculationFactory('quantumespresso.pw')

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


bases = [PwCalculation,]

class Z2packCalculation(*bases):
    """
    Plugin for Z2pack, a code for computing topological invariants.
    See http://z2pack.ethz.ch/ for more details
    """
    # _PSEUDO_SUBFOLDER = './pseudo/'
    # _OUTPUT_SUBFOLDER = './out/'
    # _PREFIX = 'aiida'
    _INPUT_NSCF_FILE_NAME = 'aiida.nscf.in'
    _OUTPUT_NSCF_FILE_NAME = 'aiida.nscf.out'
    # _DATAFILE_XML_PRE_6_2 = 'data-file.xml'
    # _DATAFILE_XML_POST_6_2 = 'data-file-schema.xml'
    _Z2pack_folder = './'
    _Z2pack_folder_restart_files=[]


    ## Default PW output parser provided by AiiDA
    # to be defined in the subclass

    # _automatic_namelists = {}

    # in restarts, will not copy but use symlinks
    _default_symlink_usage = True

    # in restarts, it will copy from the parent the following
    _restart_copy_from_z2pack = os.path.join(_Z2pack_folder, '*')

    # in restarts, it will copy the previous folder in the following one
    _restart_copy_to_z2pack = _Z2pack_folder
    # Default verbosity; change in subclasses
    _default_verbosity = 'high'

    _use_kpoints = False
    _DEFAULT_OUTPUT_FILE = 'z2pack_aiida.out'
    _DEFAULT_INPUT_FILE='z2pack_aiida.py'
    _DEFAULT_OUTPUT_Z2PACK = 'aiida.json'
    _DEFAULT_OUTPUT_RESULTS_Z2PACK = 'results.json'
    _INPUT_W90_FILE_NAME = 'aiida.win'
    _OUTPUT_W90_FILE_NAME = 'aiida.wout'
    _ERROR_FILE_NAME = 'aiida.werr'
    _INPUT_OVERLAP_FILE_NAME = 'aiida.pw2wan.in'
    _OUTPUT_OVERLAP_FILE_NAME = 'aiida.pw2wan.out'
    _PREFIX = 'aiida'
    _SEEDNAME = 'aiida'
    _default_parser = 'z2pack'  
    _INPUT_SUBFOLDER = "./out/"
    _ALWAYS_SYM_FILES = ['UNK*', '*.mmn']
    _RESTART_SYM_FILES = ['*.amn','*.eig']
    _CHK_FILE = '*.chk'
    _DEFAULT_INIT_ONLY = False
    _DEFAULT_WRITE_UNK = False
    _DEFAULT_MIN_NEIGHBOUR_DISTANCE = 0.01
    _DEFAULT_NUM_LINES = 11
    _DEFAULT_ITERATOR = 'range(8, 27, 2)'
    _DEFAULT_GAP_TOLERANCE = 0.3
    _DEFAULT_MOVE_TOLERANCE = 0.3
    _DEFAULT_POS_TOLERANCE = 0.01
    # _blocked_keywords = PwCalculation._blocked_keywords + [['length_unit','ang']]
    _blocked_keywords = [y for x in bases for y in x._blocked_keywords] + [['length_unit','ang']]
    _blocked_precode_keywords = []

    @classmethod
    def define(cls, spec):
        super(Z2packCalculation, cls).define(spec)
        # spec.input('metadata.options.input_filename', valid_type=six.string_types, default=cls._DEFAULT_INPUT_FILE)
        # spec.input('metadata.options.output_filename', valid_type=six.string_types, default=cls._DEFAULT_OUTPUT_FILE)
        # spec.input('metadata.options.withmpi', valid_type=bool, default=True)  # Override default withmpi=False
        # spec.input('parameters', valid_type=orm.Dict,
        #     help='The input parameters that are to be used to construct the input file.')
        # spec.input('vdw_table', valid_type=orm.SinglefileData, required=False,
        #     help='Optional van der Waals table contained in a `SinglefileData`.')

        spec.input_namespace(
            'nscf_parameters', valid_type=orm.Dict, dynamic=True,
            help='Dict: Input parameters for the nscf code (pw)'
            )
        spec.input_namespace(
            'overlap_parameters', valid_type=orm.Dict, dynamic=True,
            help='Dict: Input parameters for the overlap code (pw2wannier)'
            )
        spec.input_namespace(
            'wannier90_parameters', valid_type=orm.Dict, dynamic=True,
            help='Dict: Input parameters for the wannier code (wannier90)'
            )
        spec.input(
            'nscf_code', valid_type=orm.Code, required=False,
            help='NSCF code to be used by z2pack.'
            )
        spec.input(
            'overlap_code', valid_type=orm.Code, required=False,
            help='Overlap code to be used by z2pack.'
            )
        spec.input(
            'wannier90_code', valid_type=orm.Code, required=False,
            help='Wannier code to be used by z2pack.'
            )
        spec.input('metadata.options.nscf_parser_name', valid_type=six.string_types, default='quantumespresso.pw')

    def prepare_for_submission(self, folder):
        calcinfo_nscf    = PwCalculation.prepare_for_submission(self, folder)
        calcinfo_wannier = None
        calcinfo_z2pack  = prepare_z2pack(self, folder)
        # super(Z2packCalculation, self).prepare_for_submission(folder)

        codeinfo = datastructures.orm.CodeInfo()
        codeinfo.cmdline_params = self.inputs.parameters.cmdline_params(
            file1_name=self.inputs.file1.filename,
            file2_name=self.inputs.file2.filename)
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.withmpi = self.inputs.metadata.options.withmpi

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = [
            (self.inputs.file1.uuid, self.inputs.file1.filename, self.inputs.file1.filename),
            (self.inputs.file2.uuid, self.inputs.file2.filename, self.inputs.file2.filename),
        ]
        calcinfo.retrieve_list = [self.metadata.options.output_filename]

        return calcinfo

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
        elif isinstance(kind, str):
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
            return 0 if fixed else 1
    
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
        # return self.get_inputs_dict()[self.get_linkname('structure')]
        return self.get_incoming().get_node_by_label('structure')

    def _set_parent_remotedata(self, remotedata):
        """
        Used to set a parent remotefolder in the restart of ph.
        """
        from aiida.common.exceptions import ValidationError

        if not isinstance(remotedata, orm.RemoteData):
            raise ValueError('remotedata must be a orm.RemoteData')

        # complain if another remotedata is already found
        # input_remote = self.get_inputs(type=orm.RemoteData)
        input_remote = self.get_incoming(node_class=orm.RemoteData).all()
        if input_remote:
            raise ValidationError("Cannot set several parent calculation to a "
                                  "{} calculation".format(self.__class__.__name__))

        self.use_parent_folder(remotedata)
       
    def use_parent_calculation(self, calc):
        """
        Set the parent calculation,
        from which it will inherit the outputsubfolder.
        The link will be created from parent orm.RemoteData and NamelistCalculation
        """
        #if not isinstance(calc, PwCalculation):
        #    raise ValueError("Parent calculation must be a Pw ")
        if not isinstance(calc, (PwCalculation,Z2packCalculation)) :
            raise ValueError("Parent calculation must be a PW or Z2pack ")
        if isinstance(calc, PwCalculation):
            # Test to see if parent PwCalculation is nscf
            par_type = calc.inputs.parameters.dict.CONTROL['calculation'].lower()
            if par_type != 'scf':
                raise ValueError("Pw calculation must be scf") 
        try:
            # remote_folder = calc.get_outputs_dict()['remote_folder']
            remote_folder = calc.get_outgoing().get_node_by_label('remote_folder')
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
                be returned by get_inputdata_dict (without the orm.Code!)
        """        
        

                
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
            raise exceptions.InputValidationError("No settings specified for this calculation")

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
        
        c1 = orm.CodeInfo()
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
        
        
        
        
        
        
        
