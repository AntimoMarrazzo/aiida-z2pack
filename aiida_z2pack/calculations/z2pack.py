# -*- coding: utf-8 -*-

import os
import six
import copy

from aiida import orm
from aiida.engine import CalcJob
from aiida.plugins import CalculationFactory
from aiida.common import datastructures, exceptions
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida_quantumespresso.calculations.namelists import NamelistsCalculation

from .utils import prepare_scf, prepare_nscf, prepare_overlap, prepare_wannier90, prepare_z2pack

PwCalculation           = CalculationFactory('quantumespresso.pw')
Wannier90Calculation    = CalculationFactory('wannier90.wannier90')
# Pw2wannier90Calculation = CalculationFactory('quantumespresso.pw2wannier90')

bases = [
    # Wannier90Calculation,
    NamelistsCalculation,
    PwCalculation
    ]
# bases = [CalcJob]

class Z2packCalculation(*bases):
    """
    Plugin for Z2pack, a code for computing topological invariants.
    See http://z2pack.ethz.ch/ for more details
    """
    # _PSEUDO_SUBFOLDER = './pseudo/'
    _OUTPUT_SUBFOLDER = './out/'
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
    # _default_verbosity = 'high'

    # _use_kpoints = False
    # _DEFAULT_OUTPUT_FILE = 'z2pack_aiida.out'
    # _OUTPUT_Z2PACK_FILE  = 'aiida.json'
    _INPUT_PW_SCF_FILE   = 'aiida.scf.in'
    _OUTPUT_PW_SCF_FILE  = 'aiida.scf.out'

    _INPUT_PW_NSCF_FILE  = 'aiida.nscf.in'
    _OUTPUT_PW_NSCF_FILE = 'aiida.nscf.out'

    _INPUT_Z2PACK_FILE   = 'z2pack_aiida.py'
    _OUTPUT_Z2PACK_FILE  = 'z2pack_aiida.out'
    _OUTPUT_SAVE_FILE    = 'save.json'
    _OUTPUT_RESULT_FILE  = 'results.json'

    _INPUT_W90_FILE      = 'aiida.win'
    _OUTPUT_W90_FILE     = 'aiida.wout'

    _INPUT_OVERLAP_FILE  = 'aiida.pw2wan.in'
    _OUTPUT_OVERLAP_FILE = 'aiida.pw2wan.out'

    _ERROR_FILE          = 'aiida.werr'

    _PREFIX              = 'aiida'
    _SEEDNAME            = 'aiida'
    _default_parser      = 'z2pack'  
    _INPUT_SUBFOLDER     = "./out/"
    _ALWAYS_SYM_FILES    = ['UNK*', '*.mmn']
    _RESTART_SYM_FILES   = ['*.amn','*.eig']
    _CHK_FILE            = '*.chk'
    _DEFAULT_INIT_ONLY   = False
    _DEFAULT_WRITE_UNK   = False

    _DEFAULT_MIN_NEIGHBOUR_DISTANCE = 0.01
    _DEFAULT_NUM_LINES              = 11
    _DEFAULT_ITERATOR               = 'range(8, 27, 2)'

    _DEFAULT_GAP_TOLERANCE  = 0.3
    _DEFAULT_MOVE_TOLERANCE = 0.3
    _DEFAULT_POS_TOLERANCE  = 0.01

    _blocked_keywords_pw = PwCalculation._blocked_keywords
    _blocked_keywords_overlap = [
        ('INPUTPP', 'outdir', _OUTPUT_SUBFOLDER),
        ('INPUTPP', 'prefix', _PREFIX),
        ('INPUTPP', 'seedname', _SEEDNAME),
        ('INPUTPP', 'write_amn', False),
        ('INPUTPP', 'write_mmn', True),
        ]
    _blocked_keywords_wannier90 = [
        ('length_unit','ang'),
        ('spinors', True)
        ]

    @classmethod
    def define(cls, spec):
        super(Z2packCalculation, cls).define(spec)

        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The input structure.'
            )
        spec.input_namespace(
            'pseudos', valid_type=orm.UpfData, dynamic=True,
            help='A mapping of `UpfData` nodes onto the kind name to which they should apply.'
            )
        spec.input(
            'pw_parameters', valid_type=orm.Dict,
            required=True,
            help='Dict: Input parameters for the nscf code (pw).'
            )
        spec.input(
            'overlap_parameters', valid_type=orm.Dict, 
            default=orm.Dict(dict={}),
            help='Dict: Input parameters for the overlap code (pw2wannier).'
            )
        spec.input(
            'wannier90_parameters', valid_type=orm.Dict,
            required=True,
            help='Dict: Input parameters for the wannier code (wannier90).'
            )
        spec.input(
            'pw_settings', valid_type=orm.Dict,
            default=orm.Dict(dict={}),
            help='Use an additional node for special settings.'
            )
        spec.input(
            'overlap_settings', valid_type=orm.Dict,
            default=orm.Dict(dict={}),
            help='Use an additional node for special settings.'
            )
        spec.input(
            'wannier90_settings', valid_type=orm.Dict,
            default=orm.Dict(dict={}),
            help='Use an additional node for special settings.'
            )
        spec.input(
            'z2pack_settings', valid_type=orm.Dict, 
            required=True,
            help='Use an additional node for special settings.'
            )
        spec.input(
            'nscf_code', valid_type=orm.Code,
            required=True,
            help='NSCF code to be used by z2pack.'
            )
        spec.input(
            'overlap_code', valid_type=orm.Code,
            required=True,
            help='Overlap code to be used by z2pack.'
            )
        spec.input(
            'wannier90_code', valid_type=orm.Code, 
            required=True,
            help='Wannier code to be used by z2pack.'
            )
        spec.input(
            'z2pack_code', valid_type=orm.Code, 
            required=True,
            help='Z2pack code.'
            )
        # spec.input(
        #     'metadata.options.nscf_parser_name', valid_type=six.string_types, 
        #     default='quantumespresso.pw'
        #     )

        spec.output(
            'output_parameters', valid_type=orm.Dict, required=True,
            help='The `output_parameters` output node of the successful calculation.'
            )

    def prepare_for_submission(self, folder):
        prepare_scf(self, folder)
        prepare_nscf(self, folder)
        prepare_overlap(self, folder)
        prepare_wannier90(self, folder)
        prepare_z2pack(self, folder)

        # codeinfo = datastructures.orm.CodeInfo()
        # # codeinfo.cmdline_params = self.inputs.parameters.cmdline_params(
        # #     file1_name=self.inputs.file1.filename,
        # #     file2_name=self.inputs.file2.filename)
        # codeinfo.code_uuid = self.inputs.code.uuid
        # # codeinfo.stdout_name = self.metadata.options.output_filename
        # codeinfo.withmpi = self.inputs.metadata.options.withmpi

        # # Prepare a `CalcInfo` to be returned to the engine
        # calcinfo = datastructures.CalcInfo()
        # calcinfo.codes_info = [codeinfo]
        # # calcinfo.local_copy_list = [
        # #     (self.inputs.file1.uuid, self.inputs.file1.filename, self.inputs.file1.filename),
        # #     (self.inputs.file2.uuid, self.inputs.file2.filename, self.inputs.file2.filename),
        # # ]
        # # calcinfo.retrieve_list = [self.metadata.options.output_filename]

        ################################################################################3
        # calcinfo = datastructures.CalcInfo()

        # calcinfo.uuid = str(self.uuid)
        # # Empty command line by default
        # cmdline_params = settings.pop('CMDLINE', [])
        # # we commented calcinfo.stin_name and added it here in cmdline_params
        # # in this way the mpirun ... pw.x ... < aiida.in
        # # is replaced by mpirun ... pw.x ... -in aiida.in
        # # in the scheduler, _get_run_line, if cmdline_params is empty, it
        # # simply uses < calcinfo.stin_name
        # calcinfo.cmdline_params = (list(cmdline_params) + ['-in', self.metadata.options.input_filename])

        # codeinfo = datastructures.CodeInfo()
        # codeinfo.cmdline_params = (list(cmdline_params) + ['-in', self.metadata.options.input_filename])
        # codeinfo.stdout_name = self.metadata.options.output_filename
        # codeinfo.code_uuid = self.inputs.code.uuid
        # calcinfo.codes_info = [codeinfo]

        # calcinfo.local_copy_list = local_copy_list
        # calcinfo.remote_copy_list = remote_copy_list
        # calcinfo.remote_symlink_list = remote_symlink_list

        # # Retrieve by default the output file and the xml file
        # calcinfo.retrieve_list = []
        # calcinfo.retrieve_list.append(self.metadata.options.output_filename)
        # calcinfo.retrieve_list.extend(self.xml_filepaths)
        # calcinfo.retrieve_list += settings.pop('ADDITIONAL_RETRIEVE_LIST', [])
        # calcinfo.retrieve_list += self._internal_retrieve_list
        ################################################################################

        # calcinfo = calcinfo_scf
        calcinfo = datastructures.CalcInfo()

        codeinfo = datastructures.CodeInfo()
        codeinfo.cmdline_params = (['>', self._OUTPUT_Z2PACK_FILE])
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.code_uuid = self.inputs.code.uuid
        calcinfo.codes_info = [codeinfo]

        calcinfo.retrieve_list           = []
        calcinfo.retrieve_temporary_list = []
        calcinfo.local_copy_list         = []
        calcinfo.remote_symlink_list     = []

        inputs = [
            self._INPUT_PW_SCF_FILE,
            self._INPUT_PW_NSCF_FILE,
            self._INPUT_W90_FILE,
            self._INPUT_OVERLAP_FILE,
            self._INPUT_Z2PACK_FILE,
            ]
        outputs = [
            self._OUTPUT_PW_SCF_FILE,
            self._OUTPUT_PW_NSCF_FILE,
            self._OUTPUT_W90_FILE,
            self._OUTPUT_OVERLAP_FILE,
            self._OUTPUT_Z2PACK_FILE,
            self._OUTPUT_SAVE_FILE,
            self._OUTPUT_RESULT_FILE,
            ]
        errors = [
            self._ERROR_FILE
            ]

        calcinfo.retrieve_list.extend(inputs)
        calcinfo.retrieve_list.extend(outputs)
        calcinfo.retrieve_list.extend(errors)

        return calcinfo

    
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
                    
            _internal_retrieve_list = [self._INPUT_NSCF_FILE,
                                   self._INPUT_W90_FILE,
                                   self._OUTPUT_Z2PACK_FILE,
                                   self._INPUT_OVERLAP_FILE,
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
        calcinfo.retrieve_list.append(self._OUTPUT_Z2PACK_FILE)
        calcinfo.retrieve_list.append(self._DEFAULT_OUTPUT_FILE)
        calcinfo.retrieve_list.append(self._DEFAULT_OUTPUT_RESULTS_Z2PACK)
        
        return calcinfo
        
        
        
        
        
        
        
