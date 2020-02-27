# -*- coding: utf-8 -*-
import os
import six

from aiida import orm
from aiida.engine import CalcJob
from aiida.plugins import CalculationFactory
from aiida.common import datastructures, exceptions

from .utils import prepare_nscf, prepare_overlap, prepare_wannier90, prepare_z2pack
from .utils import merge_dict_input_to_root, recursive_get_linked_node

from aiida_quantumespresso.calculations import _lowercase_dict

PwCalculation = CalculationFactory('quantumespresso.pw')

class Z2packCalculation(CalcJob):
    """
    Plugin for Z2pack, a code for computing topological invariants.
    See http://z2pack.ethz.ch/ for more details
    """
    _PSEUDO_SUBFOLDER = './pseudo/'
    _PREFIX              = 'aiida'
    _SEEDNAME            = 'aiida'

    _INPUT_SUBFOLDER     = "./out/" #Still used by some workchains?
    _OUTPUT_SUBFOLDER    = "./out/"

    _INPUT_PW_SCF_FILE   = 'aiida.scf.in'
    _OUTPUT_PW_SCF_FILE  = 'aiida.scf.out'

    _INPUT_PW_NSCF_FILE  = 'aiida.nscf.in'
    _OUTPUT_PW_NSCF_FILE = 'aiida.nscf.out'

    _INPUT_Z2PACK_FILE   = 'z2pack_aiida.py'
    _OUTPUT_Z2PACK_FILE  = 'z2pack_aiida.out'
    _OUTPUT_SAVE_FILE    = 'save.json'
    _OUTPUT_RESULT_FILE  = 'results.json'

    _INPUT_W90_FILE      = _SEEDNAME + '.win'
    _OUTPUT_W90_FILE     = _SEEDNAME + '.wout'

    _INPUT_OVERLAP_FILE  = 'aiida.pw2wan.in'
    _OUTPUT_OVERLAP_FILE = 'aiida.pw2wan.out'

    _ERROR_W90_FILE      = _SEEDNAME + '.werr'
    _ERROR_PW_FILE       = 'CRASH'

    _DEFAULT_MIN_NEIGHBOUR_DISTANCE = 0.01
    _DEFAULT_NUM_LINES              = 11
    _DEFAULT_ITERATOR               = 'range(8, 41, 2)'
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
        ('spinors', True),
        ('num_iter', 0),
        ('use_bloch_phases', True),
        ]

    @classmethod
    def define(cls, spec):
        super(Z2packCalculation, cls).define(spec)

        # INPUTS ###########################################################################
        spec.input('metadata.options.parser_name', valid_type=six.string_types, default='z2pack.z2pack')
        spec.input(
            'parent_folder', valid_type=orm.RemoteData,
            required=True,
            help='Output of a previous scf/z2pack calculation (start a new z2pack calclulation)/(restart from an unfinished calculation)'
            )

        spec.input(
            'pw_parameters', valid_type=orm.Dict,
            required=False,
            help='Dict: Input parameters for the nscf calculations.'
            )
        spec.input(
            'overlap_parameters', valid_type=orm.Dict,
            required=False,
            help='Dict: Input parameters for the overlap code (pw2wannier).'
            )
        spec.input(
            'wannier90_parameters', valid_type=orm.Dict,
            required=False,
            help='Dict: Input parameters for the wannier code (wannier90).'
            )

        spec.input(
            'pw_settings', valid_type=orm.Dict,
            required=False,
            help='Use an additional node for special settings.'
            )
        spec.input(
            'z2pack_settings', valid_type=orm.Dict, 
            required=False,
            help='Use an additional node for special settings.'
            )

        spec.input(
            'pw_code', valid_type=orm.Code,
            required=False,
            help='NSCF code to be used by z2pack.'
            )
        spec.input(
            'overlap_code', valid_type=orm.Code,
            required=False,
            help='Overlap code to be used by z2pack.'
            )
        spec.input(
            'wannier90_code', valid_type=orm.Code, 
            required=False,
            help='Wannier code to be used by z2pack.'
            )
        spec.input(
            'code', valid_type=orm.Code, 
            required=False,
            help='Z2pack code.'
            )

        # OUTPUTS ###########################################################################
        spec.output(
            'output_parameters', valid_type=orm.Dict, required=True,
            help='The `output_parameters` output node of the successful calculation.'
            )
        spec.default_output_node = 'output_parameters'

        # EXIT CODES ###########################################################################
        spec.exit_code(
            200, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.'
            )
        spec.exit_code(
            300, 'ERROR_OUTPUT_FILES',
            message=(
                'Something failed during the calulation and no output was produced. '
                'Inpsect the stderr file for more information.'
                )
            )
        # spec.exit_code(
        #     310, 'ERROR_UNEXPECTED_FAILURE',
        #     message=(
        #         'Something failed during the calulation and no output was produced. '
        #         'Inpsect the stderr file for more information.'
        #         )
        #     )
        spec.exit_code(
            320, 'ERROR_PW_CRASH',
             message=(
                'Something failed during the pw/pw2wannier calulation. '
                'Inpsect the \'{}\' file for more information.'.format(cls._ERROR_PW_FILE)
                )
            )
        spec.exit_code(
            330, 'ERROR_W90_CRASH',
             message=(
                'Something failed during the wannier90 calulation. '
                'Inpsect the \'{}\' file for more information.'.format(cls._ERROR_W90_FILE)
                )
            )
        spec.exit_code(
            400, 'ERROR_MISSING_RESULTS_FILE',
            message=(
                'The result file \'{}\' is missing. '.format(cls._OUTPUT_RESULT_FILE)
                # 'Calculation interrupted by walltime.'
                )
            )
        spec.exit_code(
            410, 'ERROR_MISSING_SAVE_FILE',
            message='The save file `{}` is missing.'.format(cls._OUTPUT_SAVE_FILE)
            )

    def prepare_for_submission(self, folder):
        self.inputs.metadata.options.parser_name     = 'z2pack.z2pack'
        self.inputs.metadata.options.output_filename = self._OUTPUT_Z2PACK_FILE
        self.inputs.metadata.options.input_filename  = self._INPUT_Z2PACK_FILE

        if not 'overlap_parameters' in self.inputs:
            self.inputs.overlap_parameters = orm.Dict(dict={})

        calcinfo = datastructures.CalcInfo()

        codeinfo = datastructures.CodeInfo()
        codeinfo.stdout_name = self._OUTPUT_Z2PACK_FILE
        codeinfo.stdin_name  = self._INPUT_Z2PACK_FILE
        codeinfo.code_uuid   = self.inputs.code.uuid
        calcinfo.codes_info  = [codeinfo]

        calcinfo.codes_run_mode = datastructures.CodeRunMode.SERIAL
        calcinfo.cmdline_params = []

        calcinfo.retrieve_list           = []
        calcinfo.retrieve_temporary_list = []
        calcinfo.local_copy_list         = []
        calcinfo.remote_copy_list        = []
        calcinfo.remote_symlink_list     = []

        inputs  = [
            self._INPUT_PW_NSCF_FILE,
            self._INPUT_OVERLAP_FILE,
            self._INPUT_W90_FILE,
            ] 
        outputs = [
            self._OUTPUT_Z2PACK_FILE,
            self._OUTPUT_SAVE_FILE,
            self._OUTPUT_RESULT_FILE,
            ]
        errors = [
            os.path.join('build', a) for a in [self._ERROR_W90_FILE, self._ERROR_PW_FILE]
            ]

        calcinfo.retrieve_list.extend(outputs)
        calcinfo.retrieve_list.extend(errors)

        parent = self.inputs.parent_folder
        rpath  = parent.get_remote_path()
        uuid   = parent.computer.uuid
        parent_type = parent.creator.process_class

        if parent_type == Z2packCalculation:
            self._set_inputs_from_parent_z2pack()

        try:
            settings = _lowercase_dict(self.inputs.z2pack_settings.get_dict(), 'z2pack_settings')
        except AttributeError:
            raise exceptions.InputValidationError('Must provide `z2pack_settings` input for `scf` calculation.')
        symlink  = settings.get('parent_folder_symlink', False)
        self.restart_mode = settings.get('restart_mode', True)
        ptr = calcinfo.remote_symlink_list if symlink else calcinfo.remote_copy_list

        if parent_type == PwCalculation:
            self._set_inputs_from_parent_scf()

            prepare_nscf(self, folder)
            prepare_overlap(self, folder)
            prepare_wannier90(self, folder)
        elif parent_type == Z2packCalculation:
            if self.restart_mode:
                calcinfo.remote_copy_list.append(
                    (
                        uuid,
                        os.path.join(rpath, self._OUTPUT_SAVE_FILE),
                        self._OUTPUT_SAVE_FILE,
                    ))

            calcinfo.remote_copy_list.extend(
                [(uuid, os.path.join(rpath, inp), inp) for inp in inputs]
                )
        else:
            raise exceptions.ValidationError(
                "parent node must be either from a PWscf or a Z2pack calculation."
                )

        parent_files = [self._PSEUDO_SUBFOLDER, self._OUTPUT_SUBFOLDER]
        ptr.extend(
            [(uuid, os.path.join(rpath, fname), fname) for fname in parent_files]
            )

        prepare_z2pack(self, folder)

        return calcinfo

    def _set_inputs_from_parent_scf(self):
        parent = self.inputs.parent_folder
        calc   = parent.creator

        pseudos      = calc.get_incoming(link_label_filter='pseudos%').all()
        pseudos_dict = {name[9:]:upf for upf,_,name in pseudos}
        self.inputs.pseudos       = pseudos_dict
        self.inputs.structure     = calc.inputs.structure

        merge_dict_input_to_root(
            self,
            ('pw_parameters', 'parameters'),
            ('pw_settings', 'settings')
            )

    def _set_inputs_from_parent_z2pack(self):
        parent = self.inputs.parent_folder
        calc   = parent.creator
        # calc   = self._get_root_parent()

        for label in ['pw_code', 'overlap_code', 'wannier90_code', 'code']:
            if label in self.inputs:
                continue
            # old = calc.get_incoming(link_label_filter=label).first().node
            old = recursive_get_linked_node(calc, label, Z2packCalculation)
            setattr(self.inputs, label, old)

        merge_dict_input_to_root(
            self,
            ('pw_parameters', 'parameters'),
            'overlap_parameters',
            'wannier90_parameters',
            'pw_settings',
            'z2pack_settings'
            )
       
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
        
        