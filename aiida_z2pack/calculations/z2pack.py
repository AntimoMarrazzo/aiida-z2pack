# -*- coding: utf-8 -*-

import os

from aiida import orm
from aiida.engine import CalcJob
from aiida.plugins import CalculationFactory
from aiida.common import datastructures, exceptions
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
# from aiida_quantumespresso.calculations.namelists import NamelistsCalculation

from .utils import prepare_nscf, prepare_overlap, prepare_wannier90, prepare_z2pack

PwCalculation           = CalculationFactory('quantumespresso.pw')
# Wannier90Calculation    = CalculationFactory('wannier90.wannier90')
# Pw2wannier90Calculation = CalculationFactory('quantumespresso.pw2wannier90')

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

    _REVERSE_BUILD_SUBFOLDER     = ".."
    _BUILD_SUBFOLDER     = "./build"

    _Z2pack_folder = './'
    _Z2pack_folder_restart_files=[]
    _default_parser      = 'z2pack'  


    ## Default PW output parser provided by AiiDA
    # to be defined in the subclass

    # _automatic_namelists = {}

    # in restarts, will not copy but use symlinks
    _default_symlink_usage = True

    # in restarts, it will copy from the parent the following
    _restart_copy_from_z2pack = os.path.join(_Z2pack_folder, '*')

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

    _ERROR_W90_FILE          = _SEEDNAME + '.werr'

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
        ('INPUTPP', 'outdir', os.path.join(_REVERSE_BUILD_SUBFOLDER, _OUTPUT_SUBFOLDER)),
        # ('INPUTPP', 'outdir', _OUTPUT_SUBFOLDER),
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
            'parent_folder', valid_type=orm.RemoteData,
            required=True,
            help='Output of a previous scf/z2pack calculation (start a new z2pack calclulation)/(restart from an unfinished calculation)'
            )

        spec.input(
            'overlap_parameters', valid_type=orm.Dict,
            required=False,
            help='Dict: Input parameters for the overlap code (pw2wannier).'
            )
        spec.input(
            'wannier90_parameters', valid_type=orm.Dict,
            default=orm.Dict(dict={}),
            help='Dict: Input parameters for the wannier code (wannier90).'
            )

        spec.input(
            'pw_settings', valid_type=orm.Dict,
            required=False,
            help='Use an additional node for special settings.'
            )
        spec.input(
            'z2pack_settings', valid_type=orm.Dict, 
            required=True,
            help='Use an additional node for special settings.'
            )

        spec.input(
            'pw_code', valid_type=orm.Code,
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
            'code', valid_type=orm.Code, 
            required=True,
            help='Z2pack code.'
            )

        spec.output(
            'output_parameters', valid_type=orm.Dict, required=True,
            help='The `output_parameters` output node of the successful calculation.'
            )

        spec.exit_code(
            199, 'ERROR_UNEXPECTED_FAILURE',
            message='Something failed during the calulation. Inpsect the \'{}\' file for more information.'.format(cls._ERROR_W90_FILE)
            )
        spec.exit_code(
            200, 'ERROR_NO_RETRIEVED_FOLDER',
            message='The retrieved folder data node could not be accessed.'
            )
        spec.exit_code(
            201, 'ERROR_MISSING_RESULTS_FILE',
            message='The result file \'{}\' is missing!'.format(cls._OUTPUT_RESULT_FILE)
            )
        spec.exit_code(
            202, 'ERROR_MISSING_Z2PACK_OUTFILE',
            message='The output file of z2pack \'{}\' is missing!'.format(cls._OUTPUT_Z2PACK_FILE)
            )

    def prepare_for_submission(self, folder):
        from aiida.common.datastructures import CodeRunMode

        self.inputs.metadata.options.parser_name     = 'z2pack'
        self.inputs.metadata.options.output_filename = self._OUTPUT_Z2PACK_FILE
        self.inputs.metadata.options.input_filename  = self._INPUT_Z2PACK_FILE

        if not 'overlap_parameters' in self.inputs:
            self.inputs.overlap_parameters = orm.Dict(dict={})

        calcinfo = datastructures.CalcInfo()

        codeinfo = datastructures.CodeInfo()
        # codeinfo.cmdline_params = (['>', self._OUTPUT_Z2PACK_FILE])
        codeinfo.stdout_name = self._OUTPUT_Z2PACK_FILE
        codeinfo.stdin_name  = self._INPUT_Z2PACK_FILE
        codeinfo.code_uuid   = self.inputs.code.uuid
        calcinfo.codes_info  = [codeinfo]

        calcinfo.codes_run_mode = CodeRunMode.SERIAL
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

        calcinfo.retrieve_list.extend(outputs)

        parent = self.inputs.parent_folder
        rpath  = parent.get_remote_path()
        uuid   = parent.computer.uuid
        parent_type = self._get_parent_type()

        save_path       = os.path.join(self._OUTPUT_SUBFOLDER, '{}.save'.format(self._PREFIX))
        remote_xml_path = os.path.join(save_path, PwCalculation._DATAFILE_XML_POST_6_2)

        self.restart_mode = False
        if parent_type == PwCalculation:
            pw_calc = parent.get_incoming(node_class=PwCalculation).first().node

            pseudos = pw_calc.get_incoming(link_label_filter='pseudos%').all()
            pseudos_dict = {name[9:]:upf for upf,_,name in pseudos}

            self.inputs.pw_parameters = pw_calc.get_incoming(link_label_filter='parameters').first().node
            self.inputs.structure     = pw_calc.get_incoming(link_label_filter='structure').first().node
            self.inputs.pseudos       = pseudos_dict
            if not 'pw_settings' in self.inputs:
                settings = pw_calc.get_incoming(link_label_filter='settings').first()
                if settings is None:
                    self.inputs.pw_settings = orm.Dict(dict={})
                else:
                    self.inputs.pw_settings = settings

            prepare_nscf(self, folder)
            prepare_overlap(self, folder)
            prepare_wannier90(self, folder)

            # Hack the data-file-schema.xml to get pseudos from ../pseudo instead of ./pseudo
            sub_out  = folder.get_subfolder(self._OUTPUT_SUBFOLDER, create=True)
            sub_save = sub_out.get_subfolder('{}.save'.format(self._PREFIX), create=True)
            parent.getfile(remote_xml_path, folder.get_abs_path('app.xml'))

            with folder.open('app.xml') as f:
                xml_content = f.read()

            xml_content = xml_content.replace('./pseudo', '../pseudo')

            with sub_save.open(PwCalculation._DATAFILE_XML_POST_6_2, 'w') as f:
                f.write(xml_content)

            calcinfo.remote_copy_list.append(
                (
                    uuid,
                    os.path.join(rpath, save_path, 'charge-density.dat'),
                    os.path.join(save_path, 'charge-density.dat'),
                ))

        elif parent_type == Z2packCalculation:
            self.restart_mode = True
            calcinfo.remote_copy_list.extend(
                [(uuid, os.path.join(rpath, inp), inp) for inp in inputs]
                )
            calcinfo.remote_copy_list.append(
                (
                    uuid,
                    os.path.join(rpath, save_path),
                    save_path,
                ))
            calcinfo.remote_copy_list.append(
                (
                    uuid,
                    os.path.join(rpath, self._OUTPUT_SAVE_FILE),
                    self._OUTPUT_SAVE_FILE,
                ))
        else:
            raise exceptions.ValidationError(
                "parent node must be either from a PWscf or a Z2pack calculation."
                )


        calcinfo.remote_copy_list.append(
            (
                uuid,
                os.path.join(rpath, self._PSEUDO_SUBFOLDER),
                self._PSEUDO_SUBFOLDER
            ))

        prepare_z2pack(self, folder)

        return calcinfo

    def _get_parent_type(self):
        parent = self.inputs.parent_folder

        scf = parent.get_incoming(node_class=PwCalculation)
        if scf:
            return PwCalculation

        z2pack = parent.get_incoming(node_class=Z2packCalculation)
        if z2pack:
            return Z2packCalculation
    
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
        
        
        
