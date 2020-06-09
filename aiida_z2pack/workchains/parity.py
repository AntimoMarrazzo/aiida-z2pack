"""`Z2QSHworkchain` workchain definition."""
from __future__ import absolute_import
import numpy as np

from aiida import orm
from aiida.common import AttributeDict, exceptions
from aiida.plugins import CalculationFactory, WorkflowFactory
from aiida.engine import WorkChain, if_, ToContext, calcfunction

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
# from aiida_quantumespresso.calculations import _lowercase_dict

from six.moves import zip
from six.moves import range

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
BandsxCalculation = CalculationFactory('quantumespresso.bandsx')
Z2packBaseWorkChain = WorkflowFactory('z2pack.base')


@calcfunction
def generate_bands_input_parameters() -> orm.Dict:
    """Generate the input dictionary for a BandsxCalculation."""
    parameters = {
        'BANDS': {
            'parity': True,
            'lsym': False,
        }
    }

    return orm.Dict(dict=parameters)


@calcfunction
def generate_trim(structure: orm.StructureData,
                  dimensionality: orm.Int) -> orm.KpointsData:
    """Generate the TRIM point KpointsData for the given structure and dimensionality."""
    from itertools import product
    null = [0.0]
    l = [0.0, 0.5]

    dim = dimensionality.value

    if dim == 2:
        grid = np.array(list(product(l, l, null)))
    elif dim == 3:
        grid = np.array(list(product(l, l, l)))
    else:
        raise exceptions.InputValidationError(
            'Invalid dimensionality {}'.format(dim))

    res = orm.KpointsData()
    res.set_cell_from_structure(structure)
    res.set_kpoints(grid)

    return res


@calcfunction
def calculate_invariant_with_parities(dimensionality: orm.Int,
                                      scf_out_params: orm.Dict,
                                      par_data: orm.ArrayData) -> orm.Dict:
    """Calculate the z2 invariant from the parities using the output of a BandsxCalculation."""
    dim = dimensionality.value

    parities = par_data.get_array('par')

    n_el = int(scf_out_params.get_dict()['number_of_electrons'])
    if dim == 2:
        x = 1
        for p in parities:
            delta = 1
            for i in range(0, n_el, 2):
                delta *= p[i]

            x *= delta

        if x == 1:
            res = {'nu': 0}
        elif x == -1:
            res = {'nu': 1}
        else:
            raise exceptions.OutputParsingError(
                'Invalid result for z2 using parities')

    elif dim == 3:
        raise NotImplemented('dimensionality = 3  not implemented.')
    else:
        raise exceptions.InputValidationError(
            'dimensionality must be either 2 or 3')

    return orm.Dict(dict=res)


@calcfunction
def extract_z2_from_z2pack(z2pack_output_parameters: orm.Dict) -> orm.Dict:
    """Extract the Z2 invariant result form a z2pack calculation."""
    dct = z2pack_output_parameters.get_dict()

    res = {}

    res['nu'] = dct['invariant']['Z2']

    return orm.Dict(dict=res)


class Z2QSHworkchain(WorkChain):
    """Workchain to compute the Z2 topological invariant using z2pack or parities when possible."""
    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)

        # INPUTS ############################################################################
        spec.input(
            'pw_code', valid_type=orm.Code,
            help='The code for pw.x calculations.'
            )
        spec.input(
            'bands_code', valid_type=orm.Code,
            help='The code for bands.x calculations.'
            )
        spec.input(
            'structure', valid_type=orm.StructureData,
            required=False,
            help='The inputs structure.'
            )
        spec.input(
            'parent_folder', valid_type=orm.RemoteData,
            required=False,
            help='The remote_folder of an scf calculation to be used by z2pack.'
            )
        spec.input(
            'use_parity', valid_type=orm.Bool,
            required=False,
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )


        spec.input(
            'dimensionality', valid_type=orm.Int,
            required=False,
            help='The dimensionality of the system (2 or 3) to be used for the TRIM coordinate generation.'
            )

        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.code'),
            namespace_options={
                'required':False, 'populate_defaults':False,
                'help': 'Inputs for the `PwBaseWorkChain` scf calculation.'
                }
            )
        spec.expose_inputs(
            PwBaseWorkChain, namespace='band',
            exclude=('clean_workdir', 'pw.structure', 'pw.code', 'kpoints'),
            namespace_options={
                'required':False, 'populate_defaults':False,
                'help': 'Inputs for the `PwBaseWorkChain` band calculation.'
                }
            )
        spec.expose_inputs(
            Z2packBaseWorkChain, namespace='z2pack_base',
            exclude=('clean_workdir', 'structure', 'parent_folder', 'pw_code', 'scf'),
            namespace_options={
                'required':False, 'populate_defaults':False,
                'help': 'Inputs for the `Z2packBaseWorkChain`.'
                }
            )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            if_(cls.should_do_scf)(
                cls.run_scf,
                cls.inspect_scf
            ),
            if_(cls.should_use_parity)(
                cls.calculate_trim_wf,
                cls.inspect_trim_wf,
                cls.calculate_trim_parity,
                cls.inspect_trim_parity,
                cls.calculate_z2_with_parity,
            ).else_(
                cls.prepare_z2pack,
                cls.run_z2pack,
                cls.inspect_z2pack,
                ),
            cls.results
            )

        # OUTPUTS ############################################################################
        spec.output(
            'output_parameters', valid_type=orm.Dict,
            help='Dict containing the result for the z2 invariant calculation.'
            )
        # spec.output(
        #     'output_parameters', valid_type=orm.Dict,
        #     help='Dict containing the result for the z2 invariant calculation.'
        #     )
        spec.output('scf_remote_folder', valid_type=orm.RemoteData,
            required=False,
            help='The remote folder produced by the scf calculation.'
            )

        # ERRORS ############################################################################
        spec.exit_code(322, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='the scf PwBaseWorkChain sub process failed')
        spec.exit_code(324, 'ERROR_SUB_PROCESS_FAILED_BANDS',
            message='the bands PwBaseWorkChain sub process failed')
        spec.exit_code(325, 'ERROR_SUB_PROCESS_FAILED_BANDSX',
            message='the bands.x sub process failed')
        spec.exit_code(323, 'ERROR_SUB_PROCESS_FAILED_Z2PACK',
            message='the Z2packBaseWorkChain sub process failed')
        spec.exit_code(333, 'ERROR_INVALID_INPUT',
            message='Must provide either `scf` namelist or `parent_folder` RemoteData as input.')
        spec.exit_code(433, 'ERROR_INVALID_Z2_RESULT',
            message='Must provide either `scf` namelist or `parent_folder` RemoteData as input.')
        # yapf: enable

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        if 'scf' in self.inputs:
            self.ctx.current_structure = self.inputs.structure
        else:
            if not 'parent_folder' in self.inputs:
                return self.exit_codes.ERROR_INVALID_INPUT

            self.report('setting `parent_folder` from input')

            self.ctx.scf_folder = self.inputs.parent_folder

            pw_calc = self.inputs.parent_folder.creator

            self.ctx.current_structure = pw_calc.inputs.structure
            self.ctx.scf_out_params = pw_calc.outputs.output_parameters

    def should_do_scf(self):
        """Determine if the scf calculaiton should be run."""
        return 'scf' in self.inputs

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the input structure."""
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.pw.code = self.inputs.pw_code
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(
            running.pk, 'scf'))

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                'scf PwBaseWorkChain failed with exit status {}'.format(
                    workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.scf_folder = workchain.outputs.remote_folder
        self.ctx.scf_out_params = workchain.outputs.output_parameters

        self.out('scf_remote_folder', self.ctx.scf_folder)

    def should_use_parity(self):
        """Check whether parities can/should be used for th calculations instead of z2pack."""
        if 'use_parity' in self.inputs:
            return self.inputs.use_parity.value

        dct = self.ctx.scf_out_params.get_dict()

        inv_check = dct['inversion_symmetry']

        if not inv_check:
            self.report(
                "Can't use parities because the system lack inversion symmetry."
            )

        symmorph_check = True

        if not symmorph_check:
            self.report(
                'Calculation of parities not implemented for non-symmorphic systems. Defaulting to z2pack.'
            )

        return inv_check and symmorph_check

    def calculate_trim_wf(self):
        """Launch a pw_bands calculation on the TRIM points for the structure."""
        self.report(
            'Using parities at TRIM points to calculatte Z2 invariant.')
        kpoints = generate_trim(self.ctx.current_structure,
                                self.inputs.dimensionality)

        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='band'))
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.pw.code = self.inputs.pw_code
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters['CONTROL']['calculation'] = 'bands'
        inputs.pw.parent_folder = self.ctx.scf_folder
        inputs.kpoints = kpoints

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(
            running.pk, 'bands'))

        return ToContext(workchain_trim=running)

    def inspect_trim_wf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_trim

        if not workchain.is_finished_ok:
            self.report(
                'bands PwBaseWorkChain failed with exit status {}'.format(
                    workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.ctx.band_folder = workchain.outputs.remote_folder

    def calculate_trim_parity(self):
        """Run a BandsxCalculation to get the wf parities."""
        params = generate_bands_input_parameters()

        inputs = {
            'code': self.inputs.bands_code,
            'parameters': params,
            'parent_folder': self.ctx.band_folder,
            'metadata': {
                'options': {
                    'resources': {
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 1
                    },
                    'max_wallclock_seconds': 3600,
                    'parser_name': 'quantumespresso.bandsx'
                }
            }
        }

        inputs['metadata']['options'][
            'account'] = self.inputs.band.pw.metadata.options.account

        running = self.submit(BandsxCalculation, **inputs)

        self.report('launching BandsxCalculation<{}>'.format(running.pk))

        return ToContext(workchain_parity=running)

    def inspect_trim_parity(self):
        """Verify that the BandsxCalculation for wf parities run finished successfully."""
        workchain = self.ctx.workchain_parity

        if not workchain.is_finished_ok:
            self.report('BandsxCalculation failed with exit status {}'.format(
                workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDSX

        self.ctx.parities = workchain.outputs.filband

    def calculate_z2_with_parity(self):
        """Calculate the z2 unvariant from the parities result."""
        scf_out_params = self.ctx.scf_folder.creator.outputs.output_parameters
        self.ctx.z2 = calculate_invariant_with_parities(
            self.inputs.dimensionality, scf_out_params, self.ctx.parities)

    def prepare_z2pack(self):
        """Prepare the inputs for the z2ack calculation."""
        self.report('Using z2pack to calculatte Z2 invariant.')
        inputs = AttributeDict(
            self.exposed_inputs(Z2packBaseWorkChain, namespace='z2pack_base'))
        inputs.pw_code = self.inputs.pw_code
        inputs.structure = self.ctx.current_structure
        inputs.parent_folder = self.ctx.scf_folder

        settings = inputs.z2pack.z2pack_settings.get_dict()
        settings.update({
            'dimension_mode': '2D',
            'invariant': 'Z2',
        })

        self.ctx.inputs = inputs

    def run_z2pack(self):
        """Launch the Z2packBaseWorkChain for the z2pack calculation."""
        running = self.submit(Z2packBaseWorkChain, **self.ctx.inputs)

        self.report(
            'launching Z2packBaseWorkChain<{}> for Z2 calculation'.format(
                running.pk))

        self.to_context(workchain_z2pack=running)

    def inspect_z2pack(self):
        """Verify that the Z2packBaseWorkChain finished successfully."""
        workchain = self.ctx.workchain_z2pack

        if not workchain.is_finished_ok:
            self.report(
                'Z2packBaseWorkChain failed with exit status {}'.format(
                    workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_Z2PACK

        self.ctx.z2 = extract_z2_from_z2pack(
            workchain.outputs.output_parameters)

    def results(self):
        """Output the workchain results."""
        self.out('output_parameters', self.ctx.z2)

        self.report('FINISHED')
