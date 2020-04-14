"""`RefineCrossingsPosition` workchain definition."""
from __future__ import absolute_import
import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain, while_, if_, ToContext

from .functions import (generate_kpt_cross, analyze_kpt_cross,
                        finilize_cross_results)
from six.moves import zip

PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


class RefineCrossingsPosition(WorkChain):
    """Refine the position of a crossing point by moving it along th x,y,z directions in a cross pattern."""
    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)

        # INPUTS ############################################################################
        spec.input('code', valid_type=orm.Code, help='The PWscf code.')

        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The inputs structure.'
            )
        spec.input(
            'parent_folder', valid_type=orm.RemoteData,
            required=False,
            help='The remote data of a previously performed scf calculation.'
            )
        spec.input_namespace(
            'pseudos', valid_type=orm.UpfData,
            dynamic=True,
            help='A mapping of `UpfData` nodes onto the kind name to which they should apply.'
            )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )

        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.code', 'pw.pseudos'),
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for the BANDS calculation.'
            })

        spec.expose_inputs(
            PwBaseWorkChain, namespace='bands',
            exclude=('clean_workdir', 'pw.structure', 'pw.code', 'pw.pseudos', 'pw.parent_folder'),
            namespace_options={
                'required': True, 'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for the BANDS calculation.'
            })

        spec.input('crossings',
                   valid_type=orm.ArrayData,
                   help='Starting crossings to refine.')
        spec.input(
            'step_size',
            valid_type=orm.Float,
            default=orm.Float(1.E-4),
            help='Distance between center point and edges of the cross.')
        spec.input(
            'gap_threshold',
            valid_type=orm.Float,
            default=orm.Float(0.0005),
            help=
            'kpoints with gap < `gap_threshold` are considered possible crossings.'
        )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            if_(cls.do_scf)(
                cls.setup_scf,
                cls.run_scf,
                cls.inspect_scf
            ).else_(
                cls.setup_remote
                ),
            while_(cls.do_loop)(
                cls.setup_kpt,
                cls.setup_bands,
                cls.run_bands,
                cls.inspect_bands,
                cls.analyze_bands
                ),
            cls.results)

        # OUTPUTS ############################################################################
        spec.output(
            'scf_remote_folder',
            valid_type=orm.RemoteData,
            required=True,
            help='The remote data produced by the `scf` calculation.'
        )
        spec.output(
            'crossings',
            valid_type=orm.ArrayData,
            required=True,
            help='The array containing a list of bands crossing found as rows.'
        )

        # ERRORS ############################################################################
        spec.exit_code(322,
                       'ERROR_SUB_PROCESS_FAILED_SCF',
                       message='the bands PwBaseWorkChain sub process failed')
        spec.exit_code(332,
                       'ERROR_SUB_PROCESS_FAILED_BANDS',
                       message='the bands PwBaseWorkChain sub process failed')
        # yapf: enable

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.counter = 0

        app = self.inputs.crossings.get_array('crossings')
        ncross = len(app)
        self.ctx.ncross = ncross

        self.ctx.step = self.inputs.step_size
        self.ctx.gap_thr = self.inputs.gap_threshold

        self.ctx.current_kpt = [self.inputs.crossings]
        self.ctx.skip_kpt = [0] * ncross

    def do_scf(self):
        """Determine whether the `scf` calculation should be erformed."""
        return 'scf' in self.inputs

    def setup_scf(self):
        """Set the inputs for the `scf` calculation."""
        if 'parent_folder' in self.inputs:
            self.report('IGNORING `parent_folder` input.')

        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.code = self.inputs.code
        inputs.pw.pseudos = self.inputs.pseudos
        inputs.pw.structure = self.inputs.structure
        inputs.clean_workdir = self.inputs.clean_workdir

        self.ctx.inputs = inputs

    def run_scf(self):
        """Run the scf calculation."""
        running = self.submit(PwBaseWorkChain, **self.ctx.inputs)

        self.report('launching PwBaseWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report('PwBaseWorkChain failed with exit status {}'.format(
                workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.remote = workchain.outputs.remote_folder

        self.out('scf_remote_folder', self.ctx.remote)

    def setup_remote(self):
        """Set remote from previous calculation."""
        self.ctx.remote = self.inputs.parent_folder

    def do_loop(self):
        """Check whether to stop the loop."""
        return not all(self.ctx.skip_kpt) and self.ctx.counter < 50

    def setup_kpt(self):
        """Create the cross grid for the calculation."""
        unique, counts = np.unique(self.ctx.skip_kpt, return_counts=True)
        app = dict(list(zip(unique, counts)))
        self.report('Starting iteration number <{:3d}> for {}/{} kpts'.format(
            self.ctx.counter, app[0], self.ctx.ncross))
        curr_kpt = self.ctx.current_kpt[-1]
        self.ctx.kpt_data = generate_kpt_cross(self.inputs.structure, curr_kpt,
                                               self.ctx.step)

        self.ctx.counter += 1

    def setup_bands(self):
        """Set the inputs for the `bands` calculation."""
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='bands'))

        inputs.pw.code = self.inputs.code
        inputs.kpoints = self.ctx.kpt_data

        inputs.pw.pseudos = self.inputs.pseudos
        inputs.pw.structure = self.inputs.structure
        inputs.pw.parent_folder = self.ctx.remote
        inputs.clean_workdir = self.inputs.clean_workdir

        self.ctx.inputs = inputs

    def run_bands(self):
        """Run the bands calculation."""
        running = self.submit(PwBaseWorkChain, **self.ctx.inputs)

        self.report('launching PwBaseWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """Verify that the PwBaseWorkChain finished successfully."""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report('PwBaseWorkChain failed with exit status {}'.format(
                workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.ctx.bands = workchain.outputs.output_band

    def analyze_bands(self):
        """Determine next set of origin point for cross search."""
        result = analyze_kpt_cross(self.ctx.bands, self.ctx.gap_thr)

        self.ctx.current_kpt.append(result)

        self.ctx.skip_kpt = result.get_array('skips')

    def results(self):
        """Output the workchain results."""
        res = finilize_cross_results(self.ctx.current_kpt[-1],
                                     self.ctx.gap_thr)

        self.out('crossings', res)

        self.report('FINISHED')
