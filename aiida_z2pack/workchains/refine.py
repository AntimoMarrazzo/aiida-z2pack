"""`RefineCrossingsPosition` workchain definition."""
from __future__ import absolute_import
import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain, while_, ToContext

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

        spec.expose_inputs(
            PwBaseWorkChain, namespace='bands',
            exclude=('pw.code', ),
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
            while_(cls.do_loop)(cls.setup_kpt, cls.run_bands,
                                cls.inspect_bands, cls.analyze_bands),
            cls.results)

        # OUTPUTS ############################################################################
        spec.output(
            'crossings',
            valid_type=orm.ArrayData,
            required=True,
            help='The array containing a list of bands crossing found as rows.'
        )

        # ERRORS ############################################################################
        spec.exit_code(332,
                       'ERROR_SUB_PROCESS_FAILED_BANDS',
                       message='the bands PwBaseWorkChain sub process failed')
        # yapf: enable

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.counter = 0

        self.ctx.structure = self.inputs.bands.pw.structure

        app = self.inputs.crossings.get_array('crossings')
        ncross = len(app)
        self.ctx.ncross = ncross

        self.ctx.step = self.inputs.step_size
        self.ctx.gap_thr = self.inputs.gap_threshold.value

        self.ctx.current_kpt = [self.inputs.crossings]
        self.ctx.skip_kpt = [0] * ncross

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
        self.ctx.kpt_data = generate_kpt_cross(self.ctx.structure, curr_kpt,
                                               self.ctx.step)

        self.ctx.counter += 1

    def run_bands(self):
        """Run the bands calculation."""
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='bands'))
        inputs.pw.code = self.inputs.code
        inputs.kpoints = self.ctx.kpt_data

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
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

        self.ctx.skip_kpt = self.ctx.current_kpt.get_array('skips')

    def results(self):
        """Output the workchain results."""
        res = finilize_cross_results(self.ctx.current_kpt[-1],
                                     self.ctx.gap_thr)

        self.out('crossings', res)

        self.report('FINISHED')
