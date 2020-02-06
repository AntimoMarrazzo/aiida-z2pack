from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.engine import WorkChain, ToContext, if_, else_, while_

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

Z2packCalculation = CalculationFactory('z2pack.z2pack')

PwBaseWorkChain   = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain  = WorkflowFactory('quantumespresso.pw.relax')

class Z2pack3DChernWorkChain(WorkChain):
    """Workchain to compute topological invariants (Z2 or Chern number) using z2pack."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.expose_inputs(
            PwRelaxWorkChain, namespace='relax',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` for the RELAX calculation.'
                }
            )
        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'
                }
            )
        spec.expose_inputs(
            PwBaseWorkChain, namespace='nscf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` for the BANDS calculation.'
                }
            )

        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The inputs structure.'
            )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ).else_(
                cls.run_scf,
                cls.inspect_scf
                ),
            while_(cls.should_find_zero_gap)(
                cls.setup_grid,
                cls.run_nscf,
                cls.analyze_bands
                ),
            cls.run_z2pack,
            cls.inspect_results
            )

        # OUTPUTS ############################################################################
        # spec.output(
        #     'output_parameters', valid_type=orm.Dict,
        #     help='Dict resulting from a z2pack calculation.'
        #     )
        spec.expose_outputs(Z2packCalculation)

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure
        
    def should_do_relax(self):
        """If the 'relax' input namespace was specified, we relax the input structure."""
        return 'relax' in self.inputs

    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation."""
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        inputs.structure = self.ctx.current_structure

        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report('launching PwRelaxWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report('PwRelaxWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.ctx.current_structure = workchain.outputs.output_structure

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(running.pk, 'scf'))

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report('scf PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder

    def should_find_zero_gap(self):
        pass

    def setup_grid(self):
        pass

    def run_nscf(self):
        pass

    def analyze_bands(self):
        pass

    def run_z2pack(self):
        pass

    def inspect_results(self):
        pass

