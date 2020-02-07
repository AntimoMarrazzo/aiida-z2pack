from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.engine import ToContext, while_

# from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.common.workchain.utils import register_error_handler, ErrorHandlerReport
from aiida_quantumespresso.common.workchain.base.restart import BaseRestartWorkChain

PwCalculation     = CalculationFactory('quantumespresso.pw')
Z2packCalculation = CalculationFactory('z2pack.z2pack')
PwBaseWorkChain   = WorkflowFactory('quantumespresso.pw.base')

# PwBaseWorkChain   = WorkflowFactory('quantumespresso.pw.base')
# PwRelaxWorkChain  = WorkflowFactory('quantumespresso.pw.relax')


class Z2packBaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a basic z2pack calculation, starting from the scf."""

    _calculation_class = Z2packCalculation

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # SCF INPUTS ###########################################################
        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={
                'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'
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

        #Z2pack inputs ###########################################################
        spec.input(
            'min_neighbour_distance_scale_factor', valid_type=orm.Float,
            default=orm.Float(10.0),
            help='Scale factor for min_neighbour_distance to be used between restarts when convergence is not achieved.'
            )
        spec.input(
            'min_neighbour_distance_threshold_minimum', valid_type=orm.Float,
            default=orm.Float(1E-4),
            help='Stop the restart iterations when `min_neighbour_distance` becomes smaller than this threshold.'
            )
        spec.expose_inputs(
            Z2packCalculation, namespace='z2pack',
            exclude=('parent_folder', 'pw_code'),
            namespace_options={
                'help': 'Inputs for the `Z2packCalculation` for the SCF calculation.'
                }
            )

        spec.outline(
            cls.setup,
            cls.run_scf,
            cls.inspect_scf,
            while_(cls.should_run_calculation)(
                cls.prepare_calculation,
                cls.run_calculation,
                cls.inspect_calculation
                ),
            cls.results
            )

        spec.expose_outputs(Z2packCalculation)

        spec.exit_code(100, 'ERROR_SUB_PROCESS_FAILED_STARTING_SCF',
            message='the starting scf PwBaseWorkChain sub process failed')
        spec.exit_code(200, 'ERROR_NOT_CONVERGED',
            message='Calculation finished, but convergence not achieved.')
        spec.exit_code(210, 'ERROR_POS_TOL_CONVERGENCE_FAILED',
            message='WCCs position is not stable when increasing k-points on a line.')
        spec.exit_code(220, 'ERROR_GAP_TOL_CONVERGENCE_FAILED',
            message='Position of largest gap between WCCs varies too much between neighboring lines.')

    def setup(self):
        super().setup()

        try:
            self.ctx.current_MND = self.inputs.z2pack.z2pack_settings.get('min_neighbour_dist')
        except:
            self.ctx.current_MND = Z2packCalculation._DEFAULT_MIN_NEIGHBOUR_DISTANCE

        self.ctx.MND_threshold    = self.inputs.min_neighbour_distance_threshold_minimum
        self.ctx.MND_scale_factor = self.inputs.min_neighbour_distance_scale_factor

    def run_scf(self):
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.structure = self.inputs.structure

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> for starting scf'.format(running.pk))

        return ToContext(workchain_scf=running)
        # self.ctx.workchain_scf = AttributeDict()
        # self.ctx.workchain_scf.is_finished_ok = True

    def inspect_scf(self):
        """Inspect the result of the starting scf `PwBaseWorkChain`."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report('Starting scf PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_STARTING_SCF

    def should_run_calculation(self):
        """Return whether a new calculation should be run.

        This is the case as long as the last calculation has not finished successfully and the maximum number of
        restarts has not yet been exceeded.
        Also stop the iterations if the `min_neighbour_distance` convergence parameter drops below the set
        threshold level.
        """
        # self.report("SHOULD DO?")

        # self.report('{}, {}, {}'.format(self.ctx.is_finished, self.ctx.iteration, self.inputs.max_iterations.value))
        # self.report('{}, {}'.format(self.ctx.current_MND, self.ctx.MND_threshold))
        return super().should_run_calculation() and self.ctx.current_MND >= self.ctx.MND_threshold

    def prepare_calculation(self):
        self.inputs.z2pack.pw_code = self.inputs.scf.pw.code
        self.inputs.z2pack.parent_folder = self.ctx.workchain_scf.outputs.remote_data
        settings = self.inputs.z2pack.z2pack_settings.get_dict()
        settings['min_neighbour_dist'] = self.ctx.current_MND

        self.inputs.z2pack.z2pack_settings = orm.Dict(dict=settings)


    def results(self):
        """Attach the output parameters and structure of the last workchain to the outputs."""
        # if self.ctx.is_converged and self.ctx.iteration <= self.inputs.max_meta_convergence_iterations.value:
        #     self.report('workchain completed after {} iterations'.format(self.ctx.iteration))
        # else:
        #     self.report('maximum number of meta convergence iterations exceeded')

        final_calc = self.ctx.calculations[-1]
        self.out('output_parameters', final_calc.outputs.output_parameters)

        # # Get the latest workchain, which is either the workchain_scf if it ran or otherwise the last regular workchain
        # try:
        #     workchain = self.ctx.workchain_scf
        #     structure = workchain.inputs.pw__structure
        # except AttributeError:
        #     workchain = self.ctx.workchains[-1]
        #     structure = workchain.outputs.output_structure

        # self.out_many(self.exposed_outputs(workchain, PwBaseWorkChain))
        # self.out('output_structure', structure)


    def _handle_calculation_sanity_checks(self, calculation):
        """Check if the calculation fnished but did not reach convergence."""
        report_convergence = self._handle_not_converged(calculation)
        if not self.ctx.is_converged:
            return report_convergence


    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [calculation.process_label, calculation.pk, calculation.exit_status, calculation.exit_message]
        self.report('{}<{}> failed with exit status {}: {}'.format(*arguments))
        self.report('Action taken: {}'.format(action))

@register_error_handler(Z2packBaseWorkChain, 600)
def _handle_unrecoverable_failure(self, calculation):
    """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
    if calculation.exit_status < 200:
        self.report_error_handled(calculation, 'unrecoverable error, aborting...')
        return ErrorHandlerReport(True, True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

@register_error_handler(Z2packBaseWorkChain, 580)
def _handle_not_converged(self, calculation):
    # try:
    #     settings = calculation.inputs.z2pack.z2pack_settings
    # except:
    #     settings = {}
    param = calculation.outputs.output_parameters

    if not param['Tests_passed']:
        # self.report_error_handled('calculation<{}> did not achieve convergence.')
        if len(param['PosCheck']['FAILED']):
            # pos_tol  = settings.get('pos_tol', Z2packCalculation._DEFAULT_POS_TOLERANCE)
            # iterator = settings.get('iterator', Z2packCalculation._DEFAULT_ITERATOR)
            # self.report_error_handled('Convergence across line failed with `pos_tol={}` and `iterator={}`'.format(
            #     pos_tol, iterator
            #     ))
            return ErrorHandlerReport(True, True, self.exit_codes.ERROR_POS_TOL_CONVERGENCE_FAILED)

        # if len(param['GapCheck']['FAILED']):
        #     # gap_tol  = settings.get('gap_tol', Z2packCalculation._DEFAULT_GAP_TOLERANCE)
        #     # MND = settings.get('min_neighbour_dist', Z2packCalculation._DEFAULT_MIN_NEIGHBOUR_DISTANCE)
        #     # self.report_error_handled(
        #     #     'Convergence of gap position between lines failed with `gap_tol={}` and `min_neighbour_dist={}`'.format(
        #     #     gap_tol, MND
        #     #     ))
        #     return ErrorHandlerReport(True, True, self.exit_codes.ERROR_GAP_TOL_CONVERGENCE_FAILED)

        if len(param['MoveCheck']['FAILED']) or len(param['GapCheck']['FAILED']):
            self.ctx.current_MND /= self.ctx.MND_scale_factor
            self.report_error_handled(
                calculation,
                'Convergence between lines failed. Reducing `min_neighbour_dist` and rerunning calculation.'
                )
            return ErrorHandlerReport(True, True)

    self.ctx.is_converged = True
    return ErrorHandlerReport(True, True)
# @register_error_handler(Z2packBaseWorkChain, 560)
# def _handle_relax_recoverable_ionic_convergence_error(self, calculation):
#     """Handle various exit codes for recoverable `vc-relax` or `relax` calculations with failed ionic convergence.

#     These exit codes signify that the ionic convergence thresholds were not met, but the output structure is usable, so
#     the solution is to simply restart from scratch but from the output structure.
#     """
#     exit_code_labels = [
#         'ERROR_IONIC_CONVERGENCE_NOT_REACHED',
#         'ERROR_IONIC_CYCLE_EXCEEDED_NSTEP',
#         'ERROR_IONIC_CYCLE_BFGS_HISTORY_FAILURE',
#         'ERROR_IONIC_CYCLE_BFGS_HISTORY_AND_FINAL_SCF_FAILURE',
#     ]

#     if calculation.exit_status in PwCalculation.get_exit_statuses(exit_code_labels):
#         self.ctx.restart_calc = None
#         self.ctx.inputs.structure = calculation.outputs.output_structure
#         action = 'no ionic convergence but clean shutdown: restarting from scratch but using output structure.'
#         self.report_error_handled(calculation, action)
#         return ErrorHandlerReport(True, True)

# from aiida.engine import calcfunction
# @calcfunction
# def update_min_neighbour_dist(settings, value):
#     settings = settings.get_dict()
#     settings['min_neighbour_dist'] = value.value

#     return orm.Dict(dict=settings)


