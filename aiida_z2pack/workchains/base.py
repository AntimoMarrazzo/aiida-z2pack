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
            cls.setup_z2pack,
            while_(cls.should_run_calculation)(
                cls.prepare_calculation,
                cls.run_calculation,
                cls.inspect_calculation
                ),
            cls.results
            )

        spec.expose_outputs(Z2packCalculation)
        spec.exit_code(101, 'ERROR_UNRECOVERABLE_FAILURE', message='Can\'t recover. Aborting!')
        spec.exit_code(111, 'ERROR_SUB_PROCESS_FAILED_STARTING_SCF',
            message='the starting scf PwBaseWorkChain sub process failed')
        spec.exit_code(201, 'ERROR_NOT_CONVERGED',
            message='Calculation finished, but convergence not achieved.')
        spec.exit_code(211, 'ERROR_POS_TOL_CONVERGENCE_FAILED',
            message='WCCs position is not stable when increasing k-points on a line.')
        spec.exit_code(221, 'ERROR_GAP_TOL_CONVERGENCE_FAILED',
            message='Position of largest gap between WCCs varies too much between neighboring lines.')

    def setup(self):
        super().setup()

        try:
            self.ctx.current_MND = self.inputs.z2pack.z2pack_settings.get('min_neighbour_dist')
        except:
            self.ctx.current_MND = Z2packCalculation._DEFAULT_MIN_NEIGHBOUR_DISTANCE

        self.ctx.MND_threshold    = self.inputs.min_neighbour_distance_threshold_minimum.value
        self.ctx.MND_scale_factor = self.inputs.min_neighbour_distance_scale_factor.value

    def run_scf(self):
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.structure = self.inputs.structure

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> for starting scf'.format(running.pk))

        return ToContext(workchain_scf=running)
        # from aiida.orm.utils import load_node
        # self.ctx.workchain_scf = load_node(5886)
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
        Same behaviour as the BaseRestartWorkChain from the qe plugin. 
        Also stop the iterations if the `min_neighbour_distance` convergence parameter drops below the set
        threshold level.
        """
        return super().should_run_calculation() and self.ctx.current_MND >= self.ctx.MND_threshold

    def setup_z2pack(self):
        inputs = AttributeDict(self.exposed_inputs(Z2packCalculation, 'z2pack'))
        inputs.pw_code = self.inputs.scf.pw.code
        inputs.parent_folder = self.ctx.workchain_scf.outputs.remote_folder
        inputs.z2pack_settings = inputs.z2pack_settings.get_dict()

        if not 'wannier90_parameters' in inputs:
            inputs.wannier90_parameters = self._autoset_wannier90_paremters()
        else:
            params = inputs.wannier90_parameters
            if any(not var in params for var in ['num_wann', 'num_bands', 'exclude_bands']):
                inputs.wannier90_parameters = self._autoset_wannier90_paremters()

        self.ctx.inputs = inputs

    def prepare_calculation(self):
        self.ctx.inputs.z2pack_settings['min_neighbour_dist'] = self.ctx.current_MND
        if self.ctx.iteration > 0:
            previous = self.ctx.calculations[-1]
            remote   = previous.outputs.remote_folder 
            self.ctx.inputs.parent_folder = remote

    def results(self):
        """Attach the output parameters of the last workchain to the outputs."""

        final_calc = self.ctx.calculations[-1]
        self.out('output_parameters', final_calc.outputs.output_parameters)

    def _autoset_wannier90_paremters(self):
        self.report("Required w90 parameters are missing. Guessing them from the output of the scf calculation.")
        pw_params = self.ctx.workchain_scf.outputs.output_parameters.get_dict()
        n_bnd     = pw_params['number_of_bands']
        n_el      = pw_params['number_of_electrons']
        spin      = pw_params['spin_orbit_calculation']

        if not spin:
            n_el /= 2
        n_el = int(n_el)

        w90_params = {}
        w90_params['num_wann'] = n_el
        w90_params['num_bands'] = n_el
        w90_params['exclude_bands'] = [*range(n_el+1, n_bnd+1)]

        return w90_params

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

    self.ctx.is_converged = False
    if not param['Tests_passed']:
        report = param['convergence_report']
        # self.report_error_handled('calculation<{}> did not achieve convergence.')
        if len(report['PosCheck']['FAILED']):
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

        if len(report['MoveCheck']['FAILED']) or len(report['GapCheck']['FAILED']):
            self.ctx.current_MND /= self.ctx.MND_scale_factor
            self.report_error_handled(
                calculation,
                'Convergence between lines failed. Reducing `min_neighbour_dist` and rerunning calculation.'
                )
            return ErrorHandlerReport(True, True)

    self.ctx.is_converged = True
    return ErrorHandlerReport(True, True)



