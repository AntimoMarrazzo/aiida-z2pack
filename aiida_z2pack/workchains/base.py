"""`Z2packBaseWorkChain` workchains definition."""
from __future__ import absolute_import
from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.engine import ToContext, while_, if_, BaseRestartWorkChain, process_handler, ProcessHandlerReport
from aiida.common.exceptions import InputValidationError

from six.moves import range

PwCalculation = CalculationFactory('quantumespresso.pw')
Z2packCalculation = CalculationFactory('z2pack.z2pack')
PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')


class Z2packBaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a basic z2pack calculation, starting from the `scf` calculation."""

    _process_class = Z2packCalculation

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)

        # SCF INPUTS ###########################################################
        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.code'),
            namespace_options={
                'required':False, 'populate_defaults':False,
                'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'
                }
            )
        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The inputs structure.'
            )
        spec.input(
            'pw_code', valid_type=orm.AbstractCode,
            help='The code for pw calculations.'
            )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )
        spec.input(
            'parent_folder', valid_type=orm.RemoteData,
            required=False,
            help=(
                'Output of a previous scf calculation to start a new z2pack calclulation from. '
                'If specified, will not run the scf calculation and start straight from z2pack.'
                )
            )

        #Z2pack inputs ###########################################################
        spec.input(
            'min_neighbour_distance_scale_factor', valid_type=orm.Float,
            default=lambda: orm.Float(10.0),
            help='Scale factor for min_neighbour_distance to be used between restarts when convergence is not achieved.'
            )
        spec.input(
            'min_neighbour_distance_threshold_minimum', valid_type=orm.Float,
            default=lambda: orm.Float(1E-4),
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
            if_(cls.should_do_scf)(
                cls.run_scf,
                cls.inspect_scf,
                ),
            cls.setup_z2pack,
            while_(cls.should_run_process)(
                cls.prepare_process,
                cls.run_process,
                cls.inspect_process
                ),
            cls.results
            )

        spec.expose_outputs(Z2packCalculation)
        spec.expose_outputs(PwBaseWorkChain, namespace='scf',
            namespace_options={
                'required':False,
                })
        spec.output(
            'last_z2pack_remote_folder', valid_type=orm.RemoteData,
            required=False, help=''
            )
        spec.output(
            'wannier90_parameters', valid_type=orm.Dict,
            required=False,
            help='Auto-setted w90parameters.'
            )
        spec.exit_code(101, 'ERROR_UNRECOVERABLE_FAILURE', message='Can\'t recover. Aborting!')
        spec.exit_code(111, 'ERROR_SUB_PROCESS_FAILED_STARTING_SCF',
            message='the starting scf PwBaseWorkChain sub process failed')
        spec.exit_code(201, 'ERROR_NOT_CONVERGED',
            message='Calculation finished, but convergence not achieved.')
        spec.exit_code(211, 'ERROR_POS_TOL_CONVERGENCE_FAILED',
            message='WCCs position is not stable when increasing k-points on a line.')
        spec.exit_code(221, 'ERROR_MOVE_TOL_CONVERGENCE_FAILED',
            message='Position of largest gap between WCCs varies too much between neighboring lines.')
        spec.exit_code(222, 'ERROR_GAP_TOL_CONVERGENCE_FAILED',
            message='The WCC gap between neighboring lines are too close.')
        spec.exit_code(223, 'ERROR_MOVE_GAP_TOL_CONVERGENCE_FAILED',
            message='Both gap_tol and move_tol convergence failed.')
        spec.exit_code(231, 'ERROR_FAILED_SAVEFILE_TWICE',
            message='The calculation failed to produce the savefile for a restart twice.')
        # yapf: enable

    def setup(self):
        """Perform intial setup of the workchain."""
        super().setup()

        try:
            self.ctx.current_MND = self.inputs.z2pack.z2pack_settings.get_attribute(
                'min_neighbour_dist')
        except:
            self.ctx.current_MND = Z2packCalculation._DEFAULT_MIN_NEIGHBOUR_DISTANCE

        self.ctx.restart = False

        self.ctx.MND_threshold = self.inputs.min_neighbour_distance_threshold_minimum.value
        self.ctx.MND_scale_factor = self.inputs.min_neighbour_distance_scale_factor.value

    def should_do_scf(self):
        """Check if the `scf` calculation should be performed or the parent folder should be taken from the inputs."""
        if 'parent_folder' in self.inputs:
            if 'scf' in self.inputs:
                self.report(
                    'WARNING: both `scf` and `parent_folder` input ports specfied. `scf` will be ignored'
                )

            pc = self.inputs.parent_folder.creator.process_class
            if issubclass(pc, PwCalculation):
                pass
            elif issubclass(pc, Z2packCalculation):
                self.ctx.restart = True
            else:
                raise InputValidationError(
                    '`parent_folder` must be subclas of either `PwCalculation` or `Z2packCalculation`.'
                )

            self.ctx.parent_folder = self.inputs.parent_folder

            return False

        return True

    def run_scf(self):
        """Run the `scf` calculation."""
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.structure = self.inputs.structure
        inputs.pw.code = self.inputs.pw_code

        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> for starting scf'.format(
            running.pk))

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Inspect the result of the starting scf `PwBaseWorkChain`."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                'Starting scf PwBaseWorkChain failed with exit status {}'.
                format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_STARTING_SCF

        self.ctx.parent_folder = self.ctx.workchain_scf.outputs.remote_folder

        self.out_many(
            self.exposed_outputs(workchain, PwBaseWorkChain, namespace='scf'))

    def should_run_process(self):
        """Return whether a new calculation should be run.

        Same behaviour as the BaseRestartWorkChain from the qe plugin.
        Also stop the iterations if the `min_neighbour_distance` convergence parameter drops below the set
        threshold level.
        """
        return super().should_run_process(
        ) and self.ctx.current_MND >= self.ctx.MND_threshold

    def setup_z2pack(self):
        """Prepare the inputs for the z2pack CalcJob."""
        inputs = AttributeDict(self.exposed_inputs(Z2packCalculation,
                                                   'z2pack'))
        inputs.pw_code = self.inputs.pw_code
        inputs.parent_folder = self.ctx.parent_folder
        inputs.z2pack_settings = inputs.z2pack_settings.get_dict()

        if not self.ctx.restart:
            if 'wannier90_parameters' not in inputs:
                inputs.wannier90_parameters = self._autoset_wannier90_paremters(
                )
            else:
                params = inputs.wannier90_parameters.get_dict()
                if any(var not in params
                       for var in ['num_wann', 'num_bands', 'exclude_bands']):
                    inputs.wannier90_parameters = self._autoset_wannier90_paremters(
                    )

        self.ctx.inputs = inputs

    def prepare_process(self):
        """Prepare the inputs for a loop restart calculation."""
        self.ctx.inputs.z2pack_settings[
            'min_neighbour_dist'] = self.ctx.current_MND
        if self.ctx.iteration > 0:
            previous = self.ctx.children[-1]
            try:
                remote = previous.outputs.remote_folder
            except:
                remote = self.ctx.parent_folder
            self.ctx.inputs.parent_folder = remote

    def inspect_process(self):
        """Check the outputs of the calculation."""
        self.ctx.inputs.z2pack_settings['restart_mode'] = True

        res = super().inspect_process()

        if res is None or res.status != 0:
            node = self.ctx.children[self.ctx.iteration - 1]
            self.out('last_z2pack_remote_folder', node.outputs.remote_folder)

        return res

    # def on_terminated(self):
    #     """Store last available z2pack remote data."""
    #     node = self.ctx.children[self.ctx.iteration - 1]

    #     for node in self.ctx.children[::-1]:
    #         if 'remote_folder' in node.outputs:
    #             self.out('last_z2pack_remote_folder', node.outputs.remote_folder)
    #             break

    #     super().on_terminated()

    # def on_failed(self):
    #     """Store last available z2pack remote data."""
    #     node = self.ctx.children[self.ctx.iteration - 1]

    #     for node in self.ctx.children[::-1]:
    #         if 'remote_folder' in node.outputs:
    #             self.out('last_z2pack_remote_folder', node.outputs.remote_folder)
    #             break

    #     super().on_failed()

    def _autoset_wannier90_paremters(self):
        """If not given, set the number of wannier functions and band as all the bands up to the valence one. Ignore the rest."""
        self.report(
            'Required w90 parameters are missing. Guessing them from the output of the scf calculation.'
        )
        pw_params = self.ctx.parent_folder.creator.outputs.output_parameters.get_dict(
        )
        n_bnd = pw_params['number_of_bands']
        n_el = pw_params['number_of_electrons']
        spin = pw_params['spin_orbit_calculation']

        if not spin:
            n_el /= 2
        n_el = int(n_el)

        w90_params = {}
        w90_params['num_wann'] = n_el
        w90_params['num_bands'] = n_el
        w90_params['exclude_bands'] = [*list(range(n_el + 1, n_bnd + 1))]

        res = orm.Dict(dict=w90_params)

        self.out('wannier90_parameters', res)

        return res

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        arguments = [
            calculation.process_label, calculation.pk, calculation.exit_status,
            calculation.exit_message
        ]
        if calculation.exit_status:
            self.report(
                '{}<{}> failed with exit status {}: {}'.format(*arguments))
        else:
            self.report('{}<{}> sanity check failed'.format(*arguments[:2]))

        self.report('Action taken: {}'.format(action))

    @process_handler(priority=600)
    def handle_unrecoverable_failure(self, calculation):
        """Handle calculations with an exit status below 400 which are unrecoverable, so abort the work chain."""
        if calculation.is_failed and calculation.exit_status < 400:
            self.report_error_handled(calculation,
                                      'unrecoverable error, aborting...')
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

    @process_handler(
        priority=590,
        exit_codes=[
            Z2packCalculation.exit_codes.ERROR_MISSING_RESULTS_FILE,
        ])
    def handle_out_of_walltime(self, calculation):
        """Handle calculation that did not finish because the walltime was exceeded."""
        self.report_error_handled(
            calculation,
            'The calculation died because of exceeded walltime. Restarting...')
        return ProcessHandlerReport(True)

    @process_handler(priority=580,
                     exit_codes=[
                         Z2packCalculation.exit_codes.ERROR_MISSING_SAVE_FILE,
                     ])
    def handle_no_save_file(self, calculation):
        """Try to relaunch calculation that did not produce a save file once. Exit if it fails twice."""
        if 'restart_no_save' not in self.ctx:
            self.ctx.restart_no_save = True
            self.ctx.inputs.z2pack_settings['restart_mode'] = False
            self.report_error_handled(
                calculation,
                'The calculation died before the savefile for a restart was produced, trying to restart it from scratch.'
            )
            return ProcessHandlerReport(True)
        else:
            self.report_error_handled(
                calculation,
                self.exit_codes.ERROR_FAILED_SAVEFILE_TWICE.message +
                ' Aborting...')
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_FAILED_SAVEFILE_TWICE)

    @process_handler(priority=570)
    def handle_failed(self, calculation):
        """Handle calculation that did not produce an output."""
        try:
            calculation.outputs.output_parameters
        except:
            return ProcessHandlerReport(
                True, self.exit_codes.ERROR_UNRECOVERABLE_FAILURE)

    @process_handler(priority=560)
    def handle_not_converged(self, calculation):
        """Lower threshold and restart calculation that finished ok, but did not reach convergence because of min threshold parameters."""
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
                return ProcessHandlerReport(
                    True, self.exit_codes.ERROR_POS_TOL_CONVERGENCE_FAILED)

            # if len(param['GapCheck']['FAILED']):
            #     # gap_tol  = settings.get('gap_tol', Z2packCalculation._DEFAULT_GAP_TOLERANCE)
            #     # MND = settings.get('min_neighbour_dist', Z2packCalculation._DEFAULT_MIN_NEIGHBOUR_DISTANCE)
            #     # self.report_error_handled(
            #     #     'Convergence of gap position between lines failed with `gap_tol={}` and `min_neighbour_dist={}`'.format(
            #     #     gap_tol, MND
            #     #     ))
            #     return ErrorHandlerReport(True, True, self.exit_codes.ERROR_GAP_TOL_CONVERGENCE_FAILED)

            gcheck = len(report['GapCheck']['FAILED'])
            mcheck = len(report['MoveCheck']['FAILED'])
            if mcheck or gcheck:
                self.ctx.current_MND /= self.ctx.MND_scale_factor
                if self.ctx.current_MND < self.ctx.MND_threshold:
                    self.report_error_handled(
                        calculation,
                        'Convergence between lines failed. `min_neighbour_dist` already at minimum value.'
                    )
                    if gcheck:
                        if mcheck:
                            error = self.exit_codes.ERROR_MOVE_GAP_TOL_CONVERGENCE_FAILED
                        else:
                            error = self.exit_codes.ERROR_GAP_TOL_CONVERGENCE_FAILED
                    else:
                        error = self.exit_codes.ERROR_MOVE_TOL_CONVERGENCE_FAILED
                    return ProcessHandlerReport(True, error)

                self.report_error_handled(
                    calculation,
                    'Convergence between lines failed. Reducing `min_neighbour_dist` and rerunning calculation.'
                )
                return ProcessHandlerReport(True)

        # self.ctx.is_converged = True
        # return ProcessHandlerReport(True)
