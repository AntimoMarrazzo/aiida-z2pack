from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.utils.mapping import prepare_process_inputs

from .functions import (
    generate_cubic_grid, get_kpoint_grid_dimensionality,
    get_crossing_and_lowgap_points, merge_crossing_results,
    merge_chern_results
    )

# Z2packCalculation   = CalculationFactory('z2pack.z2pack')

PwBaseWorkChain     = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain    = WorkflowFactory('quantumespresso.pw.relax')
Z2packBaseWorkChain = WorkflowFactory('z2pack.base')

class FindCrossingsWorkChain(WorkChain):
    """Workchain to find bands crossing in the Brillouin Zone using
    a series of quantum espresso bands calculations."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input(
            'code', valid_type=orm.Code,
            help='The PWscf code.'
            )
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
            PwRelaxWorkChain, namespace='relax',
            exclude=('clean_workdir', 'structure', 'base.pw.code', 'base.pw.pseudos'),
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for the RELAX calculation.'
                }
            )
        spec.expose_inputs(
            PwBaseWorkChain, namespace='scf',
            exclude=('clean_workdir', 'pw.structure', 'pw.code', 'pw.pseudos'),
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'
                }
            )
        spec.expose_inputs(
            PwBaseWorkChain, namespace='bands',
            exclude=('clean_workdir', 'pw.structure', 'pw.code', 'pw.pseudos', 'kpoints'),
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for the BANDS calculation.'
                }
            )

        spec.input(
            'min_kpoints_distance', valid_type=orm.Float,
            default=orm.Float(5.E-4),
            help='Stop iterations when `kpoints_distance`  drop below this value.'
            )
        spec.input(
            'starting_kpoints_distance', valid_type=orm.Float,
            default=orm.Float(0.2),
            help='Strating distance between kpoints.'
            )
        spec.input(
            'scale_kpoints_distance', valid_type=orm.Float,
            default=orm.Float(5.0),
            help='Across iterations divide `kpoints_distance` by this scaling factor.'
            )
        spec.input(
            'starting_kpoints', valid_type=orm.KpointsData,
            required=False,
            help='Starting mesh of kpoints'
            )
        spec.input(
            'gap_threshold', valid_type=orm.Float,
            default=orm.Float(0.0025),
            help='kpoints ith gap < `gap_threshold` are considered possible crossings.'
            )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.inspect_relax,
                ),
            if_(cls.should_do_scf)(
                cls.run_scf,
                cls.inspect_scf,
            ).else_(
                cls.set_remote_scf
                ),
            cls.setup_bands_loop,
            if_(cls.should_do_first_bands)(
                cls.first_bands_step,
                cls.inspect_bands,
            ).else_(
                cls.start_from_scf
                ),
            cls.analyze_bands,
            while_(cls.should_find_zero_gap)(
                cls.setup_grid,
                cls.run_bands,
                cls.inspect_bands,
                cls.analyze_bands,
                cls.stepper
                ),
            cls.results
            )

        # OUTPUTS ############################################################################
        spec.output('crossings', valid_type=orm.ArrayData,
            required=True,
            help='The array containing a list of bands crossing found as rows.'
            )
        spec.output('scf_remote_folder', valid_type=orm.RemoteData,
            required=False,
            help='The remote folder produced by the scf calculation.'
            )
        spec.output('output_structure', valid_type=orm.StructureData,
            required=False,
            help='The structure produced by the relax calculation.'
            )

        # ERRORS ############################################################################
        spec.exit_code(112, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='the relax PwRelaxWorkChain sub process failed')
        spec.exit_code(122, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='the scf PwBaseWorkChain sub process failed')
        spec.exit_code(132, 'ERROR_SUB_PROCESS_FAILED_BANDS',
            message='the bands PwBaseWorkChain sub process failed')
        spec.exit_code(142, 'ERROR_CANT_PINPOINT_LOWGAP_ZONE',
            message='After two iterations, no points with low_gap found. Aborting calculation!')
        spec.exit_code(152, 'ERROR_MINIMUM_DISTANCE_RAECHED',
            message='The minimum distance was reached without finding any crossings.')
        spec.exit_code(162, 'ERROR_TOO_MANY_ARRAYS',
            message='An ArrayData node contains more arrays than expected.')
        spec.exit_code(172, 'ERROR_MEMORY_TOO_MANY_KPOINTS',
            message='The generation of the kpoints failed because the mesh size was too big.')

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.pseudos           = self.inputs.pseudos
        self.ctx.current_structure = self.inputs.structure
        
    def should_do_relax(self):
        """If the 'relax' input namespace was specified, we relax the input structure."""
        return 'relax' in self.inputs

    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation."""
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        inputs.structure = self.ctx.current_structure
        inputs.base.pw.pseudos = self.inputs.pseudos
        inputs.base.pw.code    = self.inputs.code
        inputs.clean_workdir   = self.inputs.clean_workdir

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

        self.out('output_structure', self.ctx.current_structure)

    def should_do_scf(self):
        return not 'parent_folder' in self.inputs

    def set_remote_scf(self):
        self.ctx.scf_folder    = self.inputs.parent_folder
        self.ctx.workchain_scf = self.ctx.scf_folder.creator

        self.out('scf_remote_folder', self.ctx.scf_folder)

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.pw.pseudos    = self.inputs.pseudos
        inputs.pw.code       = self.inputs.code
        inputs.pw.structure  = self.ctx.current_structure
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

        self.ctx.scf_folder = workchain.outputs.remote_folder

        self.out('scf_remote_folder', self.ctx.scf_folder)

    def setup_bands_loop(self):
        self.ctx.iteration = 0
        # self.ctx.max_iteration = self.inputs.max_iter
        if 'bands' in self.inputs:
            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='bands'))
        else:
            inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))

        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters['CONTROL']['calculation'] = 'bands'
        
        inputs.pw.parent_folder = self.ctx.scf_folder
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.pw.structure  = self.ctx.current_structure
        inputs.pw.pseudos    = self.inputs.pseudos
        inputs.pw.code       = self.inputs.code

        self.ctx.inputs = inputs

        self.ctx.current_kpoints_distance  = self.inputs.starting_kpoints_distance.value
        self.ctx.min_kpoints_distance      = self.inputs.min_kpoints_distance.value
        self.ctx.scale_kpoints_distance    = self.inputs.scale_kpoints_distance.value

        self.ctx.gap_threshold = self.inputs.gap_threshold.value

        if 'starting_kpoints' in self.inputs:
            self.ctx.dim = get_kpoint_grid_dimensionality(self.inputs.starting_kpoints)
        else:
            self.ctx.dim = orm.Int(3)

        self.ctx.found_crossings = []
        self.ctx.do_loop = True
        self.ctx.flag = False

        self.report('Starting loop to find bands crossings.')

    def should_do_first_bands(self):
        """Determine if the first bands should be run using starting_kpoints"""
        return 'starting_kpoints' in self.inputs

    def start_from_scf(self):
        self.report('No `starting_kpoints` provided. Using BandsData from `scf` calculation...')
        self.ctx.bands = self.ctx.workchain_scf.outputs.output_band

    def first_bands_step(self):
        self.ctx.iteration += 1
        inputs = self.ctx.inputs
        inputs.kpoints = self.inputs.starting_kpoints

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode, iteration {}'.format(
            running.pk, 'bands', self.ctx.iteration
            ))

        return ToContext(workchain_bands=append_(running))

    def should_find_zero_gap(self):
        """Limit iterations over kpoints meshes."""
        return  self.ctx.do_loop

    def setup_grid(self):
        distance = orm.Float(self.ctx.current_kpoints_distance)
        self.ctx.current_kpoints = generate_cubic_grid(
            self.ctx.current_structure, self.ctx.found_crossings[-1], distance, self.ctx.dim
            )

    def run_bands(self):
        self.ctx.iteration += 1
        inputs = self.ctx.inputs
        inputs.kpoints = self.ctx.current_kpoints

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode, iteration {}'.format(
            running.pk, 'bands', self.ctx.iteration
            ))

        return ToContext(workchain_bands=append_(running))
    
    def inspect_bands(self):
        """Verify that the PwBaseWorkChain for the bands run finished successfully."""
        workchain = self.ctx.workchain_bands[self.ctx.iteration - 1]

        if not workchain.is_finished_ok:
            self.report('scf PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.ctx.bands = workchain.outputs.output_band

    def analyze_bands(self):
        """Extract kpoints with gap lower than the gap threshold"""
        bands = self.ctx.bands

        self.report('Analyzing bands results for BandsData<{}>'.format(bands.pk))
        res = get_crossing_and_lowgap_points(
            bands, self.inputs.gap_threshold
            )

        pinned = res.get_array('pinned')
        found  = res.get_array('found')
        n_pinned = len(pinned)
        n_found  = len(found)
        self.report('`{}` low-gap points found.'.format(n_pinned))
        self.report('`{}` crossing points found.'.format(n_found))

        if not n_pinned:
            self.report('No low-gap points found to continue loop. iteration <{}>'.format(self.ctx.iteration))
            self.ctx.do_loop = False

        self.ctx.found_crossings.append(res)

    def stepper(self):
        """Perform the loop step operation of modifying the thresholds"""
        if self.ctx.flag:
            self.ctx.do_loop = False
            return

        if self.ctx.do_loop:
            self.ctx.current_kpoints_distance /= self.ctx.scale_kpoints_distance
            if self.ctx.current_kpoints_distance <= self.ctx.min_kpoints_distance:
                self.ctx.current_kpoints_distance = self.ctx.min_kpoints_distance
                self.ctx.flag = True

            self.report('Kpoints distance reduced to `{}`'.format(self.ctx.current_kpoints_distance))

    def results(self):
        calculation = self.ctx.workchain_bands[self.ctx.iteration - 1]

        found = merge_crossing_results(
            structure=self.ctx.current_structure,
            **{'found_{}'.format(n):array for n,array in enumerate(self.ctx.found_crossings)}
            )

        n_found = len(found.get_array('crossings'))
        if self.ctx.flag and not n_found:
            self.report(
                'WARNING: No crossing found. Reached the minimum kpoints distance {}: last ran PwBaseWorkChain<{}>'.format(
                self.ctx.min_kpoints_distance, calculation.pk
                ))
        if not self.ctx.do_loop and not n_found:
            self.report(
                'WARNING: No crossing found. Did not find any low-gap points to continue loop. iteration <{}>'.format(
                    self.ctx.iteration
                    ))

        self.out('crossings', found)

        self.report('FINISHED')


class Z2pack3DChernWorkChain(WorkChain):
    """Workchain to compute topological invariants (Z2 or Chern number) using z2pack."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input(
            'pw_code', valid_type=orm.Code,
            help='The code for pw calculations.'
            )
        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The inputs structure.'
            )
        spec.input(
            'crossings', valid_type=orm.ArrayData,
            required=False,
            help='Skip the FindCrossing and use the result of a previously run workchain.'
            )
        spec.input(
            'sphere_radius', valid_type=orm.Float,
            default=orm.Float(0.005),
            help='Radius for the sphere of kpoints.'
            )
        spec.input(
            'scf_parent_folder', valid_type=orm.RemoteData,
            required=False,
            help='The remote_folder of an scf calculation to be used by z2pack.'
            )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )

        spec.expose_inputs(
            FindCrossingsWorkChain, namespace='find',
            exclude=('clean_workdir', 'structure', 'code'),
            namespace_options={
                'required':False, 'populate_defaults':False,
                'help': 'Inputs for the `FindCrossingsWorkChain`.'
                }
            )
        spec.expose_inputs(
            Z2packBaseWorkChain, namespace='z2pack_base',
            exclude=('clean_workdir', 'structure', 'parent_folder', 'pw_code'),
            namespace_options={
                'help': 'Inputs for the `FindCrossingsWorkChain`.'
                }
            )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            cls.validate_crossings,
            if_(cls.should_do_find_crossings)(
                cls.run_find_crossings,
                cls.inspect_find_crossings,
            ).else_(
                cls.set_crossings_from_input
                ),
            cls.prepare_z2pack,
            if_(cls.should_do_alltogheter)(
                cls.run_z2pack_all,
                cls.inspect_z2pack_all
            ).else_(
                while_(cls.do_z2pack_one)(
                    cls.run_z2pack_one,
                    cls.inspect_z2pack_one,
                    ),
                ),
            cls.results
            )

        # OUTPUTS ############################################################################
        spec.output(
            'output_parameters', valid_type=orm.Dict,
            help='Dict resulting from a z2pack calculation.'
            )
        spec.expose_outputs(
            FindCrossingsWorkChain, namespace='find',
            exclude=('scf_remote_folder', ),
            namespace_options={
                'required':False, 'populate_defaults': False,
                'help': 'Outputs for the `FindCrossingsWorkChain`.'
                }
            )

        # ERRORS ############################################################################
        spec.exit_code(113, 'ERROR_SUB_PROCESS_FAILED_FINDCROSSING',
            message='the FindCrossingsWorkChain sub process failed')
        spec.exit_code(123, 'ERROR_SUB_PROCESS_FAILED_Z2PACK',
            message='the Z2packBaseWorkChain sub process failed')
        spec.exit_code(133, 'ERROR_INVALID_INPUT_CROSSINGS',
            message='Must provide either `find` namelist or `crossings` ArrayData as input.')
        spec.exit_code(143, 'ERROR_INVALID_INPUT_SCF_Z2PACK',
            message='If `crossings` is given, must also specify `scf_parent_folder` or `z2pack.scf`.')

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.report('STARTING Z2pack3DChernWorkChain')
        self.ctx.current_structure = self.inputs.structure
        self.ctx.radius = self.inputs.sphere_radius.value

    def validate_crossings(self):
        """Check validity of crossings input"""
        if all([key not in self.inputs for key in ['find', 'crossings']]):
            return self.exit_codes.ERROR_INVALID_INPUT_CROSSINGS
        
    def should_do_find_crossings(self):
        """If the 'find' input namespace was specified and `crossings` is not set, then try to find band crossings."""
        return not 'crossings' in self.inputs and 'find' in self.inputs

    def set_crossings_from_input(self):
        """Set crossings from given input. Ignore `find` namelist."""
        self.report('Setting `crossings` from input.')
        if 'find' in self.inputs:
            self.report('Both `crossings` and `find` provided as input. Ignoring `find`.')
        self.report('Taking crossings<{}> from input.'.format(self.inputs.crossings.pk))
        self.ctx.crossings_node = self.inputs.crossings
        self.ctx.crossings = self.inputs.crossings.get_array('crossings')

    def run_find_crossings(self):
        # """Run the FindCrossingsWorkChain to find bands crossings."""
        inputs = AttributeDict(self.exposed_inputs(FindCrossingsWorkChain, namespace='find'))
        inputs.structure = self.ctx.current_structure
        inputs.code = self.inputs.pw_code

        running = self.submit(FindCrossingsWorkChain, **inputs)

        self.report('launching FindCrossingsWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_find=running)

    def inspect_find_crossings(self):
        # """Verify that the FindCrossingsWorkChain finished successfully."""
        workchain = self.ctx.workchain_find

        if not workchain.is_finished_ok:
            self.report('FindCrossingsWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_FINDCROSSING

        self.ctx.crossings_node = workchain.outputs.crossings
        self.ctx.crossings  = workchain.outputs.crossings.get_array('crossings')
        self.ctx.remote_scf = workchain.outputs.scf_remote_folder
        if 'output_structure' in workchain.outputs:
            self.ctx.current_structure = workchain.outputs.output_structure

        self.out_many(
            self.exposed_outputs(
                workchain,
                FindCrossingsWorkChain,
            )
        )

    def prepare_z2pack(self):
        inputs = AttributeDict(self.exposed_inputs(Z2packBaseWorkChain, namespace='z2pack_base'))
        inputs.pw_code   = self.inputs.pw_code
        inputs.structure = self.ctx.current_structure

        if 'remote_scf' in self.ctx:
            # self.report('Setting scf remote_data from `FindCrossingsWorkChain` output.')
            inputs.parent_folder = self.ctx.remote_scf
        else:
            if not 'scf' in inputs.z2pack and not 'scf_parent_folder' in self.inputs:
                self.report('Neither `scf` nor `scf_parent_folder` was specified as an input. Aborting...')
                return self.exit_codes.ERROR_INVALID_INPUT_SCF_Z2PACK
            if 'scf_parent_folder' in self.inputs:
                # self.report('Setting scf remote_data from inputs.')
                inputs.parent_folder = self.inputs.scf_parent_folder

        self.ctx.inputs = inputs
        self.ctx.iteration = 0
        self.ctx.max_iteration = len(self.ctx.crossings)

    def should_do_alltogheter(self):
        from aiida.schedulers.plugins.direct import DirectScheduler

        computer  = self.inputs.pw_code.computer
        scheduler = computer.get_scheduler()

        return not isinstance(scheduler, DirectScheduler)

    def run_z2pack_all(self):
        cross = self.ctx.crossings[self.ctx.iteration]
        self.ctx.iteration += 1

        old = self.ctx.inputs.z2pack.z2pack_settings.get_dict()
        for cross in self.ctx.crossings:
            old.update({
                'dimension_mode':'3D',
                'invariant':'Chern',
                'surface':'z2pack.shape.Sphere(center=({0[0]:11.7f}, {0[1]:11.7f}, {0[1]:11.7f}), radius={1})'.format(cross, self.ctx.radius)
                })
            self.ctx.inputs.z2pack.z2pack_settings = orm.Dict(dict=old)

            running = self.submit(Z2packBaseWorkChain, **self.ctx.inputs)

            self.report('launching Z2packBaseWorkChain<{}> on center {}'.format(running.pk, cross))

            self.to_context(workchain_z2pack=append_(running))

    def inspect_z2pack_all(self):
        """Verify that the FindCrossingsWorkChain finished successfully."""
        for workchain in self.ctx.workchain_z2pack:
            if not workchain.is_finished_ok:
                self.report('Z2packBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_Z2PACK

    def do_z2pack_one(self):
        return self.ctx.iteration < self.ctx.max_iteration

    def run_z2pack_one(self):
        cross = self.ctx.crossings[self.ctx.iteration]
        self.ctx.iteration += 1

        old = self.ctx.inputs.z2pack.z2pack_settings.get_dict()
        old.update({
            'dimension_mode':'3D',
            'invariant':'Chern',
            'surface':'z2pack.shape.Sphere(center=({0[0]:11.7f}, {0[1]:11.7f}, {0[1]:11.7f}), radius={1})'.format(cross, self.ctx.radius)
            })
        self.ctx.inputs.z2pack.z2pack_settings = orm.Dict(dict=old)

        running = self.submit(Z2packBaseWorkChain, **self.ctx.inputs)

        self.report('launching Z2packBaseWorkChain<{}> on center {}'.format(running.pk, cross))

        return ToContext(workchain_z2pack=append_(running))

    def inspect_z2pack_one(self):
        """Verify that the FindCrossingsWorkChain finished successfully."""
        workchain = self.ctx.workchain_z2pack[-1]

        if not workchain.is_finished_ok:
            self.report('Z2packBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_Z2PACK

    def results(self):
        res = merge_chern_results(
            crossings=self.ctx.crossings_node,
            **{'z2calcOut_{}'.format(n):calc.outputs.output_parameters for n,calc in enumerate(self.ctx.workchain_z2pack)}
            )
        
        self.out('output_parameters', res)

        self.report('FINISHED')

