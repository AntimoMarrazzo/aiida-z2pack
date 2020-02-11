import numpy as np
from scipy.spatial import KDTree

from aiida import orm
from aiida.common import AttributeDict
from aiida.plugins import WorkflowFactory, CalculationFactory, DataFactory
from aiida.engine import WorkChain, ToContext, if_, while_, append_

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.functions import create_kpoints_from_distance

from .functions import crop_kpoints, copy_array_data

# Z2packCalculation   = CalculationFactory('z2pack.z2pack')

PwBaseWorkChain     = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain    = WorkflowFactory('quantumespresso.pw.relax')
Z2packBaseWorkChain = WorkflowFactory('z2pack.base')

ArrayData = DataFactory('array')

class FindCrossingsWorkChain(WorkChain):
    """Workchain to find bands crossing in the Brillouin Zone using
    a series of quantum espresso nscf calculations."""

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
                'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'
                }
            )
        spec.expose_inputs(
            PwBaseWorkChain, namespace='nscf',
            exclude=('clean_workdir', 'pw.structure', 'pw.code', 'pw.pseudos', 'kpoints'),
            namespace_options={
                'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwBaseWorkChain` for the NSCF calculation.'
                }
            )

        spec.input(
            'max_iter', valid_type=orm.Int,
            default=orm.Int(4),
            help='Maximum number of iterations to perform.'
            )
        spec.input(
            'starting_kpoints', valid_type=orm.KpointsData,
            required=True,
            help='Starting mesh of kpoints'
            )
        spec.input(
            'mesh_scale_factor', valid_type=(orm.Int, orm.ArrayData),
            default=orm.Int(3),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )

        spec.input(
            'starting_gap_threshold', valid_type=orm.Float,
            default=orm.Float(0.03),
            help=(
                'Starting value for `gap_threshold`. All kpoints with a gap between valence and conduction '
                'lower than this threshold will be selected for the successive loops.'
                )
            )
        spec.input(
            'min_gap_threshold', valid_type=orm.Float,
            default=orm.Float(0.001),
            help=('Across iterations, `gap_threshold` will never drop below this value.'
                )
            )
        spec.input(
            'scale_gap_threshold', valid_type=orm.Float,
            default=orm.Float(3.0),
            help=('Between every iteration, divide `gap_threshold` by this scaling factor.'
                )
            )

        spec.input(
            'starting_crop_radius', valid_type=orm.Float,
            default=orm.Float(0.5),
            help=(
                'Starting value for `gap_threshold`. All kpoints with a gap between valence and conduction '
                'lower than this threshold will be selected for the successive loops.'
                )
            )
        spec.input(
            'min_crop_radius', valid_type=orm.Float,
            default=orm.Float(0.01),
            help=('Across iterations, `gap_threshold` will never drop below this value.'
                )
            )
        spec.input(
            'scale_crop_radius', valid_type=orm.Float,
            default=orm.Float(5.0),
            help=('Between every iteration, divide `gap_threshold` by this scaling factor.'
                )
            )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.run_scf,
            cls.inspect_scf,
            cls.setup_nscf_loop,
            while_(cls.should_find_zero_gap)(
                cls.setup_grid,
                cls.run_nscf,
                cls.analyze_bands,
                cls.post_analysis,
                cls.stepper
                ),
            cls.results
            )

        # OUTPUTS ############################################################################
        spec.output('crossings', valid_type=ArrayData,
            required=True,
            help='The array containing a list of bands crossing found as rows.'
            )

        # ERRORS ############################################################################
        spec.exit_code(112, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='the relax PwRelaxWorkChain sub process failed')
        spec.exit_code(122, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='the scf PwBaseWorkChain sub process failed')
        spec.exit_code(132, 'ERROR_SUB_PROCESS_FAILED_NSCF',
            message='the nscf PwBaseWorkChain sub process failed')
        spec.exit_code(142, 'ERROR_CANT_PINPOINT_LOWGAP_ZONE',
            message='After two iterations, no points with low_gap found. Aborting calculation!')
        spec.exit_code(152, 'ERROR_MAXIMUM_ITERATIONS_EXCEEDED',
            message='The maximum number of iterations was exceeded.')
        spec.exit_code(162, 'ERROR_TOO_MANY_ARRAYS',
            message='An ArrayData node contains more arrays than expected.')

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

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.pseudos = self.inputs.pseudos
        inputs.pw.code = self.inputs.code
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

    def setup_nscf_loop(self):
        self.ctx.iteration = 0
        self.ctx.max_iteration = self.inputs.max_iter
        if 'nscf' in self.inputs:
            self.ctx.inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='nscf'))
        else:
            self.ctx.inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))

        self.ctx.inputs.pw.parameters = self.ctx.inputs.pw.parameters.get_dict()

        workchain = self.ctx.workchain_scf
        pw_params = workchain.outputs.output_parameters.get_dict()
        n_el      = pw_params['number_of_electrons']
        spin      = pw_params['spin_orbit_calculation']

        self.ctx.n_el = n_el
        self.ctx.cb   = int(n_el) // (int(not spin) + 1)
        self.ctx.vb   = self.ctx.cb - 1

        mesh, offset = self.inputs.starting_kpoints.get_kpoints_mesh()
        self.ctx.offset                 = offset
        self.ctx.current_kpoints        = self.inputs.starting_kpoints
        self.ctx.current_mesh           = np.array(mesh)
        self.ctx.current_offset         = np.array(offset)

        scale_mesh = self.inputs.mesh_scale_factor
        if isinstance(scale_mesh, orm.Int):
            self.ctx.scale_mesh = scale_mesh.value
        else:
            arrays = list(scale_mesh.get_iterarrays())
            if len(arrays) > 1:
                return self.exit_codes.ERROR_TOO_MANY_ARRAYS
            self.ctx.scale_mesh = arrays[0][1]

        # self.ctx.scale_mesh             = self.inputs.mesh_scale_factor

        self.ctx.current_gap_threshold  = self.inputs.starting_gap_threshold.value
        self.ctx.min_gap_threshold      = self.inputs.min_gap_threshold.value
        self.ctx.scale_gap_threshold    = self.inputs.scale_gap_threshold.value

        self.ctx.current_crop_radius  = self.inputs.starting_crop_radius.value
        self.ctx.min_crop_radius      = self.inputs.min_crop_radius.value
        self.ctx.scale_crop_radius    = self.inputs.scale_crop_radius.value

        self.ctx.failed_find = 0

        self.ctx.do_loop = True
        self.ctx.found_crossings = []

        self.report('Starting loop to find bands crossings.')

    def should_find_zero_gap(self):
        """Limit iterations over kpoints meshes."""
        return self.ctx.iteration < self.ctx.max_iteration and self.ctx.do_loop

    def setup_grid(self):
        mesh = self.ctx.current_mesh

        kpt = orm.KpointsData()
        kpt.set_cell_from_structure(self.ctx.current_structure)
        kpt.set_kpoints_mesh(mesh, offset=self.ctx.offset)

        if self.ctx.iteration > 0 and len(self.ctx.pinned_points):
            centers = orm.ArrayData()
            centers.set_array('centers', np.array(self.ctx.pinned_points))

            cropped  = crop_kpoints(
                self.ctx.current_structure,
                kpt,
                centers,
                orm.Float(self.ctx.current_crop_radius)
                )

            self.ctx.current_kpoints = cropped

            nkpt = len(cropped.get_kpoints())
            self.report('Calculation with cropped mesh=`{}`, centers=`{}`, radius=`{}`'.format(
                mesh, self.ctx.pinned_points, self.ctx.current_crop_radius
                ))
            self.report('Cropped `{}` k-points out of the original grid.'.format(nkpt))
        else:
            self.report('Calculation with non-cropped mesh=`{}`'.format(mesh))
            self.ctx.current_kpoints = kpt

    def run_nscf(self):
        self.ctx.iteration += 1
        inputs = self.ctx.inputs
        inputs.kpoints       = self.ctx.current_kpoints
        inputs.clean_workdir = self.inputs.clean_workdir
        inputs.pw.structure  = self.ctx.current_structure
        inputs.pw.pseudos    = self.inputs.pseudos
        inputs.pw.code       = self.inputs.code
        # inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters['CONTROL']['calculation'] = 'nscf'
        inputs.pw.parent_folder = self.ctx.scf_folder

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode, iteration {}'.format(
            running.pk, 'nscf', self.ctx.iteration
            ))

        return ToContext(workchain_nscf=append_(running))
    
    def inspect_nscf(self):
        """Verify that the PwBaseWorkChain for the nscf run finished successfully."""
        workchain = self.ctx.workchain_nscf[self.ctx.iteration - 1]

        if not workchain.is_finished_ok:
            self.report('scf PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.bands = workchain.outputs.output_band

    def analyze_bands(self):
        """Extract kpoints with gap lower than the gap threshold"""
        self.report('Analyzing nscf results for BandsData')
        workchain = self.ctx.workchain_nscf[self.ctx.iteration - 1]
        node      = workchain.outputs.output_band
        bands     = node.get_bands()
        try:
            kpoints = node.get_kpoints()
        except:
            kpoints = node.get_kpoints_mesh(print_list=True)

        vb        = self.ctx.vb
        cb        = self.ctx.cb
        gaps      = bands[:,cb] - bands[:,vb]

        where     = np.where(gaps < self.ctx.current_gap_threshold)

        self.ctx.pinned_points = kpoints[where]
        self.ctx.pinned_gaps   = gaps[where]


    def post_analysis(self):
        """Sort found kpoints into `found_crossing` if the gap is lower then `min_gap_threshold` or
        `pinned_points` otherwise"""
        pinned = np.array(self.ctx.pinned_points)
        gaps   = np.array(self.ctx.pinned_gaps)
        if self.ctx.found_crossings:
            found = KDTree(self.ctx.found_crossings)
            curr  = KDTree(self.ctx.pinned_points)

            query = curr.query_ball_tree(found, r=self.ctx.min_crop_radius/5.)
            where = [n for n,l in enumerate(query) if not l]

            pinned = pinned[where]
            gaps   = gaps[where]


        self.ctx.found_some = bool(len(pinned))

        res = []
        for kpt, gap in zip(pinned, gaps):
            if gap < self.ctx.min_gap_threshold:
                self.ctx.found_crossings.append(kpt)
            else:
                res.append(kpt)

        self.ctx.pinned_points = res


    def stepper(self):
        """Perform the loop step operation of modifying the thresholds"""
        self.ctx.current_mesh = self.ctx.current_mesh * self.ctx.scale_mesh

        n_found = len(self.ctx.pinned_points) + len(self.ctx.found_crossings)
        if self.ctx.found_some:
            self.ctx.failed_find = 0

            cgt = self.ctx.current_gap_threshold
            mgt = self.ctx.min_gap_threshold
            sgt = self.ctx.scale_gap_threshold

            ccr = self.ctx.current_crop_radius
            mcr = self.ctx.min_crop_radius
            scr = self.ctx.scale_crop_radius

            self.report('`{}` points found with gap lower than the threshold `{}`'.format(n_found, cgt))

            new = max(cgt/sgt, mgt)
            self.ctx.current_gap_threshold = new
            self.report('Gap threshold reduced to `{}`'.format(new))

            new = max(ccr/scr, mcr)
            self.ctx.current_crop_radius = new
            
        else:
            self.report('No points with small gap found in this iteraton.')
            self.ctx.failed_find += 1
            if self.ctx.failed_find > 1:
                self.report('Failed to find a low_gap point after two consecutive iterations.')
                if self.ctx.found_crossings:
                    self.ctx.do_loop = False
                    return
                else:
                    return self.exit_codes.ERROR_CANT_PINPOINT_LOWGAP_ZONE

    def results(self):
        calculation = self.ctx.workchain_nscf[self.ctx.iteration - 1]
        if self.ctx.iteration >= self.ctx.max_iteration and not len(self.ctx.found_crossings):
            self.report('No crossing found. Reached the maximum number of iterations {}: last ran PwBaseWorkChain<{}>'.format(
                self.ctx.max_iteration, calculation.pk))
            return self.exit_codes.ERROR_MAXIMUM_ITERATIONS_EXCEEDED

        app = np.unique(np.array(self.ctx.found_crossings), axis=0)

        res = orm.ArrayData()
        res.set_array('crossings', app)

        res = copy_array_data(res)

        self.out('crossings', res)




class Z2pack3DChernWorkChain(WorkChain):
    """Workchain to compute topological invariants (Z2 or Chern number) using z2pack."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # INPUTS ############################################################################
        spec.input(
            'structure', valid_type=orm.StructureData,
            help='The inputs structure.'
            )
        spec.input(
            'clean_workdir', valid_type=orm.Bool,
            default=orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.'
            )

        spec.expose_inputs(
            FindCrossingsWorkChain, namespace='findc',
            exclude=('clean_workdir', 'structure'),
            namespace_options={
                'help': 'Inputs for the `FindCrossingsWorkChain`.'
                }
            )
        spec.expose_inputs(
            Z2packBaseWorkChain, namespace='z2pack',
            exclude=('clean_workdir', 'structure', 'scf'),
            namespace_options={
                'help': 'Inputs for the `FindCrossingsWorkChain`.'
                }
            )

        # OUTLINE ############################################################################
        spec.outline(
            cls.setup,
            if_(cls.should_do_find_crossings)(
                cls.run_find_crossings,
                cls.inspect_find_crossings,
            ),
            cls.run_z2pack,
            cls.inspect_z2pack,
            cls.results
            )

        # OUTPUTS ############################################################################
        # spec.output(
        #     'output_parameters', valid_type=orm.Dict,
        #     help='Dict resulting from a z2pack calculation.'
        #     )
        # spec.expose_outputs(Z2packCalculation)

        # ERRORS ############################################################################
        spec.exit_code(113, 'ERROR_SUB_PROCESS_FAILED_FINDCROSSING',
            message='the FindCrossingsWorkChain sub process failed')
        spec.exit_code(123, 'ERROR_SUB_PROCESS_FAILED_Z2PACK',
            message='the Z2packBaseWorkChain sub process failed')

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure
        
    def should_do_find_crossings(self):
        # """If the 'findc' input namespace was specified, we try to find band crossings."""
        return 'findc' in self.input_namespace

    def run_find_crossings(self):
        # """Run the FindCrossingsWorkChain to find bands crossings."""
        inputs = AttributeDict(self.exposed_inputs(FindCrossingsWorkChain, namespace='findc'))
        inputs.structure = self.ctx.current_structure

        running = self.submit(FindCrossingsWorkChain, **inputs)

        self.report('launching FindCrossingsWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_findc=running)

    def inspect_find_crossings(self):
        # """Verify that the FindCrossingsWorkChain finished successfully."""
        workchain = self.ctx.workchain_findc

        if not workchain.is_finished_ok:
            self.report('FindCrossingsWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_FINDCROSSING

        # self.ctx.current_structure = workchain.outputs.output_structure
        # pass

    def run_z2pack(self):
        inputs = AttributeDict(self.exposed_inputs(Z2packBaseWorkChain, namespace='findc'))
        inputs.structure = self.ctx.current_structure

        running = self.submit(Z2packBaseWorkChain, **inputs)

        self.report('launching Z2packBaseWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_z2pack=running)

    def inspect_z2pack(self):
        # """Verify that the FindCrossingsWorkChain finished successfully."""
        workchain = self.ctx.workchain_z2pack

        if not workchain.is_finished_ok:
            self.report('Z2packBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_Z2PACK

    def results(self):
        pass

