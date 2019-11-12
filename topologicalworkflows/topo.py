# -*- coding: utf-8 -*-
from aiida.orm.workflow import Workflow
from aiida.orm import DataFactory, Code, CalculationFactory, Group, Computer
from aiida.orm.calculation.job import z2pack as z2plugin
import helpers
from pw import PwrestartWorkflow
from pw import PwWorkflow
from aiida.common.exceptions import ValidationError,InternalError,NotExistent
from aiida.orm.calculation.inline import make_inline
import numpy
from itertools import chain

__copyright__ = u"Copyright (c), 2016, École Polytechnique Fédérale de Lausanne (EPFL), Switzerland, Laboratory of Theory and Simulation of Materials (THEOS). All rights reserved."
__license__ = "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file"
__version__ = "0.6.0"
__contributors__ = "Antimo Marrazzo (antimo.marrazzo@epfl.ch), Davide Campi (davide.campi@epfl.ch), THEOS, EPFL, Switzerland."

UpfData = DataFactory('upf')
ParameterData = DataFactory('parameter')
KpointsData = DataFactory('array.kpoints')
BandsData = DataFactory('array.bands')
StructureData = DataFactory('structure')
PwCalculation = CalculationFactory('quantumespresso.pw')
BandsCalculation = CalculationFactory('quantumespresso.bands')
Z2packCalculation = CalculationFactory('z2pack')
@make_inline
def store_invariant_inline(parameters):
    out_dict = parameters.get_dict()
    di = out_dict['invariant']
    inv_type = di.keys()[0]
    inv = di[inv_type]
    info = out_dict['convergence_report']
    result_dict = {'output_invariants':ParameterData(dict={'invariant_type':inv_type,
                                                         'invariant':inv,
                                                         'method':'z2pack',
                                                         'info': info})}

    return result_dict
@make_inline
def store_invariant_parities_inline(parameters):
    import numpy as np
    out_dict = parameters.get_dict()
    nu = out_dict['TRIM_product']
    info = out_dict['invariants_at_kpoints']
    warnings = []
    inv = None
    if np.isclose(nu,1,rtol=1e-9,atol=0.0):
        inv = 0
    elif np.isclose(nu,-1,rtol=1e-9,atol=0.0):
        inv = 1
    else:
        warnings.append("Nu = {}, Z2  computed with parities is neither 0 or 1! Check".format(nu))
    if len(warnings)!=0:
        result_dict = {'output_invariants':ParameterData(dict={'invariant_type':'Z2',
                                                         'invariant':inv,
                                                         'method':'parities',
                                                         'info': info,
                                                         'warnings':warnings,
                                                         }),
                   }
    else:
        result_dict = {'output_invariants':ParameterData(dict={'invariant_type':'Z2',
                                                         'invariant':inv,
                                                         'method':'parities',
                                                         'info': info,
                                                         }),
        }
    return result_dict
@make_inline 
def generate_trim_inline(input_parameters, structure):
    """
    :dimensionality: 2d or 3d material
    :return: dict with 'output_parameters' : ParameterData with 'output_kpoints' KpointData containing the grid
    """
    dimensionality = input_parameters.get_dict()['dimensionality']
    if dimensionality == 2:
        trim_grid = numpy.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0]])
    elif dimensionality == 3:
        trim_grid = numpy.array([[0.0,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.5,0.5,0.0],[0.0,0.0,0.5],[0.5,0.0,0.5],[0.0,0.5,0.5],[0.5,0.5,0.5]])
    else:
        raise KeyError("Dimensionality needs to be either 2 or 3")


    trim_grid_data = KpointsData()
    trim_grid_data.set_cell(structure.cell, structure.pbc)
    trim_grid_data.set_kpoints(trim_grid)

    output_dict = {
    'output_kpoints' : trim_grid_data }

    return output_dict

@make_inline 
def calculate_invariants_inline(nelec_data, symmetries_data):
    invariants = []
    nelec = nelec_data.get_attrs()['nelec']
    symmetries = symmetries_data.get_bands()
    nu = 1
 
    for i in range(0,symmetries.shape[0]):
        inv_k = 1
        parity_zero = -1
        for j in range(0,nelec,2):
            if ( symmetries[i][j] != symmetries[i][j+1] ):
                raise RuntimeError("symmetries should be in equal pairs, there is something fishy")
            if ( symmetries[i][j] == 'ND' ):
                raise RuntimeError("Parity is not defined")
            if abs(symmetries[i][j])<1e-2:
                parity_factor = parity_zero * 1
                parity_zero *= -1
            else:
                parity_factor = symmetries[i][j]
            inv_k = inv_k * parity_factor  #* symmetries[i][j]
        invariants.append(inv_k)
        nu = nu * inv_k

    #TODO implement the multiple Z2s for the 3D case

    output_dict = {
    'output_invariants' : ParameterData( dict = { 'invariants_at_kpoints': invariants,
						  'TRIM_product': nu, })
    }

    return output_dict
    
class TopologicalWorkflow(Workflow):
    """
    :dimensionality: 2D or 3D material
    """
    _default_dimensionality = 2

    def __init__(self, **kwargs):
        super(TopologicalWorkflow, self).__init__(**kwargs)

    @Workflow.step
    def start(self):
        """
        Check input parameters
        :return:
        """
        self.append_to_report("Checking input parameters")
        params = self.get_parameters()
        always_mandatory_key=[('use_parities', bool, "True to use parities instead of WCC")]
        helpers.validate_keys(params, always_mandatory_key)
        try:
            par=params['use_parities']
            if par is True:
                global_mandatory_keys = [  ('dimensionality', int, "2D or 3D material"),
                    ('pseudo_family', basestring, 'the pseudopotential family'),
        	        ('structure',StructureData,"the structure (a previously stored StructureData object)"),
                    ('pw_codename',basestring,'the PW codename'),
                    ('pw_input',dict,'pw workflow inputs'),
                    ('pw_calculation_set',dict,'A dictionary with resources, walltime, ... for pw calcs.'),
        	        ('bandsx_codename',basestring,'the bands.x codename'),
        	        ('bandsx_input',dict,'bands.x inputs'),
                    ]
                helpers.validate_keys(params, global_mandatory_keys)
                try:
                    bands=params['previous_bands']
                    self.append_to_report('Found previous bands, (pk {})'.format(bands.pk))
                    self.next(self.calculate_symmetries)
                except KeyError:
                    self.next(self.calculate_trim)
            else:
                global_mandatory_keys = [  ('dimensionality', int, "2D or 3D material"),
                    ('pseudo_family', basestring, 'the pseudopotential family'),
                    ('structure',StructureData,"the structure (a previously stored StructureData object)"),
                    ('pw_codename',basestring,'the PW codename'),
                    ('pw_input',dict,'pw workflow inputs'),
                    ('pw_calculation_set',dict,'A dictionary with resources, walltime, ... for pw calcs.'),
                    ('z2pack_codename',basestring,'the z2pack codename'),
                    ('z2pack_input',dict,'z2pack inputs'),
                    ]
                helpers.validate_keys(params, global_mandatory_keys)
                try:
                    scf=params['previous_scf']
                    self.append_to_report('Found previous scf, (pk {})'.format(scf.pk))
                    try:
                        z2pack_calc = params['previous_z2pack_calc']
                        self.append_to_report('Found previous z2pack calc, (pk {})'.format(z2pack_calc.pk))
                        self.next(self.calculate_invariant)
                    except KeyError:
                        self.next(self.z2pack_calc)
                except KeyError:
                    self.next(self.scf_step)
        except KeyError:
            self.append_to_report('Error in the use_parities reading.')
        helpers.validate_keys(params, global_mandatory_keys)

    #TODO check if non-symmorphic because symmetry decomposition at borderd not available for non-symmorphic groups
    #TODO recenter cell in a clever way

    @Workflow.step
    def calculate_trim(self):
    
        self.append_to_report("Building TRIM points and computing scf and bands")

        from pw import PwWorkflow

        params = self.get_parameters()
   
        structure = params['structure']
        input_dict = { 'dimensionality': params['dimensionality'] }
        input_parameters = ParameterData(dict=input_dict)
        _, result_dict = generate_trim_inline(input_parameters=input_parameters,structure=structure)

        pw_params = {}

        pw_params = {'structure': structure}
        for k,v in params.iteritems():
            if k.startswith('pw_'):
                new_k = k[3:]
                pw_params[new_k] = v
            elif k == 'pseudo_family':
                pw_params[k] = v
                
        
        for k,v in params.iteritems():
            if k.startswith('band_'):
                pw_params[k] = v                


        pw_params['band_kpoints'] = result_dict['output_kpoints']
    # TODO: raise an error if some kpoints specifications are put in params['band_input']
  
        wf_pw = PwWorkflow(params=pw_params)
        self.attach_workflow(wf_pw)
        wf_pw.start()

        self.next(self.calculate_symmetries)
    @Workflow.step
    def scf_step(self):
    
        self.append_to_report("Launching scf")

        from pw import PwWorkflow

        params = self.get_parameters()
   
        structure = params['structure']
        pw_params = {}

        pw_params = {'structure': structure}
        for k,v in params.iteritems():
            if k.startswith('pw_'):
                new_k = k[3:]
                pw_params[new_k] = v
            elif k == 'pseudo_family':
                pw_params[k] = v

        wf_pw = PwWorkflow(params=pw_params)
        self.attach_workflow(wf_pw)
        wf_pw.start()
        self.next(self.z2pack_calc)

    @Workflow.step
    def calculate_symmetries(self):
    #   
        self.append_to_report("Computing symmetries with bands.x")
        params = self.get_parameters()
        try:
            pw_calculation=params['previous_bands']
            self.append_to_report('Using previous bands structure as input for bands.x, (pk {})'
                                  .format(pw_calculation.pk))
        except KeyError:
            pw_calculation=self.get_all_calcs()[-1]
        params = self.get_parameters()


        bandsx_parameters = ParameterData(dict=params['bandsx_input'])
        
        code = Code.get_from_string(params['bandsx_codename'])
        bandsx_calc = code.new_calc()
        bandsx_calc.use_parameters(bandsx_parameters)
        bandsx_calc.use_parent_calculation(pw_calculation)

        bandsx_settings_dict = { 'resources' : { 'num_machines': pw_calculation.get_resources()['num_machines'],
                                                 'num_mpiprocs_per_machine': pw_calculation.get_resources()['default_mpiprocs_per_machine'],
                                                },
	                         'max_wallclock_seconds' : pw_calculation.get_max_wallclock_seconds(),
	                         'custom_scheduler_commands' : pw_calculation.get_custom_scheduler_commands(),
				}

        bandsx_calc = helpers.set_the_set(bandsx_calc,bandsx_settings_dict) 
    #bandsx_calc.use_settings(ParameterData(dict=bandsx_settings_dict))
        bandsx_calc.store_all()

        self.append_to_report("Launching bands.x (pk: {})".format(bandsx_calc.pk))
        self.attach_calculation(bandsx_calc)


        self.next(self.calculate_invariant)

    @Workflow.step
    def z2pack_calc(self):
        params = self.get_parameters()
        code_z2pack = Code.get_from_string(params['z2pack_codename'])
        code_overlap = Code.get_from_string(params['overlap_codename'])
        code_wannier90 = Code.get_from_string(params['wannier90_codename'])
        code_nscf = Code.get_from_string(params['pw_codename'])
        pseudo_family = params['pseudo_family']
        z2calc = code_z2pack.new_calc()
        try:
            parent_scf=params['previous_scf']            
        except KeyError:
            parent_scf=self.get_step_workflows(self.scf_step)[0].get_result('pw_calculation')
        try:
            max_iterations_z2pack = params['max_iterations_z2pack']            
        except KeyError:
            max_iterations_z2pack = 10 

        structure=parent_scf.inp.structure
        nelec=int(parent_scf.res.number_of_electrons)
        s=structure
        ecut=parent_scf.inp.parameters.get_dict()['SYSTEM']['ecutwfc']
        ecut_rho=parent_scf.inp.parameters.get_dict()['SYSTEM']['ecutrho']
        try:
            occupations = parent_scf.inp.parameters.get_dict()['SYSTEM']['occupations']
            if occupations != 'fixed':
                try:
                    smearing_type = parent_scf.inp.parameters.get_dict()['SYSTEM']['smearing']
                    smearing_degauss = parent_scf.inp.parameters.get_dict()['SYSTEM']['degauss']
                except KeyError:
                    smearing_type = 'mv'
                    smearing_degauss = '0.02'
        except KeyError:
            occupations = 'fixed'
        settings = ParameterData(dict={'mpi_command':'srun',
                'invariant':'Z2',
                'restart':False,
                })
        settings.update_dict(params['z2pack_input'])
        z2pack_resources={'resources':{'num_machines': 1, 'num_mpiprocs_per_machine':1},
                          'time':60}
        z2pack_resources.update(params['z2pack_resources'])
        
        #structure.set_pbc([True,True,False])
        nscf_parameters_dict = {
        'CONTROL': {
                    'calculation': 'nscf',
                    'restart_mode': 'from_scratch',
                    'wf_collect': True,
                    },
        'SYSTEM': {
                    'ecutwfc': ecut,
                    'ecutrho': ecut_rho,
                    'noncolin': True,
                    'lspinorb': True,
                    'nosym':True,
                    'occupations':occupations
                  },
        'ELECTRONS': {
                    'conv_thr': 1.e-10,
                    'diagonalization':'cg',
                  }
                                              }
        if occupations != 'fixed':
            nscf_parameters_dict['SYSTEM']['smearing'] = smearing_type
            nscf_parameters_dict['SYSTEM']['degauss'] = smearing_degauss
        try:
            tefield = parent_scf.inp.parameters.get_dict()['CONTROL']['tefield']
            self.append_to_report('E-field mode (tefield variable was set to True in scf calc).'.format(tefield))
            if tefield:
                try:
                    efield_amp = parent_scf.inp.parameters.get_dict()['SYSTEM']['eamp']
                    try:
                        edir = parent_scf.inp.parameters.get_dict()['SYSTEM']['edir']
                    except KeyError:
                        self.append_to_report('E-field direction (edir) not found,'
                                              ' resorting to default (edir = 3)')
                        edir = 3
                    try:
                        emaxpos = parent_scf.inp.parameters.get_dict()['SYSTEM']['emaxpos']
                    except KeyError:
                        self.append_to_report('E-field max pos (emaxpos) not found,'
                                                  ' resorting to default (emaxpos = 0.98)')
                        emaxpos = 0.98
                    try:
                        eopreg = parent_scf.inp.parameters.get_dict()['SYSTEM']['eopreg']
                    except KeyError:
                        self.append_to_report('E-field eopreg (eopreg) not found,'
                                                  ' resorting to default (eopreg = 0.04)')
                        eopreg = 0.04
                    nscf_parameters_dict['CONTROL']['tefield'] = tefield
                    nscf_parameters_dict['SYSTEM']['eamp'] = efield_amp
                    nscf_parameters_dict['SYSTEM']['edir'] = edir
                    nscf_parameters_dict['SYSTEM']['emaxpos'] = emaxpos
                    nscf_parameters_dict['SYSTEM']['eopreg'] = eopreg
                except KeyError:
                    raise KeyError('ERROR: E-field amplitude (eamp) not found!')
        except KeyError:
            tefield = None
        nscf_parameters = ParameterData(dict=nscf_parameters_dict)
        overlap_parameters={'wan_mode':'standalone',
                            'write_amn' : False,
                            'write_mmn' : True,
                            'regular_mesh' : False,
                            }
        overlap_parameters = ParameterData(dict=overlap_parameters)
        use_2d_cutoff = False
        dim=params['dimensionality']
        if dim==2:
            self.append_to_report('2D system. Updating input files accordingly.')
            dic=nscf_parameters.get_dict()
            if use_2d_cutoff:
                dic['SYSTEM'].update({'do_cutoff_2D': True})
            nscf_parameters = ParameterData(dict=dic)
            settings.update_dict({'dimension_mode':'2D'})
        wannier90_parameters = ParameterData(dict={ 'num_wann':nelec,
                        'num_bands': nelec,
                        'shell_list': 1,
                        'use_bloch_phases': True,
                        'spinors':True,
                        'num_iter': 0,
                        'use_bloch_phases' : True,
                        'skip_b1_tests':True,
                        'postproc_setup':True,
                                })
        
        z2pack_list=list(self.get_step(self.z2pack_calc).get_calculations().order_by('ctime'))
        launch=True
        if len(z2pack_list)==0:
            self.append_to_report('No previous Z2pack calculation found, launching the first')
            launch=True
            z2calc.use_parent_calculation(parent_scf)
        elif len(z2pack_list)< max_iterations_z2pack:
            last_z2pack_calc=z2pack_list[-1]
            settings_dict = last_z2pack_calc.inp.settings.get_dict()
            settings = ParameterData(dict=settings_dict)
            last_z2pack_calc_status=last_z2pack_calc.get_state()
            if last_z2pack_calc_status=='FAILED':
                if 'TO TIME LIMIT' in last_z2pack_calc.get_scheduler_error():
                    z2calc.use_parent_calculation(last_z2pack_calc)
                    wall_time=last_z2pack_calc.get_max_wallclock_seconds()
                    new_wall_time=wall_time*2
                    max_wall_time_allowed = 60 * 60 * 23 
                    if new_wall_time > max_wall_time_allowed:
                        new_wall_time = max_wall_time_allowed
                    z2pack_resources.update({'time':new_wall_time})
                    self.append_to_report('Last Z2pack calculation (pk: {}) reached wall time ({} hr). '
                                          'I increased it ({} hr).'.format(last_z2pack_calc.pk,
                                                                        wall_time/3600,
                                                                        new_wall_time/3600))
                    launch=True
                    settings.update_dict({'restart':True})
                else:
                    diago=last_z2pack_calc.inp.nscf_parameters.get_dict()['ELECTRONS']['diagonalization']
                    if diago=='cg':
                        self.append_to_report('Last Z2pack calculation (pk: {}) failed. '
                                          'I try to change the diagonalization mode'.format(last_z2pack_calc.pk))
                        nscf_parameters.update_dict({'ELECTRONS':{'diagonalization':'david'}})
                        self.append_to_report('Diagonalization mode set to davidson. Restarting from scf.')
                        settings.update_dict({'restart':False})
                        z2calc.use_parent_calculation(parent_scf)
                    else:
                        self.append_to_report('Last Z2pack calculation (pk: {}) failed. '
                                          'I have already tried to change from cg to david.'
                                          'Invariant cannot be computed'.format(last_z2pack_calc.pk))
                        
                        launch=False
  
                    
                
            elif last_z2pack_calc_status=='SUBMISSIONFAILED':
                self.append_to_report('Last Z2pack calculation (pk: {}) failed submission'
                                          'I relaunch it from the scf.'.format(last_z2pack_calc.pk))
                launch=True
                settings.update_dict({'restart':False})
                z2calc.use_parent_calculation(parent_scf)
            elif last_z2pack_calc_status=='FINISHED':    
                tests=last_z2pack_calc.res.Tests_passed
                if tests is True:
                    self.append_to_report('Z2pack calculation found (pk: {}),'
                                          'all tests passed'.format(last_z2pack_calc.pk))
                    launch=False
                else:
                    d=last_z2pack_calc.res.convergence_report
                    move_fails=len(d['MoveCheck']['FAILED'])
                    if move_fails!=0:
                        self.append_to_report('MoveCheck test has failed {} times'.format(move_fails))
                        try:
                            min_neigh_dist = last_z2pack_calc.inp.settings.get_dict()['min_neighbour_dist']
                        except KeyError:
                            min_neigh_dist = 0.01
                        min_neigh_dist = min_neigh_dist*0.5
                        num_lin = last_z2pack_calc.inp.settings.get_dict()['num_lines']
                        num_lin = int(num_lin*1.5)
                        max_num_lin = 100
                        settings.update_dict({'min_neighbour_dist':min_neigh_dist})
                        self.append_to_report('min_neighbour_dist reduced of 50%, new value {}'.format(min_neigh_dist))
                        if num_lin>max_num_lin:
                            num_lin = max_num_lin
                            self.append_to_report('num_lines cannot be higher than {},'
                                                  ' new value {}'.format(max_num_lin,num_lin))
                        else:
                            self.append_to_report('num_lines increased by 50%, new value {}'.format(num_lin))

                        settings.update_dict({'num_lines':num_lin})                        
                        #self.append_to_report('num_lines doubled, new value {}'.format(num_lin))
                        launch=True
                        settings.update_dict({'restart':True})
                        z2calc.use_parent_calculation(last_z2pack_calc)
                    else:
                        self.append_to_report('GapCheck or PosCheck tests failed, goint to final step anyway.')
                        launch=False
            else:
                self.append_to_report('Warning! Last Z2pack calculation (pk: {})' 
                                      'was neither FINISHED nor FAILED'.format(last_z2pack_calc.pk))   
        else:
            self.append_to_report('More than {} Z2pack calculations done, computing invariant.'.format(max_iterations_z2pack))
            launch=False
        z2calc.use_structure(s)
        z2calc.use_nscf_parameters(nscf_parameters)
        z2calc.use_overlap_parameters(overlap_parameters)
        z2calc.use_wannier90_parameters(wannier90_parameters)
    
        z2calc.use_pseudos_from_family(pseudo_family)
        z2calc.use_nscf_code(code_nscf)   
        z2calc.use_overlap_code(code_overlap)
        z2calc.use_wannier90_code(code_wannier90)
        z2calc.use_settings(settings)
        z2calc.set_max_wallclock_seconds(z2pack_resources['time']) 
        z2calc.set_resources(z2pack_resources['resources'])
        try:
            z2calc.set_custom_scheduler_commands(z2pack_resources['custom_scheduler_commands'])
        except KeyError:
            pass        
        #z2calc.label = "..."
        if launch:
            z2calc.store_all()
            self.append_to_report("Launching Z2pack (pk: {})".format(z2calc.pk))
            self.attach_calculation(z2calc)
            self.next(self.z2pack_calc)
        else:       
            self.next(self.calculate_invariant)
    @Workflow.step
    def calculate_invariant(self):
        import numpy as np
        params = self.get_parameters()
        par=params['use_parities']
        try:
            group_name = params['group_name']
        except KeyError:
            group_name = None
        if par is True:     
            result_dict={}     
            
            bandsx_calculation=self.get_all_calcs()[-1]
            pw_calculation = bandsx_calculation.inp.parent_calc_folder.inp.remote_folder
            nelec_data = ParameterData( dict = {'nelec': int(pw_calculation.res.number_of_electrons),})
            _, out_dict = calculate_invariants_inline(nelec_data=nelec_data, symmetries_data=bandsx_calculation.out.output_parities)
            #nu=out_dict['output_invariants'].get_dict()['TRIM_product']
            #info=out_dict['output_invariants'].get_dict()['invariants_at_kpoints']

            _,result_dict=store_invariant_parities_inline(parameters=out_dict['output_invariants'])
                                                   
            try:
                warnings = result_dict['output_invariants'].get_dict()['warnings']
            except KeyError:
                warnings = []
            inv = result_dict['output_invariants'].get_dict()['invariant']
            info = result_dict['output_invariants'].get_dict()['info'] 
            self.append_to_report('Z2 invariant = {}'.format(inv))
            self.append_to_report('Parities at 4 kpoints = {}'.format(info))
            if len(warnings)==0:
                try:  
                    g,_=Group.get_or_create(name=group_name)
                    g.add_nodes(result_dict['output_invariants'])
                    self.append_to_report('Results (pk: {}) added to group {} (pk: {})'.format(result_dict['output_invariants'].pk,g.name,g.pk))
                except KeyError:
                    pass
            else:
                for warn in warnings:
                    self.append_to_report(warn)
            self.add_result('invariants', result_dict['output_invariants'])
        else:
            try:
                z2pack_calc=params['previous_z2pack_calc']
            except KeyError:
                z2pack_calc=list(self.get_step(self.z2pack_calc).get_calculations().order_by('ctime'))[-1]            
            try:
                di=z2pack_calc.res.invariant
                self.append_to_report('{} invariant = {}'.format(di.keys()[0], di[di.keys()[0]]))
                _,result_dict=store_invariant_inline(parameters=
                                                   z2pack_calc.get_outputs_dict()['output_parameters'])
                try:  
                    g,_=Group.get_or_create(name=group_name)
                    g.add_nodes(result_dict['output_invariants'])
                    self.append_to_report('Results (pk: {}) added to group {} (pk: {})'.format(result_dict['output_invariants'].pk,g.name,g.pk))
                except KeyError:
                    pass
                self.add_result('invariants', result_dict['output_invariants'])
            except NotExistent:
                self.append_to_report('No invariant found.')

        self.next(self.exit)

