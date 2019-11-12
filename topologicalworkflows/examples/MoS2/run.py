from aiida.workflows.user.topologicalworkflows.topo import TopologicalWorkflow
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.structure import StructureData
#codes

z2pack_codename = 'z2pack@fidis'
codename = 'pw_6.1@fidis'
overlap_codename = 'pw2wannier90_6.1@fidis'
wannier90_codename = 'wannier90_2.1@fidis'
structure = load_node('2c602d9f-f2ba-40a2-922c-d30bd7cf72a8')
pseudo_family = 'dojo_GS_rel_v1'
wfc_cutoff = 50
dual = 4
kpoints = KpointsData()
kpoints.set_kpoints_mesh([6,6,1])
kpoints.store()
group_name = 'test_topowf'
input_dict = {'CONTROL': {
               'wf_collect':True,  # It helps with automatic parallelization
               },
              'SYSTEM': {
                 'ecutwfc': wfc_cutoff,
                 'ecutrho': wfc_cutoff*dual,
	             'noncolin' : True,
	             'lspinorb' : True,
                 'occupations' : 'fixed',
                 },
              'ELECTRONS': {
                 'conv_thr': 1.e-10,
                 }}

bandsx_input_dict = {'BANDS' : {
                 'parity': True,     #Custom flag for parity analysis (in your case maybe remove this flag and set lsym=.true.
                  'lsym':False,
                 },
                 }		   

z2pack_input={'invariant':'Z2','num_lines':7,'min_neighbour_dist':0.01}
z2pack_resources={'resources':{'num_machines': 1, 'num_mpiprocs_per_machine':28},
                  'time':60*10,'custom_scheduler_commands':'#SBATCH --partition=debug'}
set_dict={'custom_scheduler_commands':'#SBATCH --partition=debug'
                                      }
relaxation_scheme = 'scf'
max_time_seconds  = 60*60
target_time_seconds = 60*10
max_num_machines = 1
topo_wf_params = {
            #'previous_scf':load_node(15391),
            #'previous_bands':load_node(...),  #-->Uncomment if you want to start from a intermediate with TRIM states already computed
            'dimensionality': 2,  #-->Here you specify if you work in 2D "2" or 3D "3".
            'use_parities':False, #-->Here you specify if you want to use parities (False) or z2pack (True)
            'pseudo_family': pseudo_family,
            'pw_codename': codename,
            'pw_calculation_set': set_dict,
            'pw_input':{#'volume_convergence_threshold': 1.e-2,
                     'relaxation_scheme': relaxation_scheme,             
                     'automatic_parallelization': {             #You can disable automatic parallelization by commenting these 4 lines
                     'max_wall_time_seconds': max_time_seconds,  
                     'target_time_seconds': target_time_seconds,
                     'max_num_machines': max_num_machines
                     }
		     },
            'calculation_set': set_dict,
            #'band_calculation_set': set_dict,
            'structure': structure,
            'kpoints': kpoints,
            'pw_parameters': input_dict,
            #'settings': settings,
            #'band_settings': settings,
            #'input':{'volume_convergence_threshold': 1.e-2,
            #         'relaxation_scheme': relaxation_scheme,
            #         'automatic_parallelization': {
            #         'max_wall_time_seconds': max_time_seconds,
            #         'target_time_seconds': target_time_seconds,
            #         'max_num_machines': max_num_machines
            #         }
            #        },
	        #'band_input_set': set_dict,
            #'band_kpoints': kpoints,
             'band_parameters_update': {
                                       'ELECTRONS':{
                                                    'diagonalization':'cg',
                                                    }
                                       },
 	    # 'bandsx_codename': bandsxcodename,
	    # 'bandsx_input': bandsx_input_dict,
	      'z2pack_codename': z2pack_codename,
	      'wannier90_codename':wannier90_codename,
              'overlap_codename':overlap_codename,
	      'z2pack_input': z2pack_input,
              'z2pack_resources':z2pack_resources,
              'group_name':group_name,
            }
wf = TopologicalWorkflow(params=topo_wf_params)
wf.start()
print 'TopologicalWorkflow pk: {} submitted.'.format(wf.pk)
