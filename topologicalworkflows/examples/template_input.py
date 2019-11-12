from aiida.workflows.user.topologicalworkflows.topo import TopologicalWorkflow
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

z2pack_input={'invariant':'Z2','num_lines':51,'min_neighbour_dist':0.01}
z2pack_resources={'resources':{'num_machines': 4, 'num_mpiprocs_per_machine':18},
                  'time':60*60*4}
set_dict={}
relaxation_scheme = 'scf'
topo_wf_params = {#'pw_calculation':load_node(...),
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
 	     'bandsx_codename': bandsxcodename,
	     'bandsx_input': bandsx_input_dict,
	     'z2pack_codename': z2pack_codename,
	     'wannier90_codename':wannier90_codename,
             'overlap_codename':overlap_codename,
	     'z2pack_input': z2pack_input,
             'z2pack_resources':z2pack_resources,
             'group_name':group_name,
            }
wf = TopologicalWorkflow(params=topo_wf_params)
wf.start()
