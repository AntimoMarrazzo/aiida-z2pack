#!/usr/bin/env python
import z2pack
import json

z2cmd =(
    'ln -s ../out .; ln -s ../pseudo .;'
    ' /bin/true aiida -pp;' +
    ' mpirun -np 1 /bin/true < aiida.nscf.in >& aiida.nscf.out;' +
    ' mpirun -np 1 /bin/true < aiida.pw2wan.in  >& aiida.pw2wan.out;'
)

input_files = ['aiida.nscf.in', 'aiida.pw2wan.in', 'aiida.win']
system = z2pack.fp.System(
    input_files = input_files,
    kpt_fct     = [z2pack.fp.kpoint.qe_explicit, z2pack.fp.kpoint.wannier90_full],
    kpt_path    = ['aiida.nscf.in', 'aiida.win'],
    command     = z2cmd,
    executable  = '/bin/bash',
    mmn_path    = 'aiida.mmn'
)

gap_check={}
move_check={}
pos_check={}
res_dict={'convergence_report':{'GapCheck':{}, 'MoveCheck':{}, 'PosCheck':{}}, 'invariant':{}}

result = z2pack.surface.run(
    system             = system,
    surface            = lambda t1,t2: [t1, t2, 0],
    pos_tol            = 0.01,
    gap_tol            = 0.3,
    move_tol           = 0.3,
    num_lines          = 11,
    min_neighbour_dist = 0.01,
    iterator           = range(8, 41, 2),
    save_file          = 'save.json',
    )
Chern = z2pack.invariant.chern(result)
res_dict['invariant'].update({'Chern':Chern})

gap_check['PASSED']  = result.convergence_report['surface']['GapCheck']['PASSED']
gap_check['FAILED']  = result.convergence_report['surface']['GapCheck']['FAILED']
move_check['PASSED'] = result.convergence_report['surface']['MoveCheck']['PASSED']
move_check['FAILED'] = result.convergence_report['surface']['MoveCheck']['FAILED']
pos_check['PASSED']  = result.convergence_report['line']['PosCheck']['PASSED']
pos_check['FAILED']  = result.convergence_report['line']['PosCheck']['FAILED']
pos_check['MISSING'] = result.convergence_report['line']['PosCheck']['MISSING']

res_dict['convergence_report']['GapCheck'].update(gap_check)
res_dict['convergence_report']['MoveCheck'].update(move_check)
res_dict['convergence_report']['PosCheck'].update(pos_check)

with open('results.json', 'w') as fp:
    json.dump(res_dict, fp)



