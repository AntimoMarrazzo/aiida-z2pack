import os
import copy
import shutil
from aiida import orm, load_profile
from aiida.orm.utils import load_node
from aiida.common import datastructures

from aiida_quantumespresso.utils.resources import get_default_options

def test_z2pack_inputs(
    aiida_profile, fixture_sandbox, generate_calc_job, fixture_code, generate_structure, generate_kpoints_mesh,
    generate_upf_data, file_regression, generate_remote_data, fixture_localhost
):
    """Test a default `PwCalculation`."""
    entry_point_name = 'z2pack'

    pw_parameters = {
        'CONTROL': {
            'calculation': 'scf'
            },
        'SYSTEM': {
            'ecutrho': 240.0, 
            'ecutwfc': 30.0
            }
        }

    wannier90_parameters = {
        # 'num_bands':84
        }

    z2pack_settings = {
        'mpi_command':'mpirun -np 1',
        'dimension_mode':'3D',
        'invariant':'Chern',
        'surface':'lambda t1,t2: [t1, t2, 0]'
        }

    upf = generate_upf_data('Si')
    upf.store()
    param  = orm.Dict(dict=pw_parameters)
    param.store()
    struct = generate_structure()
    struct.store()
    pseudo = upf
    remote = generate_remote_data(
        fixture_localhost, '/tmp',
        'quantumespresso.pw',
        extras_root=[
            (param, 'parameters'),
            (struct, 'structure'),
            (pseudo, 'pseudos__Si'),
            ]
        )
    # calc   = remote.get_incoming().first().node


    inputs = {
        'code': fixture_code('z2pack'),
        # 'structure': generate_structure(),
        'parent_folder':remote,

        # 'pw_parameters': orm.Dict(dict=pw_parameters),
        'wannier90_parameters': orm.Dict(dict=wannier90_parameters),

        'z2pack_settings': orm.Dict(dict=z2pack_settings),
        'pw_code': fixture_code('quantumespresso.pw'),
        'overlap_code': fixture_code('quantumespresso.pw2wannier90'),
        'wannier90_code': fixture_code('wannier90.wannier90'),
        # 'pseudos': {
        #     'Si': upf
        # },
        'metadata': {
            'options': get_default_options()
        }
    }

    # print(inputs)

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    inputs   = ['aiida.nscf.in', 'aiida.pw2wan.in', 'aiida.win', 'z2pack_aiida.py']
    # _outputs = ['aiida.scf.out', 'aiida.nscf.out', 'aiida.pw2wan.out', 'aiida.wout']
    outputs  = ['z2pack_aiida.out', 'save.json', 'results.json']
    # errors   = ['aiida.werr']

    # cmdline_params = ['-in', 'aiida.scf.in']
    cmdline_params = []
    # local_copy_list = [(upf.uuid, upf.filename, u'./pseudo/Si.upf')]
    local_copy_list = []
    # retrieve_list = ['aiida.out', './out/aiida.save/data-file-schema.xml', './out/aiida.save/data-file.xml']
    # retrieve_list = inputs + outputs + errors
    retrieve_list = outputs
    # retrieve_temporary_list = [['./out/aiida.save/K*[0-9]/eigenval*.xml', '.', 2]]
    retrieve_temporary_list = []

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert sorted(calc_info.cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    assert sorted(calc_info.retrieve_temporary_list) == sorted(retrieve_temporary_list)
    assert sorted(calc_info.remote_symlink_list) == sorted([])

    # Checks on the files written to the sandbox folder as raw input
    # print(os.path.realpath(os.path.curdir))
    retrieved_list = inputs
    target = './tests/test_z2pack'
    # for name in inputs:
    #     shutil.copy(
    #         fixture_sandbox.get_abs_path(name),
    #         os.path.join(target, name)
    #         )
    # assert sorted(fixture_sandbox.get_content_list()) == sorted(retrieved_list)
    for name in retrieved_list:
        path = os.path.join(target, name)
        with open(path, 'r') as f:
            base_input = f.read()

        path = fixture_sandbox.get_abs_path(name)
        with open(path, 'r') as f:
            written_input = f.read()

        assert base_input == written_input

    # for name in inputs:
    #     ext = os.path.splitext(name)[1]
    #     print(name, ext)
    #     with fixture_sandbox.open(name) as handle:
    #         input_written = handle.read()

    #     file_regression.check(input_written, encoding='utf-8', extension=ext)
