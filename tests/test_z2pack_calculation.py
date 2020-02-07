import os
import pytest
from aiida import orm
from aiida.common import datastructures
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.utils.resources import get_default_options

Z2packCalculation = CalculationFactory('z2pack.z2pack')

@pytest.fixture()
def pw_parameters():
    param = {
        'CONTROL': {
            'calculation': 'scf'
            },
        'SYSTEM': {
            'ecutrho': 240.0, 
            'ecutwfc': 30.0
            }
        }
    return param

@pytest.fixture()
def z2pack_settings():
    z2pack_settings = {
        'mpi_command':'mpirun -np 23',
        'dimension_mode':'3D',
        'invariant':'Chern',
        'surface':'lambda t1,t2: [t1, t2, 0]'
        }
    return z2pack_settings

def test_z2pack_inputs(
    aiida_profile, fixture_sandbox, generate_calc_job, fixture_code, generate_structure, generate_kpoints_mesh,
    generate_upf_data, file_regression, generate_remote_data, fixture_localhost,
    pw_parameters, z2pack_settings, tmpdir
):
    """Test a default `PwCalculation`."""
    new_param = {'SYSTEM':{'nbnd':50}}

    upf    = generate_upf_data('Si')
    struct = generate_structure()
    remote = generate_remote_data(
        fixture_localhost, str(tmpdir),
        'quantumespresso.pw',
        extras_root=[
            (pw_parameters, 'parameters'),
            (struct, 'structure'),
            (upf, 'pseudos__Si'),
            ]
        )
    f = tmpdir.mkdir('out').mkdir('aiida.save').join('data-file-schema.xml')
    f.write('123')

    inputs = {
        'code': fixture_code('z2pack.z2pack'),
        'parent_folder':remote,
        'pw_parameters': orm.Dict(dict=new_param),
        'z2pack_settings': orm.Dict(dict=z2pack_settings),
        'pw_code': fixture_code('quantumespresso.pw'),
        'overlap_code': fixture_code('quantumespresso.pw2wannier90'),
        'wannier90_code': fixture_code('wannier90.wannier90'),
        'metadata': {
            'options': get_default_options()
        }
    }

    process   = generate_calc_job('z2pack.z2pack', inputs)
    calc_info = process.prepare_for_submission(fixture_sandbox)

    inputs   = ['aiida.nscf.in', 'aiida.pw2wan.in', 'aiida.win', 'z2pack_aiida.py']
    outputs  = ['z2pack_aiida.out', 'save.json', 'results.json']
    errors   = ['build/aiida.werr', 'build/CRASH']

    cmdline_params = []
    local_copy_list = []
    retrieve_list = outputs + errors
    retrieve_temporary_list = []

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert sorted(calc_info.cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)
    assert sorted(calc_info.retrieve_temporary_list) == sorted(retrieve_temporary_list)
    assert sorted(calc_info.remote_symlink_list) == sorted([])

    # Checks on the files written to the sandbox folder as raw input
    retrieved_list = inputs + ['out']
    target = './tests/fixtures/calculation'
    # import shutil
    # for name in inputs:
    #     shutil.copy(
    #         fixture_sandbox.get_abs_path(name),
    #         os.path.join(target, name)
    #         )
    assert sorted(fixture_sandbox.get_content_list()) == sorted(retrieved_list)
    for name in retrieved_list:
        path = os.path.join(target, name)
        if not os.path.isfile(path):
            continue
            
        with open(path, 'r') as f:
            base_input = f.read()

        path = fixture_sandbox.get_abs_path(name)
        with open(path, 'r') as f:
            written_input = f.read()

        assert base_input == written_input

def test_nested_restart(
    aiida_profile, generate_calc_job, fixture_code, fixture_sandbox, generate_structure,
    generate_upf_data, generate_remote_data, fixture_localhost,
    pw_parameters, z2pack_settings, tmpdir
    ):
    """Test a default `PwCalculation`."""
    tmp_scf      = tmpdir.mkdir('scf')
    tmp_remote_1 = tmpdir.mkdir('remote_1')
    tmp_remote_2 = tmpdir.mkdir('remote_2')

    upf    = generate_upf_data('Si')
    param  = orm.Dict(dict=pw_parameters)
    struct = generate_structure()

    pseudo = upf
    remote_scf = generate_remote_data(
        fixture_localhost, str(tmp_scf),
        'quantumespresso.pw',
        extras_root=[
            (param, 'parameters'),
            (struct, 'structure'),
            (pseudo, 'pseudos__Si'),
            ]
        )
    remote_scf.store()

    pw_code            = fixture_code('quantumespresso.pw')
    overlap_code       = fixture_code('quantumespresso.pw2wannier90')
    wannier90_code     = fixture_code('wannier90.wannier90')

    remote_1 = generate_remote_data(
        fixture_localhost, str(tmp_remote_1),
        'z2pack.z2pack',
        extras_root=[
            ({'SYSTEM':{'lspinorb':True}}, 'pw_parameters'),
            (z2pack_settings, 'z2pack_settings'),
            ({}, 'wannier90_settings'),
            (remote_scf, 'parent_folder'),
            (pw_code, 'pw_code'),
            (overlap_code, 'overlap_code'),
            (wannier90_code, 'wannier90_code'),
            ]
        )
    remote_1.store()

    remote_2 = generate_remote_data(
        fixture_localhost, str(tmp_remote_2),
        'z2pack.z2pack',
        extras_root=[
            ({'SYSTEM':{'nbnd':50}}, 'pw_parameters'),
            (remote_1, 'parent_folder')
            ]
        )

    inputs = {
        'code': fixture_code('z2pack.z2pack'),
        'parent_folder':remote_2,
        'metadata': {
            'options': get_default_options()
        }
    }

    process   = generate_calc_job('z2pack.z2pack', inputs=inputs)
    process.prepare_for_submission(fixture_sandbox)
    
    test = process.inputs.pw_parameters.get_dict()
    assert test['SYSTEM']['ecutwfc'] == 30.0
    assert test['SYSTEM']['nbnd'] == 50
    assert test['SYSTEM']['lspinorb'] == True

