from __future__ import absolute_import
import os
import pytest
from aiida import orm
from aiida.common import datastructures
from aiida.plugins import CalculationFactory

from aiida_quantumespresso.utils.resources import get_default_options

Z2packCalculation = CalculationFactory('z2pack.z2pack')

@pytest.fixture()
def pw_parameters():
    """Fixture: parameters for pw calculation."""
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

@pytest.fixture(params=[(),], ids=['base',])
def z2pack_settings(request):
    """Fixture: parameters for z2pack calculation."""
    z2pack_settings = {
        'mpi_command':'mpirun -np 23',
        'dimension_mode':'3D',
        'invariant':'Chern',
        'surface':'lambda t1,t2: [t1, t2, 0]'
        }

    if request.param:
        k,v = request.param
        z2pack_settings[k] = v

    return z2pack_settings

@pytest.fixture()
def inputs(fixture_code):
    """Fixture: inputs for Z2packBaseWorkChain."""
    inputs = {
        'code': fixture_code('z2pack.z2pack'),
        'pw_code': fixture_code('quantumespresso.pw'),
        'overlap_code': fixture_code('quantumespresso.pw2wannier90'),
        'wannier90_code': fixture_code('wannier90.wannier90'),
        'metadata': {
            'options': get_default_options()
        }
    }
    return inputs

@pytest.fixture
def remote(
    aiida_profile, fixture_localhost, tmpdir,
    generate_upf_data, generate_structure, generate_remote_data,
    pw_parameters
    ):
    """Fixture: Remote folder created by a CalcJob with inputs.

    :param pw_parameters: parameters to be linked as input node to the parent CalcJob.
    """
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

    return remote

@pytest.fixture
def calc_info(
    aiida_profile, fixture_sandbox,
    remote, generate_calc_job,
    z2pack_settings, inputs
    ):
    """Fixture: calc_info obtained by the bound method `prepare_for_submission` of Z2packCalculation."""
    inputs['parent_folder'] = remote
    inputs['z2pack_settings'] = orm.Dict(dict=z2pack_settings)

    process   = generate_calc_job('z2pack.z2pack', inputs)
    calc_info = process.prepare_for_submission(fixture_sandbox)

    return calc_info

class Test_z2pack_calc():
    """Test class for Z2packCalculation."""
    inputs  = ['aiida.nscf.in', 'aiida.pw2wan.in', 'aiida.win', 'z2pack_aiida.py']
    outputs = ['z2pack_aiida.out', 'save.json', 'results.json']
    errors  = ['build/aiida.werr', 'build/CRASH']
    remotes = ['./out/', './pseudo/']

    cmdline_params = []
    local_copy_list = []
    retrieve_temporary_list = []
    retrieve_list = outputs + errors

    def test_calcinfo_type(self, calc_info):
        """Test if calc_info is properly being returned by `prepare_for_submission`."""
        assert isinstance(calc_info, datastructures.CalcInfo), 'Unexpected return from `prepare_for_submission`.'

    def test_cmdline_params(self, calc_info):
        """Test if the proper `cmdlin` is being set in calc_info."""
        assert sorted(calc_info.cmdline_params) == sorted(self.cmdline_params)

    def test_local_copy_list(self, calc_info):
        """Test if the proper `local_copy_list` is being set in calc_info."""
        assert sorted(calc_info.local_copy_list) == sorted(self.local_copy_list)

    def test_retrieve_list(self, calc_info):
        """Test if the proper `retrieve_list` is being set in calc_info."""
        assert sorted(calc_info.retrieve_list) == sorted(self.retrieve_list)

    def test_retrieve_temporary_list(self, calc_info):
        """Test if the proper `retrieve_temporary_list` is being set in calc_info."""
        assert sorted(calc_info.retrieve_temporary_list) == sorted(self.retrieve_temporary_list)

    @pytest.mark.parametrize(
        'z2pack_settings,with_symlink',
        [
            ((), False),
            (('parent_folder_symlink', True),True)], ids=['base', 'with_symlink'
        ],
        indirect=['z2pack_settings']
        )
    def test_remote_copy_list(self, calc_info, remote, with_symlink):
        """Test if the proper `remote_copy_list` is being set in calc_info."""
        if not with_symlink:
            remote_copy_list = [
                (
                    remote.computer.uuid,
                    os.path.join(remote.get_remote_path(), path),
                    path
                ) for path in self.remotes
                ]

            assert sorted(calc_info.remote_copy_list) == remote_copy_list
        else:
            assert sorted(calc_info.remote_copy_list) == sorted([])

    @pytest.mark.parametrize(
        'z2pack_settings,with_symlink',
        [
            ((), False),
            (('parent_folder_symlink', True),True)], ids=['base', 'with_symlink'
        ],
        indirect=['z2pack_settings']
        )
    def test_remote_symlink_list(self, calc_info, remote, with_symlink):
        """Test if the proper `remote_symlink_list` is being set in calc_info."""
        if with_symlink:
            remote_symlink_list = [
                (
                    remote.computer.uuid,
                    os.path.join(remote.get_remote_path(), path),
                    path
                ) for path in self.remotes
                ]

            assert sorted(calc_info.remote_symlink_list) == remote_symlink_list
        else:
            assert sorted(calc_info.remote_symlink_list) == sorted([])

    def test_input_created(self, calc_info, fixture_sandbox):
        """Test if all the required input files are bing created."""
        assert sorted(fixture_sandbox.get_content_list()) == sorted(self.inputs)

    @pytest.mark.parametrize('name', inputs)
    def test_input_files(self, name, calc_info, fixture_sandbox, file_regression):
        """Test if all the required input files are equal to the test prototype."""
        path = fixture_sandbox.get_abs_path(name)
        with open(path, 'r') as f:
            written_input = f.read()

        file_regression.check(written_input, encoding='utf-8', extension='.in')


def test_nested_restart(
    aiida_profile, generate_calc_job, fixture_code, fixture_sandbox, generate_structure,
    generate_upf_data, generate_remote_data, fixture_localhost,
    pw_parameters, z2pack_settings, tmpdir
    ):
    """Test a Z2packCalculation with nested restarts."""
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

