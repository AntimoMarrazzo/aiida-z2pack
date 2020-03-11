"""Tests for the Z2packParser."""
from __future__ import absolute_import
import os
from aiida import orm
from aiida.common import LinkType
# from aiida.plugins.entry_point import format_entry_point_string


def test_z2pack_default(
    aiida_profile, fixture_localhost,
    generate_parser, generate_calc_job_node,
    data_regression
    ):
    """Test the parsing of a successful calculation."""
    name = 'default'
    entry_point_calc_job = 'z2pack.z2pack'
    entry_point_parser = 'z2pack.z2pack'

    parser = generate_parser(entry_point_parser)
    node   = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name)

    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert not orm.Log.objects.get_logs_for(node)

    data_regression.check({
        'output_parameters': results['output_parameters'].get_dict()
        })

def test_z2pack_failed_missing(
    aiida_profile, fixture_localhost,
    generate_parser, generate_calc_job_node,
    data_regression
    ):
    """Test the parsing of calculation finished with no output files."""
    name = 'failed_missing'
    entry_point_calc_job = 'z2pack.z2pack'
    entry_point_parser = 'z2pack.z2pack'

    parser = generate_parser(entry_point_parser)
    node   = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name)

    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status, node.process_class.exit_codes.ERROR_OUTPUT_FILES.status
    assert orm.Log.objects.get_logs_for(node)

def test_z2pack_failed_missing_results(
    aiida_profile, fixture_localhost,
    generate_parser, generate_calc_job_node,
    data_regression
    ):
    """Test the parsing of calculation finished with no result file."""
    name = 'failed_missing_results'
    entry_point_calc_job = 'z2pack.z2pack'
    entry_point_parser = 'z2pack.z2pack'

    parser = generate_parser(entry_point_parser)
    node   = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name)

    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status, node.process_class.exit_codes.ERROR_MISSING_RESULTS_FILE.status
    assert orm.Log.objects.get_logs_for(node)

def test_z2pack_failed_missing_save(
    aiida_profile, fixture_localhost,
    generate_parser, generate_calc_job_node,
    data_regression
    ):
    """Test the parsing of calculation finished with no save file."""
    name = 'failed_missing_save'
    entry_point_calc_job = 'z2pack.z2pack'
    entry_point_parser = 'z2pack.z2pack'

    parser = generate_parser(entry_point_parser)
    node   = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name)

    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status, node.process_class.exit_codes.ERROR_MISSING_SAVE_FILE.status
    assert orm.Log.objects.get_logs_for(node)

def test_z2pack_failed_pwcrash(
    aiida_profile, fixture_localhost,
    generate_parser, generate_calc_job_node,
    data_regression
    ):
    """Test the parsing of calculation finished with a pw CRASH error file."""
    name = 'failed_pwcrash'
    entry_point_calc_job = 'z2pack.z2pack'
    entry_point_parser = 'z2pack.z2pack'

    parser = generate_parser(entry_point_parser)
    node   = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name)

    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status, node.process_class.exit_codes.ERROR_PW_CRASH.status
    assert orm.Log.objects.get_logs_for(node)

def test_z2pack_failed_w90crash(
    aiida_profile, fixture_localhost,
    generate_parser, generate_calc_job_node,
    data_regression
    ):
    """Test the parsing of calculation finished with a w90 *.werr error file."""
    name = 'failed_w90crash'
    entry_point_calc_job = 'z2pack.z2pack'
    entry_point_parser = 'z2pack.z2pack'

    parser = generate_parser(entry_point_parser)
    node   = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name)

    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_failed, calcfunction.exit_status
    assert calcfunction.exit_status, node.process_class.exit_codes.ERROR_W90_CRASH.status
    assert orm.Log.objects.get_logs_for(node)

