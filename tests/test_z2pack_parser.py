import os
from aiida import orm
from aiida.common import LinkType
# from aiida.plugins.entry_point import format_entry_point_string


def test_z2pack_parser(
    aiida_profile, fixture_localhost,
    generate_parser,
    data_regression
    ):
    target = './tests/fixtures/parser'
    target = os.path.abspath(target)
    parser = generate_parser('z2pack.z2pack')
    # entry_point = format_entry_point_string('aiida.calculations', 'z2pack.z2pack')

    # node = generate_calc_job_node('z2pack.z2pack', fixture_localhost, name)
    node = orm.CalcJobNode(computer=fixture_localhost, process_type='aiida.calculations:z2pack.z2pack')
    node.store()

    retrieved = orm.FolderData()
    retrieved.put_object_from_tree(os.path.join(target))
    retrieved.add_incoming(node, link_type=LinkType.CREATE, link_label='retrieved')
    retrieved.store()

    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    data_regression.check({
        'output_parameters': results['output_parameters'].get_dict()
        })
