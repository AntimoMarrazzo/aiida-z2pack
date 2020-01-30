from aiida import load_profile
from aiida.orm.utils import load_node
from aiida.manage.manager import get_manager
from aiida.plugins import CalculationFactory
from aiida.engine.utils import instantiate_process
from aiida.orm.nodes.data.upf import get_pseudos_from_structure
from aiida_quantumespresso.utils.resources import get_default_options

load_profile()

manager = get_manager()
runner  = manager.get_runner()

c = CalculationFactory('z2pack')

structure = load_node(1786)
pseudos   = get_pseudos_from_structure(structure, 'SSSP')

# print(structure)
# print(pseudos)

builder = c.get_builder()
# builder.structure = structure
# builder.pseudos = pseudos

print(type(builder))
# print(builder)
# builder.metadata.options.resources = {'num_machines': 1}
# print(builder)

print(builder.process_class)

inputs = {
	'metadata':{
		'options': get_default_options()
		}
	}
process = instantiate_process(runner, c, **inputs)
# c(
# 	inputs={
# 		'metadata.options.resources':{'num_machines': 1}
# 		}
# 	)
