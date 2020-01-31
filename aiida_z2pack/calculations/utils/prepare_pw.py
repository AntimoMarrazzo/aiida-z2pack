import copy
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.calculations import _uppercase_dict

PwCalculation = CalculationFactory('quantumespresso.pw')
PwCalculation._use_kpoints = False

def prepare_pw(cls, folder, calculation):
	parameters = copy.deepcopy(cls.inputs.pw_parameters)

	parameters['CONTROL']['calculation'] = calculation

	arguments = [
		parameters,
		_uppercase_dict(cls.inputs.settings.get_dict(), dict_name='settings'),
		cls.inputs.pseudos,
		cls.inputs.structure,
		]

	input_filecontent, _ = PwCalculation._generate_PWCPinputdata(*arguments)

	input_filename = getattr(cls, '_INPUT_PW_' + calculation.upper() + '_FILE')

	with folder.open(input_filename, 'w') as infile:
		infile.write(input_filecontent)

def prepare_scf(cls, folder):
	prepare_pw(cls, folder, 'scf')

def prepare_nscf(cls, folder):
	prepare_pw(cls, folder, 'nscf')