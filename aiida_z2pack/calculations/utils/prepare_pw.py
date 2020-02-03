import copy
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.calculations import _uppercase_dict

PwCalculation = CalculationFactory('quantumespresso.pw')
PwCalculation._use_kpoints = False

def _prepare_pw(cls, folder, calculation):
	parameters = copy.deepcopy(cls.inputs.pw_parameters)

	parameters['CONTROL']['calculation'] = calculation

	arguments = [
		parameters,
		_uppercase_dict(cls.inputs.pw_settings.get_dict(), dict_name='settings'),
		cls.inputs.pseudos,
		cls.inputs.structure,
		]

	input_filecontent, _ = PwCalculation._generate_PWCPinputdata(*arguments)

	input_filename = getattr(cls, '_INPUT_PW_{}_FILE'.format(calculation.upper()))

	with folder.open(input_filename, 'w') as infile:
		infile.write(input_filecontent)

def prepare_scf(cls, folder):
	PwCalculation._OUTPUT_SUBFOLDER = cls._OUTPUT_SUBFOLDER
	_prepare_pw(cls, folder, 'scf')

def prepare_nscf(cls, folder):
	PwCalculation._OUTPUT_SUBFOLDER = cls._SCFTMP_SUBFOLDER
	_prepare_pw(cls, folder, 'nscf')