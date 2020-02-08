import os
import copy
from aiida.plugins import CalculationFactory
from aiida_quantumespresso.calculations import _uppercase_dict

PwCalculation = CalculationFactory('quantumespresso.pw')

class Temp(PwCalculation):
	_use_kpoints = False

def _prepare_pw(cls, folder, calculation):
	parameters = copy.deepcopy(cls.inputs.pw_parameters)

	parameters['CONTROL']['calculation'] = calculation
	parameters['SYSTEM']['nosym'] = True

	arguments = [
		parameters,
		_uppercase_dict(cls.inputs.pw_settings.get_dict(), dict_name='settings'),
		cls.inputs.pseudos,
		cls.inputs.structure,
		]

	input_filecontent, _ = Temp._generate_PWCPinputdata(*arguments)

	input_filename = getattr(cls, '_INPUT_PW_{}_FILE'.format(calculation.upper()))

	with folder.open(input_filename, 'w') as infile:
		infile.write(input_filecontent)

# def prepare_scf(cls, folder):
# 	Temp._OUTPUT_SUBFOLDER = cls._OUTPUT_SUBFOLDER
# 	_prepare_pw(cls, folder, 'scf')

def prepare_nscf(cls, folder):
	# Temp._OUTPUT_SUBFOLDER = os.path.join(cls._REVERSE_BUILD_SUBFOLDER, cls._OUTPUT_SUBFOLDER)
	# Temp._PSEUDO_SUBFOLDER = os.path.join(cls._REVERSE_BUILD_SUBFOLDER, cls._PSEUDO_SUBFOLDER)
	_prepare_pw(cls, folder, 'nscf')