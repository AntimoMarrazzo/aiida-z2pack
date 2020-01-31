import copy

def prepare_pw(cls, calculation):
	cls.inputs.parameters = copy.deepcopy(cls.inputs.pw_parameters)

	cls._blocked_keywords = cls._blocked_keywords_pw
	cls.inputs.metadata.options.input_filename = getattr(cls, '_DEFAULT_INPUT_' + calculation.upper())
	cls.inputs.metadata.options.output_filename = getattr(cls, '_DEFAULT_OUTPUT_' + calculation.upper())
	cls.inputs.parameters['CONTROL']['calculation'] = calculation
	cls.inputs.settings = cls.inputs.pw_settings

def prepare_scf(cls):
	prepare_pw(cls, 'scf')

def prepare_nscf(cls):
	prepare_pw(cls, 'nscf')