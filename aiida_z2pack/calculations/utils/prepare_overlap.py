from aiida_quantumespresso.calculations.namelists import NamelistsCalculation

def prepare_overlap(cls,folder):
	parameters = cls.inputs.overlap_parameters.get_dict()
	blocked    = cls._blocked_keywords_overlap
	namelists  = NamelistsCalculation._default_namelists

	content = NamelistsCalculation.generate_input_file(parameters, namelists_toprint=namelists, blocked=blocked)

	input_filename = cls._INPUT_OVERLAP_FILE

	with folder.open(input_filename, 'w') as infile:
		infile.write(content)
