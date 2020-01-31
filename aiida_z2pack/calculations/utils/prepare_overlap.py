# from aiida import orm

def prepare_overlap(cls):
	# Filter blocked keyword without default value that would cause an exception in a NamelistsCalculation
	# old_bk = cls._blocked_keywords
	# new_bk = []
	# for t in old_bk:
	#     if len(t) == 3:
	#         new_bk.append(t)
	# cls._blocked_keywords = new_bk

	cls._blocked_keywords = cls._blocked_keywords_overlap
	cls.inputs.metadata.options.input_filename = cls._INPUT_OVERLAP_FILE_NAME
	cls.inputs.metadata.options.output_filename = cls._OUTPUT_OVERLAP_FILE_NAME
	cls.inputs.parameters = cls.inputs.overlap_parameters
	cls.inputs.settings = cls.inputs.overlap_settings

	# return old_bk

# &inputpp
#   outdir='../tmp'
#   prefix='Tinene'
#   seedname='Tinene'
#   wan_mode='standalone'
#   write_amn =.false.
#   write_mmn =.true.
#   regular_mesh = .false.
# /
