from import_toolkit import *


def redshift_num2str(z: float):
	"""
	Converts the redshift of the snapshot from numerical to
	text, in a format compatible with the file names.
	E.g. float z = 2.16 ---> str z = 'z002p160'.
	"""
	z = round(z, 3)
	integer_z, decimal_z = str(z).split('.')
	integer_z = int(integer_z)
	decimal_z = int(decimal_z)
	return f"z{integer_z:0>3d}p{decimal_z:0<3d}"