import os
from PyPDF2 import PdfFileReader, PdfFileMerger

# merge pdf files
"""
def merge_pdf(keyword, out_filename = 'merged_pdf.pdf'):

	files_dir = os.getcwd()

	all_files = list()
	# Add in main text file.
	main_text = [f for f in os.listdir(files_dir) if 'Draft Text' in f and 'pdf' in f]
	all_files.extend(main_text)

	# Add in Figures.
	figures = sorted([f for f in os.listdir(files_dir) if keyword in f and 'pdf' in f])
	all_files.extend(figures)

	# Add in Extended Data
	ext_data = sorted([f for f in os.listdir(files_dir) if 'Extended Data' in f and 'pdf' in f])
	all_files.extend(ext_data)

	# Add in Supplementary Data
	supp_data = sorted([f for f in os.listdir(files_dir) if 'Supplementary Data' in f and 'pdf' in f])
	all_files.extend(supp_data)

	# Merge the files
	merger = PdfFileMerger()
	for f in all_files:
		merger.append(PdfFileReader(f), 'rb')

	merger.write(out_filename)
"""

# Short concise function definition
def merge_pdf(keyword, out_filename = 'myOutputFile', files_dir = os.getcwd()):
	all_files = list()
	figures = sorted([f for f in os.listdir(files_dir) if keyword in f and ('pdf' in f or 'png' in f)])
	all_files.extend(figures)
	merger = PdfFileMerger()
	for f in all_files:
		merger.append(PdfFileReader(f), 'rb')
	out_filename = 'MERGED_'+out_filename+'.pdf'
	merger.write(out_filename)