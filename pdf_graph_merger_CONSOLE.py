import os
from PyPDF2 import PdfFileReader, PdfFileMerger



# Short concise function definition
def merge_pdf(keyword, out_filename = 'merged_pdf.pdf'):
	files_dir = os.getcwd()
	all_files = list()

	# Add in Figures.
	figures = sorted([f for f in os.listdir(files_dir) if keyword in f and 'pdf' in f])
	all_files.extend(figures)

	# Merge the files
	merger = PdfFileMerger()
	for f in all_files:
		merger.append(PdfFileReader(f), 'rb')
	out_filename = 'MERGED__'+out_filename+'.pdf'
	merger.write(out_filename)


print("\n\n***************************************")
print("*****  Welcome to the pdf merger  *****")
print("***************************************\n\n")
print("Check the current working directory:")
print(os.getcwd())
print("If this is not the working directory, enter the cwd below.\n\n")
cwd = input("CWD:\t\t")
os.chdir(cwd)
print("Input the keyword for searching the files to merge. NB the keyword must be a common substring of the file names.")
keyword = input("Keyword:\t\t")
print("Input the output file name.")
out_filename = input("out_filename:\t\t")
merge_pdf(str(keyword), out_filename = str(out_filename))
open_trigger = input("Would you like to open the output file? (y/n):\t\t")

if open_trigger == 'y': os.startfile('MERGED__'+out_filename+'.pdf')
else: pass

print("\n\n***************************************")
print("***** Thank you for using pdf merger  *****")
print("*****           End of file           *****")
print("***************************************\n\n")


