import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import slack

from .__init__ import pprint, rank


pathSave = '/local/scratch/altamura/analysis_results/bahamas_timing/'

def fitFunc(t, a, b):
	return a*t+b

def redshift_str2num(z: str):
	"""
	Converts the redshift of the snapshot from text to numerical,
	in a format compatible with the file names.
	E.g. float z = 2.16 <--- str z = 'z002p160'.
	"""
	z = z.strip('z').replace('p', '.')
	return round(float(z), 3)

def file_benchmarks(redshift: str) -> str:
	timing_filename = pathSave + f"bahamas_timing_{redshift}.txt"
	with open(timing_filename, "a") as benchmarks:
		pprint(f"#{redshift}", file=benchmarks)
	return timing_filename

def display_benchmarks(redshift: str):
	if rank == 0:
		timing_filename = pathSave+f"bahamas_timing_{redshift}.txt"
		plot_filename = pathSave+f"bahamas_timing_{redshift}.png"

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_xlabel('FOF cluster index')
		ax.set_ylabel('Computation time [seconds]')

		lines = np.loadtxt(timing_filename, comments="#", delimiter=",", unpack=False).T
		ax.scatter(lines[0], lines[1], marker = ',', label=f'z = {redshift_str2num(redshift)}')


		# Fit function to benchmarks
		n_fit = []
		dat_fit = []
		for i in range(int(np.max(lines[0]))):
			idx = np.where(lines[0] == i)[0]
			if len(idx) == 1:
				n_fit.append(lines[0,idx][0]+1)
				dat_fit.append(lines[1,idx][0])
			elif len(idx) > 1:
				n_fit.append(np.mean(lines[0,idx])+1)
				dat_fit.append(np.median(lines[1,idx]))

		n_fit = np.log10(np.asarray(n_fit))
		dat_fit = np.log10(np.asarray(dat_fit))
		fitParams, _ = curve_fit(fitFunc, n_fit, dat_fit)
		ax.plot(10**n_fit, 10**fitFunc(n_fit, fitParams[0], fitParams[1]), color='red')
		ax.plot(10 ** n_fit, 10 ** dat_fit, color='red')

		plt.legend()
		plt.savefig(plot_filename, dpi=300)

		# Send files to Slack: init slack client with access token
		print(f"[+] Forwarding {redshift} benchmarks to the `#personal` Slack channel...")
		slack_token = 'xoxp-452271173797-451476014913-1101193540773-57eb7b0d416e8764be6849fdeda52ce8'
		client = slack.WebClient(token=slack_token)
		response = client.files_upload(
				file=plot_filename,
				initial_comment=f"This file was sent upon completion of the plot factory pipeline.\nAttachments: {plot_filename}",
				channels='#personal'
		)


