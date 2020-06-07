import numpy as np
from matplotlib import pyplot as plt
import slack

from .__init__ import pprint, rank


pathSave = '/local/scratch/altamura/analysis_results/bahamas_timing/'

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
		# ax.set_xscale("log")
		# ax.set_yscale("log")
		ax.set_xlabel('FOF cluster index')
		ax.set_ylabel('Computation time [seconds]')

		lines = np.loadtxt(timing_filename, comments="#", delimiter=",", unpack=False).T
		ax.scatter(lines[0], lines[1], marker = '.', label=f'{redshift}')

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
