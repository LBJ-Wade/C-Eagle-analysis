"""
------------------------------------------------------------------
FILE:   memory.py
AUTHOR: Edo Altamura
DATE:   12-11-2019
------------------------------------------------------------------
This file provides methods for memory management.
Future implementations:
    - dynamic memory allocation
    - automated performance optimization
    - MPI meta-methods and multi-threading
-------------------------------------------------------------------
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from cluster import Cluster

def free_memory(var_list, invert=False):
    """
    Function for freeing memory dynamically.
    invert allows to delete all local variables that are NOT in var_list.
    """
    if not invert:
        for name in var_list:
            if not name.startswith('_') and name in dir():
                del globals()[name]
    if invert:
        for name in dir():
            if name in var_list and not name.startswith('_'):
                del globals()[name]


def delegate_independent_nodes():
    pass


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)


    Note: This is for Python 3; see the comments for details on using this in Python 2.

    Sample Usage
    import time

    # A List of Items
    items = list(range(0, 57))
    l = len(items)

    # Initial call to print 0% progress
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i, item in enumerate(items):
        # Do stuff...
        time.sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)


    Sample Output:

    Progress: |█████████████████████████████████████████████-----| 90.0% Complete
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

class SchedulerMPI:

    def __init__(self):
        self.architecture = {}
        self.requires = {}

    @classmethod
    def from_cluster(cls, cluster: Cluster):
        schedule = cls()
        cls.requires = cluster.requires
        cls.generate_arch_clusterMPI()
        return schedule

    @classmethod
    def from_dictionary(cls, requires: dict):
        schedule = cls()
        cls.requires = requires
        cls.generate_arch_clusterMPI()
        return schedule

    @staticmethod
    def dict_key_finder(dictionary : dict, search: str) -> list:
        search_output = []
        for key in list(dictionary.keys()):
            if search in key:
                search_output.append(key)
        return search_output

    def generate_arch_clusterMPI(self) -> None:

        core_counter = 0
        self.architecture[core_counter] = 'master'

        # Loop over particle type containers
        for key_partType in self.dict_key_finder(self.requires, 'partType'):
            allocation_partType = self.requires[key_partType]

            # Loop over fields
            for field_partType in allocation_partType:
                core_counter += 1
                self.architecture[core_counter] =  key_partType + '_' + field_partType

        return

    def info(self) -> None:
        for attr in dir(self):
            print("obj.%s = %r" % (attr, getattr(self, attr)))

        return



if __name__ == '__main__':
    data_required = {'partType0' : ['coordinates', 'velocities', 'temperature', 'sphkernel'],
                     'partType1' : ['coordinates', 'velocities']}
    scheduler = SchedulerMPI.from_dictionary(data_required)
    scheduler.info()





