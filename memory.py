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
from mpi4py import MPI
import networkx

comm = MPI.COMM_WORLD
size = comm.get_size()
rank = comm.get_rank()


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


def dict_key_finder(dictionary : dict, search: str) -> list:
    search_output = []
    for key in list(dictionary.keys()):
        if search in key:
            search_output.append(key)
    return search_output

def dict_key_exclusionfinder(dictionary : dict, search: str) -> list:
    search_output = []
    for key in list(dictionary.keys()):
        if search not in key:
            search_output.append(key)
    return search_output

class SchedulerMPI:

    def __init__(self, requires: dict):
        self.requires = requires
        self.architecture = {}
        self.generate_arch_clusterMPI()

    def __eq__(self, other):
        """
        Overrides the default implementation
        """

        if isinstance(other, SchedulerMPI):
            condition = (self.requires == other.requires) and (self.architecture == other.architecture)
            return condition
        return False

    @classmethod
    def from_cluster(cls, cluster):
        schedule = cls(cluster.requires)
        return schedule

    @classmethod
    def from_dictionary(cls, requires: dict):
        schedule = cls(requires)
        return schedule

    def generate_arch_clusterMPI(self) -> None:
        print('[ SchedulerMPI ]\t==> Generating MPI architecture...')
        core_counter = 0
        self.architecture[core_counter] = 'master'

        # Loop over particle type containers
        for key_partType in dict_key_finder(self.requires, 'partType'):
            allocation_partType = self.requires[key_partType]

            # Loop over partType fields
            for field_partType in allocation_partType:
                core_counter += 1
                self.architecture[core_counter] =  key_partType + '_' + field_partType

        # Finally, allocate cores to any other non-partType key leftover
        for other_key in dict_key_exclusionfinder(self.requires, 'partType'):
            core_counter += 1
            self.architecture[core_counter] = other_key

        if len(self.architecture) > comm.Get_size():
            print('[ SchedulerMPI ]\t==> Warning: requested more cores than there are available. Requested {0}, '
                  'available {1}.'.format(len(self.architecture), comm.Get_size()))

        return

    def requires_as_graph(self) -> networkx.DiGraph:
        g = networkx.DiGraph()
        g.add_edges_from(self.requires.keys())
        for k, v in self.requires.items():
            g.add_edges_from(([(k, t) for t in v]))
        return g

    def architecture_as_graph(self) -> networkx.DiGraph:
        g = networkx.DiGraph()
        g.add_edges_from(self.architecture.keys())
        for k, v in self.architecture.items():
            g.add_edges_from(([(k, t) for t in v]))
        return g


    def info(self, verbose: bool = False) -> None:
        print('------------------------------------------------------------------')
        print('                           CLASS INFO                             ')
        if not verbose:
            print("\n.......................SchedulerMPI.requires.......................")
            for x in self.requires:
                print(x)
                for y in self.requires[x]:
                    print('\t', y)
            print("\n.....................SchedulerMPI.architecture......................")
            for x in self.architecture:
                print('core ', x, ':', self.architecture[x])

        else:
            for attr in dir(self):
                print("obj.%s = %r" % (attr, getattr(self, attr)))
        print('\n------------------------------------------------------------------')
        return




if __name__ == '__main__':

    import inspect

    class TEST:

        data_required = {'partType0': ['coordinates', 'velocities', 'temperature', 'sphkernel'],
                         'partType1': ['coordinates', 'velocities']}

        def quick_build(self):
            scheduler = SchedulerMPI(self.data_required)
            return scheduler

        def from_dictionary(self):
            print(inspect.stack()[0][3])
            scheduler = SchedulerMPI.from_dictionary(self.data_required)
            scheduler.info()
            print('[ UNIT TEST ]\t==> ', self.quick_build() == scheduler)

        def from_cluster(self):
            print(inspect.stack()[0][3])
            from cluster import Cluster
            cluster = Cluster(  simulation_name = 'celr_e',
                                 clusterID = 0,
                                 redshift = 'z000p000',
                                 comovingframe = False,
                                 requires = self.data_required)
            scheduler = SchedulerMPI.from_cluster(cluster)
            scheduler.info()
            print('[ UNIT TEST ]\t==> ', self.quick_build() == scheduler)

        def scatter(self):
            import numpy
            if rank == 0:
                x = numpy.linspace(0, 100, 11)
            else:
                x = None

            if rank == 2:
                xlocal = numpy.zeros(9)
            else:
                xlocal = numpy.zeros(1)

            if rank == 0:
                print("Scatter")
            comm.Scatterv([x, (1, 1, 9), (0, 1, 2), MPI.DOUBLE], xlocal)
            print("process " + str(rank) + " has " + str(xlocal))

            comm.Barrier()
            if rank == 0:
                print("Gather")
                xGathered = numpy.zeros(11)
            else:
                xGathered = None
            comm.Gatherv(xlocal, [xGathered, (1, 1, 9), (0, 1, 2), MPI.DOUBLE])
            print("process " + str(rank) + " has " + str(xGathered))



    test = TEST()
    test.from_dictionary()
    # test.from_cluster()
    test.scatter()




