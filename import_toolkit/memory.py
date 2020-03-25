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
size = comm.Get_size()
rank = comm.Get_rank()


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
            from import_toolkit.cluster import Cluster
            cluster = Cluster(  simulation_name = 'celr_e',
                                 clusterID = 0,
                                 redshift = 'z000p000',
                                 comovingframe = False,
                                 requires = self.data_required)
            scheduler = SchedulerMPI.from_cluster(cluster)
            scheduler.info()
            print('[ UNIT TEST ]\t==> ', self.quick_build() == scheduler)

        def scatter_data(self, data):
            import numpy as np

            if rank == 0:
                test = data
                print(test)
                outputData = np.zeros(len(test))       # Create output array of same size
                split = np.array_split(test, size, axis=0)  # Split input array by the number of available cores
                split_sizes = [len(split[i]) for i in range(0, len(split), 1)]
                print(split_sizes)

                split_sizes_input = split_sizes * len(test)
                displacements_input = np.insert(np.cumsum(split_sizes_input), 0, 0)[0:-1]

                split_sizes_output = split_sizes * len(test)
                displacements_output = np.insert(np.cumsum(split_sizes_output), 0, 0)[0:-1]

                print("Input data split into vectors of sizes %s" % split_sizes_input)
                print("Input data split with displacements of %s" % displacements_input)

            else:
                # Create variables on other cores
                split_sizes_input = None
                displacements_input = None
                split_sizes_output = None
                displacements_output = None
                split = None
                test = None
                outputData = None

            split = comm.bcast(split, root=0)  # Broadcast split array to other cores
            split_sizes = comm.bcast(split_sizes_input, root=0)
            displacements = comm.bcast(displacements_input, root=0)
            split_sizes_output = comm.bcast(split_sizes_output, root=0)
            displacements_output = comm.bcast(displacements_output, root=0)

            output_chunk = np.zeros(np.shape(split[rank]))  # Create array to receive subset of data on each core, where rank specifies the core
            print("Rank %d with output_chunk shape %s" % (rank, output_chunk.shape))
            comm.Scatterv([test, split_sizes_input, displacements_input, MPI.DOUBLE], output_chunk, root=0)

            output = np.zeros([len(output_chunk), 512])  # Create output array on each core

            for i in range(0, np.shape(output_chunk)[0], 1):
                output[i, 0:len(test)] = output_chunk[i]


            print("Output shape %s for rank %d" % (output.shape, rank))

            comm.Barrier()

            comm.Gatherv(output, [outputData, split_sizes_output, displacements_output, MPI.DOUBLE], root=0)

            if rank == 0:
                outputData = outputData[0:len(test), :]
                print("Final data shape %s" % (outputData.shape,))

                print(outputData)



    test = TEST()
    # test.from_dictionary()
    # test.from_cluster()
    # import numpy as np
    # data = np.linspace(0, 100, 101)
    # test.scatter_data(data)

    if rank == 0:
        data = [i for i in range(8)]
        # dividing data into chunks
        chunks = [[] for _ in range(size)]
        for i, chunk in enumerate(data):
            chunks[i % size].append(chunk)
    else:
        data = None
        chunks = None
    data = comm.scatter(chunks, root=0)

    print("%s: %s" % (rank, data))



