__FILE__ = """
                               /T /I
                              / |/ | .-~/
                          T\ Y  I  |/  /  _
         /T               | \I  |  I  Y.-~/
        I l   /I       T\ |  |  l  |  T  /
     T\ |  \ Y l  /T   | \I  l   \ `  l Y
 __  | \l   \l  \I l __l  l   \   `  _. |
 \ ~-l  `\   `\  \  \ ~\  \   `. .-~   |                         FILE:    main.py
  \   ~-. "-.  `  \  ^._ ^. "-.  /  \   |                        AUTHOR:  Edo Altamura
.--~-._  ~-  `  _  ~-_.-"-." ._ /._ ." ./                        DATE:    10-11-2019
 >--.  ~-.   ._  ~>-"    "\   7   7   ]
^.___~"--._    ~-{  .-~ .  `\ Y . /    |                         PROJECT: Cluster-EAGLE
 <__ ~"-.  ~       /_/   \   \I  Y   : |                         AIM:     Rotational kSZ
   ^-.__           ~(_/   \   >._:   | l______
       ^--.,___.-~"  /_/   !  `-.~"--l_ /     ~"-.
              (_/ .  ~(   /'     "~"--,Y   -=b-. _)
               (_/ .  \  :           / l      c"~o |
                \ /    `.    .     .^   \_.-~"~--.  )
                 (_/ .   `  /     /       !       )/
                  / / _.   '.   .':      /        '
                  ~(_/ .   /    _  `  .-<_
                    /_/ . ' .-~" `.  / \  \          ,z=.
                    ~( /   '  :   | K   "-.~-.______//
                      "-,.    l   I/ \_    __{--->._(==.
                       //(     \  <    ~"~"     //
                      /' /\     \  \     ,v=.  ((
                    .^. / /\     "  }__ //===-  `
                   / / ' '  "-.,__ {---(==-
                 .^ '       :  T  ~"   ll      
                / .  .  . : | :!        |
               (_/  /   | | j-"          ~^
                 ~-<_(_.^-~"

                                                                 DESCRIPTION:
+-----------------------------------------------------------------------------------------+
|     This file contains the main() function and is used for testing                      |       
|     purposes. It can also be linked to shell arguments for profiling.                   |
|     By setting __PROFILE__ = False you are choosing to run the                          |
|     program with normal outputs, while __PROFILE__ = True will trigger                  |
|     the profiler and display the call stats associated with main().                     |
+-----------------------------------------------------------------------------------------+
"""

__PROFILE__ = False


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def time_func(function):
    # create a new function based on the existing one,
    # that includes the new timing behaviour
    def new_func(*args, **kwargs):
        start = datetime.datetime.now()
        if rank == 0: print('[x] Start: {}'.format(start))

        function_result = function(*args, **kwargs)
        # Calculate the elapsed time and add it to the function
        # attributes.
        end = datetime.datetime.now()
        new_func.elapsed = end - start
        if rank == 0: print('[x] End: {}'.format(end))
        if rank == 0: print('[x] Elapsed: {}'.format(new_func.elapsed))
        return function_result
    return new_func

@time_func
def main():
    import itertools
    import numpy as np
    import h5py as h5
    import sys
    import os
    from import_toolkit.simulation import Simulation
    from import_toolkit.cluster import Cluster
    from import_toolkit._cluster_retriever import redshift_str2num
    from rotvel_correlation import alignment

    SIMULATION = 'bahamas'
    REDSHIFT = 'z003p000'
    N_HALOS = 5

    # -----------------------------------------------------------------------
    # Initialise a sample cluster to get Subfind file metadata
    cluster = Cluster(simulation_name=SIMULATION,
                      clusterID=0,
                      redshift=REDSHIFT,
                      fastbrowsing=True)
    file_GN = cluster.partdata_filePaths()[0]
    del cluster
    with h5.File(file_GN, 'r') as h5file:
        Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][[0,1,4]]

    # Initialize empty arrays across all ranks
    pgn0 = np.empty(Nparticles[0], dtype='i')
    pgn0_0, pgn0_1 = np.array_split(pgn0, 2)
    pgn1 = np.empty(Nparticles[1], dtype='i')
    pgn1_0, pgn1_1 = np.array_split(pgn1, 2)
    pgn4 = np.empty(Nparticles[2], dtype='i')

    # Read the particle groupNumbers with 5 cores (2 allocated for gas and CDM)
    with h5.File(file_GN, 'r') as h5file:
        if rank == 0:
            print(f"[+] RANK {rank}: collecting gas particles groupNumber (0)...")
            pgn0_0[:] = h5file[f'/PartType0/GroupNumber'][:len(pgn0_0)]
        elif rank == 1:
            print(f"[+] RANK {rank}: collecting gas particles groupNumber (1)...")
            pgn0_1[:] = h5file[f'/PartType0/GroupNumber'][len(pgn0_0):]
        elif rank == 2:
            print(f"[+] RANK {rank}: collecting CDM particles groupNumber (0)...")
            pgn1_0[:] = h5file[f'/PartType1/GroupNumber'][:len(pgn1_0)]
        elif rank == 3:
            print(f"[+] RANK {rank}: collecting CDM particles groupNumber (1)...")
            pgn1_1[:] = h5file[f'/PartType1/GroupNumber'][len(pgn1_0):]
        elif rank == 4:
            print(f"[+] RANK {rank}: collecting star particles groupNumber...")
            pgn4[:] = h5file[f'/PartType4/GroupNumber'][:]

    comm.Barrier()
    # Merge arrays for gas and CDM and broadcast to all cores
    if rank == 0:
        comm.Recv([pgn0_1, MPI.INT], source=1, tag=77)
        pgn0[:len(pgn0_0)] = pgn0_0
        pgn0[len(pgn0_0):] = pgn0_1
    elif rank == 1:
        comm.Send([pgn0_1, MPI.INT], dest=0, tag=77)
    elif rank == 2:
        comm.Recv([pgn1_1, MPI.INT], source=3, tag=75)
        pgn1[:len(pgn1_0)] = pgn1_0
        pgn1[len(pgn1_0):] = pgn1_1
    elif rank == 3:
        comm.Send([pgn1_1, MPI.INT], dest=2, tag=75)

    del pgn0_0, pgn0_1, pgn1_0, pgn1_1
    comm.Bcast([pgn0, MPI.INT], root=0)
    comm.Bcast([pgn1, MPI.INT], root=2)
    comm.Bcast([pgn4, MPI.INT], root=4)
    # print(f"Rank: {rank}\tpgn0[10000] = {pgn0[10000]}")
    # print(f"Rank: {rank}\tpgn1[10000] = {pgn1[10000]}")
    # print(f"Rank: {rank}\tpgn4[10000] = {pgn4[10000]}")
    # comm.Barrier()
    # if rank == 0:
    #     print('pgn0 == 1', np.where(pgn0 == 1)[0])
    #     print('pgn1 == 1', np.where(pgn1 == 1)[0])
    #     print('pgn4 == 1', np.where(pgn4 == 1)[0])

    comm.Barrier()
    # Initialise the allocation for cluster reports
    clusterID_pool = np.arange(N_HALOS)
    clusterID_pool_split = np.array_split(clusterID_pool, size)
    if rank == clusterID_pool_split.index(rank):
        for i in clusterID_pool_split[rank]:
            print(f"[+] RANK {rank}: initializing report {SIMULATION:>10s} {i:<5d} {REDSHIFT:s}...")
            alignment.save_report(i, REDSHIFT, glob=[pgn0, pgn1, pgn4])





if __name__ == "__main__":

    import datetime
    import argparse

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('-p',
                           '--profile',
                           action='store_true',
                           help='Triggers the cProfile for the main() function.')
    args = my_parser.parse_args()


    if vars(args)['profile']:
        import cProfile
        cProfile.run('main()')

    else:
        # print(__FILE__)
        main()