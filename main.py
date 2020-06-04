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



    cluster = Cluster(simulation_name='bahamas',
                      clusterID=0,
                      redshift='z003p000',
                      fastbrowsing=True)
    file_GN = cluster.partdata_filePaths()[0]
    del cluster
    with h5.File(file_GN, 'r') as h5file:
        Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][[0,1,4]]
    pgn0 = np.empty(3, dtype='i')
    pgn1 = np.empty(3, dtype='i')
    pgn4 = np.empty(3, dtype='i')
    # pgn0 = np.empty(Nparticles[0], dtype='i')
    # pgn1 = np.empty(Nparticles[1], dtype='i')
    # pgn4 = np.empty(Nparticles[2], dtype='i')

    with h5.File(file_GN, 'r') as h5file:
        if rank == 0:
            print(f"[+] RANK {rank}: collecting gas particles groupNumber...")
            pgn0[:] = h5file[f'/PartType0/GroupNumber'][:3]
        elif rank == 1:
            print(f"[+] RANK {rank}: collecting CDM particles groupNumber...")
            pgn1[:] = h5file[f'/PartType1/GroupNumber'][:3]
        elif rank == 2:
            print(f"[+] RANK {rank}: collecting stars particles groupNumber...")
            pgn4[:] = h5file[f'/PartType4/GroupNumber'][:3]

    comm.Bcast([pgn0, MPI.INT], root=0)
    comm.Bcast([pgn1, MPI.INT], root=1)
    comm.Bcast([pgn4, MPI.INT], root=2)
    print("Rank: ", rank, ". pgn0 is:", pgn0)
    print("Rank: ", rank, ". pgn1 is:", pgn1)
    print("Rank: ", rank, ". pgn4 is:", pgn4)

    for i in range(3):
        if rank == i%size:
            alignment.save_report(i, 'z000p000', glob=[pgn0, pgn1, pgn4])

    comm.Barrier()



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