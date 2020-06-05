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

import sys
import os
import itertools
import argparse
from mpi4py import MPI
import numpy as np
import h5py as h5
import datetime
from scipy.sparse import csr_matrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

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

def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

def split(NProcs,MyRank,nfiles):
    nfiles=int(nfiles)
    nf=int(nfiles/NProcs)
    rmd=nfiles % NProcs
    st=MyRank*nf
    fh=(MyRank+1)*nf
    if MyRank < rmd:
        st+=MyRank
        fh+=(MyRank+1)
    else:
        st+=rmd
        fh+=rmd
    return st,fh

def commune(comm,NProcs, MyRank,data):
    tmp=np.zeros(NProcs,dtype=np.int)
    tmp[MyRank]=len(data)
    cnts=np.zeros(NProcs,dtype=np.int)
    comm.Allreduce([tmp,MPI.INT],[cnts,MPI.INT],op=MPI.SUM)
    del tmp
    dspl=np.zeros(NProcs,dtype=np.int)
    i=0
    for j in range(0,NProcs,1):
        dspl[j]=i
        i+=cnts[j]
    rslt=np.zeros(i,dtype=data.dtype)
    comm.Allgatherv([data,cnts[MyRank]],[rslt,cnts,dspl,MPI._typedict[data.dtype.char]])
    del data,cnts,dspl
    return rslt

def gather_and_isolate(comm,NProcs,MyRank,data,toRank):
    tmp=np.zeros(NProcs,dtype=np.int)
    tmp[MyRank]=len(data)
    cnts=np.zeros(NProcs,dtype=np.int)
    comm.Allreduce([tmp,MPI.INT],[cnts,MPI.INT],op=MPI.SUM)
    del tmp
    dspl=np.zeros(NProcs,dtype=np.int)
    i=0
    for j in range(0,NProcs,1):
        dspl[j]=i
        i+=cnts[j]
    rslt=np.zeros(i,dtype=data.dtype)
    comm.Allgatherv([data,cnts[MyRank]],[rslt,cnts,dspl,MPI._typedict[data.dtype.char]])
    del data,cnts,dspl
    if MyRank != toRank:
        # rslt = np.zeros_like(rslt)
        rslt = None
    return rslt

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))

def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]

@time_func
def main():

    # from import_toolkit.simulation import Simulation
    from import_toolkit.cluster import Cluster
    # from import_toolkit._cluster_retriever import redshift_str2num
    # from rotvel_correlation import alignment

    SIMULATION = 'bahamas'
    REDSHIFT = 'z003p000'
    N_HALOS = 12

    # -----------------------------------------------------------------------


    # Initialise a sample cluster to get Subfind file metadata
    cluster = Cluster(simulation_name=SIMULATION,
                      clusterID=0,
                      redshift=REDSHIFT,
                      fastbrowsing=True)
    cluster_pathSave = cluster.pathSave
    halo_num_catalogue_contiguous = cluster.halo_num_catalogue_contiguous
    file_GN = cluster.partdata_filePaths()[0]
    del cluster

    groupnumber_csrm = []
    with h5.File(file_GN, 'r') as h5file:
        Nparticles = h5file['Header'].attrs['NumPart_ThisFile'][[0,1,4]]
        for i, n_particles in enumerate(Nparticles):
            pprint(f"[+] Collecting gas particles GroupNumber {i}...")
            st, fh = split(nproc, rank, n_particles)
            pgn_slice = h5file[f'/PartType0/GroupNumber'][st:fh]

            # Clip out negative values and exceeding values
            pprint(f"[+] Computing CSR indexing matrix {i}...")
            pgn_slice = np.clip(pgn_slice, 0, np.max(halo_num_catalogue_contiguous)+2)
            csrm = get_indices_sparse(pgn_slice)
            groupnumber_csrm.append(csrm)

    # Initialise the allocation for cluster reports
    clusterID_pool = np.arange(N_HALOS)
    for i in clusterID_pool:
        if rank == i%nproc:
            print(f"[+] RANK {rank} | Initializing partGN generation... {SIMULATION:>10s} {i:<5d} {REDSHIFT:s}")
            idx = groupnumber_csrm[:][fof_id][0]
            fof_id = halo_num_catalogue_contiguous[i]+1
            for partType in range(3):
                gn = gather_and_isolate(comm, nproc, rank, idx[partType] , toRank=rank)
                print(partType, gn)


            # print(f"[+] RANK {rank}: initializing report... {SIMULATION:>10s} {i:<5d} {REDSHIFT:s}")
            # alignment.save_report(i, REDSHIFT, glob=[pgn0, pgn1, pgn4])
    comm.Barrier()




if __name__ == "__main__":
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