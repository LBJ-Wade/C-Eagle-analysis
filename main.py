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

def time_func(function):
    # create a new function based on the existing one,
    # that includes the new timing behaviour
    def new_func(*args, **kwargs):
        start = datetime.datetime.now()
        print('Start: {}'.format(start))

        function_result = function(*args, **kwargs)
        # Calculate the elapsed time and add it to the function
        # attributes.
        end = datetime.datetime.now()
        new_func.elapsed = end - start
        print('End: {}'.format(end))
        print('Elapsed: {}'.format(new_func.elapsed))
        return function_result
    return new_func

@time_func
def main():
    import itertools
    import numpy as np
    import h5py
    import sys
    import os
    from import_toolkit.simulation import Simulation
    from import_toolkit.cluster import Cluster
    from import_toolkit._cluster_retriever import redshift_str2num
    from rotvel_correlation import alignment

    def bahamas_mass_cut():
        cluster = Cluster(simulation_name='bahamas',
                        clusterID=0,
                        redshift='z000p000',
                        comovingframe=False,
                        fastbrowsing=True)
        n_largeM = 0
        n_total = 0
        gn_tot = np.zeros(0, dtype=np.int)
        N_halos = 0
        for counter, file in enumerate(cluster.groups_filePaths()):
            print(f"[+] Analysing eagle_subfind_tab file {counter}")
            with h5py.File(file, 'r') as group_file:
                m500 = group_file['/FOF/Group_M_Crit500'][:] * 10 ** 10
                n_total += len(m500)
                m_filter = np.where(m500 > 10 ** 13)[0] + N_halos
                gn_tot = np.append(gn_tot, m_filter)
                n_largeM += len(m_filter)
                N_halos = group_file['Header'].attrs['Ngroups']

        print('n_largeM:', n_largeM)
        print('n_total:', n_total)
        print('N_halos:', N_halos)
        print('gn_tot:', gn_tot)

        # def call_cluster(i):
    #     data_required = {
    #             'partType0': ['groupnumber', 'subgroupnumber', 'mass', 'coordinates', 'velocity', 'temperature', 'sphdensity'],
    #             'partType1': ['groupnumber', 'subgroupnumber', 'mass', 'coordinates', 'velocity'],
    #             'partType4': ['groupnumber', 'subgroupnumber', 'mass', 'coordinates', 'velocity']
    #     }
    #
    #     cluster = Cluster(simulation_name='bahamas',
    #                       clusterID=i,
    #                       redshift='z003p000',
    #                       comovingframe=False,
    #                       requires=data_required)
    #
    #     apertures = cluster.generate_apertures()
    #     master_dict = {}
    #     for i,r_a in enumerate(apertures):
    #         halo_output = {
    #                 **cluster.group_fofinfo(aperture_radius=cluster.r200),
    #                 **cluster.group_dynamics(aperture_radius=cluster.r200),
    #                 **cluster.group_morphology(aperture_radius=cluster.r200)
    #         }
    #         alignment_dict = alignment.group_alignment(halo_output)
    #         halo_output = {
    #                 **halo_output,
    #                 **alignment_dict
    #         }
    #         master_dict[f'aperture{i:02d}'] = halo_output
    #         del halo_output, alignment_dict
    #     print(f"\nmaster_dict: \tOutput size {sys.getsizeof(master_dict) / 1024:2.0f} kB")
    #     for key in master_dict: print(key)
    #
    # for i in range(12):
    #     call_cluster(i)


    # def check_dirs(self) -> np.ndarray:
    #     """
    #     Loops over all listed clusters and redshifts and returns a boolean for what clusters and redshifts
    #     are present in the simulation archive.
    #     :return:
    #     """
    #     iterator = itertools.product(self.clusterIDAllowed, self.redshiftAllowed)
    #     check_matrix = np.zeros((len(self.clusterIDAllowed), len(self.redshiftAllowed)), dtype=np.bool)
    #     for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
    #         c = Cluster(simulation_name=self.simulation_name,
    #                     clusterID=halo_id,
    #                     redshift=halo_z)
    #
    #         redshift_threshold = redshift_str2num(halo_z) < 1.8
    #         test = c.is_cluster() * c.is_redshift() * redshift_threshold
    #         check_matrix[halo_id][self.redshiftAllowed.index(halo_z)] = test
    #         print(process_n, halo_id, halo_z, test)
    #
    #     np.save(f'import_toolkit/{c.simulation_name}_sample_completeness.npy', check_matrix)
    #     print(len(np.where(check_matrix == True)[0])/np.product(check_matrix.shape)*100)
    #     for i in check_matrix:
    #         print([int(j) for j in i])
    #     return check_matrix
    #
    # s = Simulation(simulation_name='macsis')
    # for n in s.clusterIDAllowed:
    #     path1 = os.path.join(s.pathSave, s.simulation_name,
    #                         f'halo{s.halo_Num(n)}', f'halo{s.halo_Num(n)}_z000p024')
    #     path2 = os.path.join(s.pathSave, s.simulation_name,
    #                          f'halo{s.halo_Num(n)}', f'halo{s.halo_Num(n)}_z000p240')
    #     os.rename(path1, path2)


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