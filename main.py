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
    from import_toolkit.simulation import Simulation
    from import_toolkit.cluster import Cluster

    def check_dirs(self) -> np.ndarray:
        """
        Loops over all listed clusters and redshifts and returns a boolean for what clusters and redshifts
        are present in the simulation archive.
        :return:
        """
        iterator = itertools.product(self.clusterIDAllowed, self.redshiftAllowed)
        check_matrix = np.zeros((len(self.clusterIDAllowed), len(self.redshiftAllowed)), dtype=np.bool)
        for process_n, (halo_id, halo_z) in enumerate(list(iterator)):
            c = Cluster(simulation_name=self.simulation_name,
                        clusterID=halo_id,
                        redshift=halo_z)

            test = c.is_cluster() * c.is_redshift()
            check_matrix[halo_id][self.redshiftAllowed.index(halo_z)] = test

            if not test:
                print(process_n, halo_id, halo_z)
        print(check_matrix)
        return check_matrix

    sim = Simulation(simulation_name='ceagle')
    check_dirs(sim)

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