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

    from cluster import Cluster, Simulation
    import map_plot_parameters as plotpar
    from testing.angular_momentum import angular_momentum_PartType_alignment_matrix
    from save import fof_output as fof
    from save import save

    plotpar.set_defaults_plot()




    # sim = Simulation()
    # z_catalogue = sim.get_redshiftAllowed(dtype = str)
    # from _cluster_retriever import halo_Num, redshift_str2num, redshift_num2str
    #
    # halo = Cluster(clusterID = 3, redshift = 0.)
    # angle_off = angular_momentum_PartType_alignment_matrix(halo)
    # print(angle_off)

    # save.create_file('CELR-eagle')
    # fof.push_FOFapertures('CELR-eagle')
    fof.push_FOFcentre_of_mass('CELR-eagle')
    fof.push_FOFangular_momentum('CELR-eagle')



if __name__ == "__main__":

    import datetime

    if __PROFILE__:
        import cProfile
        cProfile.run('main()')

    else:
        print(__FILE__)
        main()

#TODO
"""
  File "./main.py", line 103, in <module>
    main()
  File "./main.py", line 54, in new_func
    function_result = function(*args, **kwargs)
  File "./main.py", line 88, in main
    fof.push_FOFcentre_of_mass('CELR-eagle')
  File "/cosma/home/dp004/dc-alta2/C-Eagle-analysis/save/fof_output.py", line 90, in push_FOFcentre_of_mass
  File "/cosma/home/dp004/dc-alta2/C-Eagle-analysis/save/save.py", line 92, in create_dataset
    dataset = file_halo_redshift.create_dataset(dataset_name, data = input_data)
  File "/cosma/local/Python/3.6.5/lib/python3.6/site-packages/h5py-2.10.0-py3.6-linux-x86_64.egg/h5py/_hl/group.py", line 139, in create_dataset
    self[name] = dset
  File "/cosma/local/Python/3.6.5/lib/python3.6/site-packages/h5py-2.10.0-py3.6-linux-x86_64.egg/h5py/_hl/group.py", line 373, in __setitem__
    h5o.link(obj.id, self.id, name, lcpl=lcpl, lapl=self._lapl)
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5o.pyx", line 202, in h5py.h5o.link
RuntimeError: Unable to create link (name already exists)
"""