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
import os.path
import datetime
import argparse
from mpi4py import MPI
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

@time_func
def main():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    from bahamas import main as bah
    from macsis import main as mac
    # bah.main()
    mac.main()

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
        main()