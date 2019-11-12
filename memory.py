"""


"""

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
