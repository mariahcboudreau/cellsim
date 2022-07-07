'''
Defines functions for making the population.
'''

# %% Imports
from re import U
import numpy as np  # Needed for a few things not provided by pl
import sciris as sc
from . import utils as cellUtil
from . import misc as cellMisc
# from . import base as cvb
from . import default as cellDef
from . import parameters as cellPar
from . import cell_mass as cellMass


# # Specify all externally visible functions this file defines
# __all__ = ['make_people', 'make_randpop', 'make_random_contacts']


def make_cells(sim, popdict=None, reset=False, verbose=None, dispersion=None, microstructure=None, **kwargs):
    '''
    Make the cells for the simulation.

    Usually called via ``sim.initialize()``.

    Args:
        sim      (Sim)  : the simulation object; population parameters are taken from the sim object
        popdict  (any)  : if supplied, use this population dictionary instead of generating a new one; can be a dict or People object
        reset    (bool) : whether to force population creation even if self.popdict/self.people exists
        verbose  (bool) : level of detail to print

    Returns:
        people (CellMass): Cell
    '''

    # Set inputs and defaults
    pop_size = int(sim['pop_size'])  # Shorten
    if verbose is None:
        verbose = sim['verbose']
    dt = sim['dt']  # Timestep

    # If a people object or popdict is supplied, use it
    if sim.cells and not reset:
        sim.cells.initialize(sim_pars=sim.pars)
        return sim.cells  # If it's already there, just return
    elif sim.popdict and popdict is None:
        popdict = sim.popdict  # Use stored one
        sim.popdict = None  # Once loaded, remove

    if popdict is None:

        pop_size = int(sim['pop_size'])  # Number of cells

        # Load age data by country if available, or use defaults.
        # Other demographic data like mortality and fertility are also available by
        # country, but these are loaded directly into the sim since they are not
        # stored as part of the people.
        # age_data =  cellDef.default_age_data
        location = sim['location']

        uids, types = set_static(pop_size, pars=sim.pars, type_ratio=0.94, dispersion=dispersion)

        # Store output
        popdict = {}
        popdict['uid'] = uids
        popdict['basal'] =
        for input in types:
            if input == 1:

        popdict['']



    # Do minimal validation and create the people
    validate_popdict(popdict, sim.pars, verbose=verbose)
    cells = cellMass.Cells(sim.pars, uid=popdict['uid'], basal=popdict['basal'], =popdict['sex'], debut=popdict['debut'],
                          partners=popdict['partners'], contacts=popdict['contacts'],
                          current_partners=popdict['current_partners'])  # List for storing the people



    return cells




# HELPFUL FOR MAKING NEW CELLS IN BULK
def set_static(new_n, existing_n=0, pars=None, type_ratio=0.96, dispersion=None):
    '''
    Set static population characteristics that do not change over time.
    Can be used when adding new births, in which case the existing popsize can be given.
    '''
    uid = np.arange(existing_n, existing_n + new_n, dtype=cellDef.default_int)
    type = np.random.binomial(1, type_ratio, new_n)


    return uid, type


def validate_popdict(popdict, pars, verbose=True):
    '''
    Check that the popdict is the correct type, has the correct keys, and has
    the correct length
    '''

    # Check it's the right type
    try:
        popdict.keys()  # Although not used directly, this is used in the error message below, and is a good proxy for a dict-like object
    except Exception as E:
        errormsg = f'The popdict should be a dictionary or hp.People object, but instead is {type(popdict)}'
        raise TypeError(errormsg) from E

    # Check keys and lengths
    required_keys = ['uid', 'age', 'sex', 'debut']
    popdict_keys = popdict.keys()
    pop_size = pars['pop_size']
    for key in required_keys:

        if key not in popdict_keys:
            errormsg = f'Could not find required key "{key}" in popdict; available keys are: {sc.strjoin(popdict.keys())}'
            sc.KeyNotFoundError(errormsg)

        actual_size = len(popdict[key])
        if actual_size != pop_size:
            errormsg = f'Could not use supplied popdict since key {key} has length {actual_size}, but all keys must have length {pop_size}'
            raise ValueError(errormsg)

        isnan = np.isnan(popdict[key]).sum()
        if isnan:
            errormsg = f'Population not fully created: {isnan:,} NaNs found in {key}.'
            raise ValueError(errormsg)

    return




# %%
