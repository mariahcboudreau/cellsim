'''
Set the defaults across each of the different files.

TODO: review/merge this across the different *sims

'''

import numpy as np
import numba as nb
import sciris as sc
import pylab as pl
from .settings import options as cellOp # To set options

# Specify all externally visible functions this file defines -- other things are available as e.g. hp.defaults.default_int
__all__ = ['default_float', 'default_int', 'get_default_colors', 'get_default_plots']


#%% Specify what data types to use

result_float = np.float64 # Always use float64 for results, for simplicity
if cellOp.precision == 32:
    default_float = np.float32
    default_int   = np.int32
    nbfloat       = nb.float32
    nbint         = nb.int32
elif cellOp.precision == 64: # pragma: no cover
    default_float = np.float64
    default_int   = np.int64
    nbfloat       = nb.float64
    nbint         = nb.int64
else:
    raise NotImplementedError(f'Precision must be either 32 bit or 64 bit, not {cellOp.precision}')


#%% Define all properties of cells

class CellMeta(sc.prettyobj):
    ''' For storing all the keys relating to a cell and cell mass '''

    def __init__(self):

        # Set the properties of a cell
        self.cell = [
            'uid',              # Int
            'type',             # String
            'viral_load',       # Int
            'split_rate'        # Float
        ]

        # Set the states that a cell can be in, all booleans per cell and per genotype except other_dead
        self.states = [
            'infected',         # bool
            'differentiated'    # bool
            'transformed',      # bool
            'alive'             # Save this as a state so we can record population sizes
        ]




        self.dates = [f'date_{state}' for state in self.states if state != 'alive'] # Convert each state into a date
        self.dates += ['date_death']

        # # Duration of different states: these are floats per person -- used in people.py
        # self.durs = [
        #     'dur_hpv', # Length of time that a person has HPV DNA present. This is EITHER the period until the virus clears OR the period until viral integration
        #     'dur_none2cin1', # Length of time to go from no dysplasia to CIN1
        #     'dur_cin12cin2', # Length of time to go from CIN1 to CIN2
        #     'dur_cin22cin3', # Length of time to go from CIN2 to CIN3
        #     'dur_cin2cancer',# Length of time to go from CIN3 to cancer
        #     'dur_cancer',  # Duration of cancer
        # ]

        self.all_states = self.cell + self.states + self.dates

        # Validate
        self.state_types = ['cell', 'states', 'dates', 'all_states']
        for state_type in self.state_types:
            states = getattr(self, state_type)
            n_states        = len(states)
            n_unique_states = len(set(states))
            if n_states != n_unique_states: # pragma: no cover
                errormsg = f'In {state_type}, only {n_unique_states} of {n_states} state names are unique'
                raise ValueError(errormsg)

        return


#%% Default result settings

# Flows: we count new and cumulative totals for each
# All are stored (1) by genotype and (2) as the total across genotypes
# the by_age vector tells the sim which results should be stored by age - should have entries in [None, 'total', 'genotype', 'both']
flow_keys   = ['infections',    'basal_count', 'parabasal_count', 'viral_load', 'transformed']
flow_names  = ['infections',    'Basal Cells', 'Parabasal Cells', 'Viral Load', 'Transformed']
flow_colors = [pl.cm.GnBu,      pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds, pl.cm.Purples,      pl.cm.GnBu]
flow_by_type = ['both',          None,           None,             'total',        'total']

# Stocks: the number in each of the following states
# All are stored (1) by genotype and (2) as the total across genotypes
# the by_type vector tells the sim which results should be stored by type - should have entries in [None, 'total', 'genotype', 'both']
stock_keys   = ['infected',  'differentiated',   'transformed',    'alive']
stock_names  = ['infected',  'differentiated',   'transformed',    'alive']
stock_colors = [pl.cm.Greens,   pl.cm.GnBu,     pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds]
stock_by_type = ['both',        None,            'total',        None]
# Incidence and prevalence. Strong overlap with stocks, but with slightly different naming conventions
# All are stored (1) by genotype and (2) as the total across genotypes
inci_keys   = ['hpv', 'differentiated', 'transformed']
inci_names  = ['hpv', 'differentiated',  'transformed']
inci_colors = [pl.cm.GnBu,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Oranges,  pl.cm.Reds]
inci_by_type = ['both',         'total',        'total']



default_birth_rates = np.array([
    [2015, 2016, 2017, 2018, 2019],
    [12.4, 12.2, 11.8, 11.6, 11.4],
])

default_init_prev = {
    'basal'             : np.array([ 0.06]), #TODO double check that this is the right way to implement proportions
    'parabasal'             : np.array([0.94])
}

#%% Default plotting settings

# Define the 'overview plots', i.e. the most useful set of plots to explore different aspects of a simulation
overview_plots = [
    'cum_total_infections',
    'cum_total_differentiated',
    'cum_total_transformed',
    'cum_total_viral_load'
]


def get_default_plots(which='default', kind='sim', sim=None):
    '''
    Specify which quantities to plot; used in sim.py.

    Args:
        which (str): either 'default' or 'overview'
    '''
    which = str(which).lower() # To make comparisons easier

    # Check that kind makes sense
    sim_kind   = 'sim'
    scens_kind = 'scens'
    kindmap = {
        None:      sim_kind,
        'sim':     sim_kind,
        'default': sim_kind,
        'msim':    scens_kind,
        'scen':    scens_kind,
        'scens':   scens_kind,
    }
    if kind not in kindmap.keys():
        errormsg = f'Expecting "sim" or "scens", not "{kind}"'
        raise ValueError(errormsg)
    else:
        is_sim = kindmap[kind] == sim_kind

    # Default plots -- different for sims and scenarios
    if which in ['none', 'default']: #TODO is this correct

        if is_sim:
            plots = sc.odict({
                'HPV prevalence': [
                    'total_hpv_prevalence',
                    'hpv_prevalence',
                ],
                'HPV incidence': [
                    'total_hpv_incidence_by_type',
                    # 'new_infections',
                ]
            })

        else: # pragma: no cover
            plots = sc.odict({
                'Cumulative infections': [
                    'cum_infections',
                ],
                'New infections per day': [
                    'new_infections',
                ],
            })

    # Show an overview
    elif which == 'overview': # pragma: no cover
        plots = sc.dcp(overview_plots)

    # Plot absolutely everything
    elif which == 'all': # pragma: no cover
        plots = sim.result_keys('all')

    # Show an overview
    elif 'overview' in which: # pragma: no cover
        plots = sc.dcp(overview_plots)

    else: # pragma: no cover
        errormsg = f'The choice which="{which}" is not supported'
        raise ValueError(errormsg)

    return plots
