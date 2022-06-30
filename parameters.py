'''
Set the parameters for hpvsim.
'''

import numpy as np
import sciris as sc
from .settings import options as cellOp  # For setting global options
from . import misc as cellMisc
from . import default as cellDef



__all__ = ['make_pars', 'reset_layer_pars', 'get_prognoses']


def make_pars(version=None, nonactive_by_age=False, set_prognoses=False, **kwargs):
    '''
    Create the parameters for the simulation. Typically, this function is used
    internally rather than called by the user; e.g. typical use would be to do
    sim = hp.Sim() and then inspect sim.pars, rather than calling this function
    directly.

    Args:
        version       (str):  if supplied, use parameters from this version
        kwargs        (dict): any additional kwargs are interpreted as parameter names

    Returns:
        pars (dict): the parameters of the simulation
    '''
    pars = {}

    # Population parameters
    pars['pop_size'] = 20e3  # Number of agents
    pars['network'] = 'random'  # What type of sexual network to use -- 'random', 'basic', other options TBC
    pars['location'] = None  # What location to load data from -- default Seattle
    pars['death_rates'] = None  # Deaths from all other causes, loaded below
    pars['birth_rates'] = None  # Birth rates, loaded below

    # Initialization parameters
    pars['init_hpv_prev'] = cellDef.default_init_prev  # Initial prevalence

    # Simulation parameters
    pars['start'] = 2015.  # Start of the simulation
    pars['end'] = None  # End of the simulation
    pars['n_years'] = 10.  # Number of years to run, if end isn't specified
    pars['dt'] = 0.2  # Timestep (in years)
    pars['rand_seed'] = 1  # Random seed, if None, don't reset
    pars[
        'verbose'] = cellOp.verbose  # Whether or not to display information during the run -- options are 0 (silent), 0.1 (some; default), 1 (default), 2 (everything)


    # Events
    pars['analyzers'] = []  # Custom analysis functions; populated by the user
    pars['timelimit'] = None  # Time limit for the simulation (seconds)
    pars['stopping_func'] = None  # A function to call to stop the sim partway through

    # Update with any supplied parameter values and generate things that need to be generated
    pars.update(kwargs)
    reset_layer_pars(pars)

    return pars









def get_births_deaths(location=None, verbose=1, by_sex=True, overall=False, die=None):
    '''
    Get mortality and fertility data by location if provided, or use default

    Args:
        location (str):  location; if none specified, use default value for XXX
        verbose (bool):  whether to print progress
        by_sex   (bool): whether to get sex-specific death rates (default true)
        overall  (bool): whether to get overall values ie not disaggregated by sex (default false)

    Returns:
        death_rates (dict): nested dictionary of death rates by sex (first level) and age (second level)
        birth_rates (arr): array of crude birth rates by year
    '''

    birth_rates = cellDef.default_birth_rates
    death_rates = cellDef.default_death_rates


    return birth_rates, death_rates


# %% Genotype/immunity parameters and functions

def get_hpv_prevalence():
    '''
    Get HPV prevalence data by age and genotype for initializing the sim

    Args:
        filename (str):  filename; if none specified, use default value for XXX

    Returns:
        hpv_prevalence (dict): nested dictionary of hpv prevalence by sex (first level),  age (second level), and genotype (third level)
    '''

    hpv_prevalence = cellDef.default_hpv_prevalence

    return hpv_prevalence


def get_genotype_choices():
    '''
    Define valid genotype names
    '''
    # List of choices available
    choices = {
        'hpv16': ['hpv16', '16'],
        'hpv18': ['hpv18', '18'],
        'hpv6': ['hpv6', '6'],
        'hpv11': ['hpv11', '11'],
        'hpv31': ['hpv31', '31'],
        'hpv33': ['hpv33', '33'],
        'hpv45': ['hpv45', '45'],
        'hpv52': ['hpv52', '52'],
        'hpv58': ['hpv58', '58'],
        'hpvlo': ['hpvlo', 'low', 'low-risk'],
        'hpvhi': ['hpvhi', 'high', 'high-risk'],
        'hpvhi5': ['hpvhi5', 'high5'],
    }
    mapping = {name: key for key, synonyms in choices.items() for name in synonyms}  # Flip from key:value to value:key
    return choices, mapping


def _get_from_pars(pars, default=False, key=None, defaultkey='default'):
    ''' Helper function to get the right output from genotype functions '''

    # If a string was provided, interpret it as a key and swap
    if isinstance(default, str):
        key, default = default, key

    # Handle output
    if key is not None:
        try:
            return pars[key]
        except Exception as E:
            errormsg = f'Key "{key}" not found; choices are: {sc.strjoin(pars.keys())}'
            raise sc.KeyNotFoundError(errormsg) from E
    elif default:
        return pars[defaultkey]
    else:
        return pars


def get_genotype_pars(default=False, genotype=None):
    '''
    Define the default parameters for the different genotypes
    '''
    pars = dict(

        hpv16=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv18=dict(
            rel_beta=0.8,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=0.8,
            rel_death_prob=0.8
        ),

        hpv31=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv33=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv45=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv52=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpv6=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=0,
            rel_cin2_prob=0,
            rel_cin3_prob=0,
            rel_cancer_prob=0,
            rel_death_prob=0
        ),

        hpv11=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=0,
            rel_cin2_prob=0,
            rel_cin3_prob=0,
            rel_cancer_prob=0,
            rel_death_prob=0
        ),

        hpvlo=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=0,
            rel_cin2_prob=0,
            rel_cin3_prob=0,
            rel_cancer_prob=0,
            rel_death_prob=0
        ),

        hpvhi=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

        hpvhi5=dict(
            rel_beta=1.0,  # Default values
            rel_cin1_prob=1.0,
            rel_cin2_prob=1.0,
            rel_cin3_prob=1.0,
            rel_cancer_prob=1.0,
            rel_death_prob=1.0
        ),

    )

    return _get_from_pars(pars, default, key=genotype, defaultkey='hpv16')

