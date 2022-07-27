'''
Define core Sim classes
'''

# Imports
import numpy as np
import random as random
import pandas as pd
import sciris as sc
from . import base as cellBase
from . import misc as cellMisc
from . import default as cellDef
from . import utils as cellUtil
from . import population as cellPop
from . import parameters as cellPar
from . import analysis as cellA
from . import plotting as cellPlt
from .settings import options as cellOp
from . import cell_mass as cellMass




# Define the model
class Sim(cellBase.BaseSim):

    def __init__(self, pars=None, label=None,
                 popfile=None, people=None, version=None, **kwargs):

        # Set attributes
        self.label = label  # The label/name of the simulation
        self.created = None  # The datetime the sim was created
        self.popfile = popfile  # The population file TODO do not need this for
        self.popdict = people  # The population dictionary
        self.cells= None  # Initialize these here so methods that check their length can see they're empty
        self.t = None  # The current time in the simulation (during execution); outside of sim.step(), its value corresponds to next timestep to be computed
        self.results = {}  # For storing results
        self.summary = None  # For storing a summary of the results
        self.initialized = False  # Whether or not initialization is complete
        self.complete = False  # Whether a simulation has completed running
        self.results_ready = False  # Whether or not results are ready
        self._default_ver = version  # Default version of parameters used
        self._orig_pars = None  # Store original parameters to optionally restore at the end of the simulation

        # Make default parameters (using values from parameters.py)
        default_pars = cellPar.make_pars(version=version)  # Start with default pars
        super().__init__(default_pars)  # Initialize and set the parameters as attributes

        # Update pars
        self.update_pars(pars, **kwargs)  # Update the parameters, if provided

        return

    def initialize(self, reset=False, init_states=True, **kwargs):
        '''
        Perform all initializations on the sim.
        '''
        self.t = 0  # The current time index
        self.validate_pars()  # Ensure parameters have valid values
        self.set_seed()  # Reset the random seed before the population is created
        self.init_genotypes()  # Initialize the genotypes
        self.init_results()  # After initializing the genotypes, create the results structure
        self.init_cells(reset=reset, init_states=init_states, **kwargs)  # Create all the cells (the heaviest step)
        self.init_analyzers()  # ...and the analyzers...
        self.set_seed()  # Reset the random seed again so the random number stream is consistent
        self.initialized = True
        self.complete = False
        self.results_ready = False
        return self

    # def layer_keys(self):
    #     '''
    #     Attempt to retrieve the current layer keys.
    #     '''
    #     try:
    #         keys = list(self['acts'].keys())  # Get keys from acts
    #     except:  # pragma: no cover
    #         keys = []
    #     return keys
    #
    # def reset_layer_pars(self, layer_keys=None, force=False):
    #     '''
    #     Reset the parameters to match the population.
    #
    #     Args:
    #         layer_keys (list): override the default layer keys (use stored keys by default)
    #         force (bool): reset the parameters even if they already exist
    #     '''
    #     if layer_keys is None:
    #         if self.cells is not None:  # If people exist
    #             layer_keys = self.cells.contacts.keys()
    #         elif self.popdict is not None:
    #             layer_keys = self.popdict['layer_keys']
    #     cellPar.reset_layer_pars(self.pars, layer_keys=layer_keys, force=force)
    #     return
    #
    # def validate_layer_pars(self):
    #     '''
    #     Handle layer parameters, since they need to be validated after the population
    #     creation, rather than before.
    #     '''
    #
    #     # First, try to figure out what the layer keys should be and perform basic type checking
    #     layer_keys = self.layer_keys()
    #     layer_pars = cellPar.layer_pars  # The names of the parameters that are specified by layer
    #     for lp in layer_pars:
    #         val = self[lp]
    #         if sc.isnumber(val):  # It's a scalar instead of a dict, assume it's all contacts
    #             self[lp] = {k: val for k in layer_keys}
    #
    #     # Handle key mismatches
    #     for lp in layer_pars:
    #         lp_keys = set(self.pars[lp].keys())
    #         if not lp_keys == set(layer_keys):
    #             errormsg = 'At least one layer parameter is inconsistent with the layer keys; all parameters must have the same keys:'
    #             errormsg += f'\nsim.layer_keys() = {layer_keys}'
    #             for lp2 in layer_pars:  # Fail on first error, but re-loop to list all of them
    #                 errormsg += f'\n{lp2} = ' + ', '.join(self.pars[lp2].keys())
    #             raise sc.KeyNotFoundError(errormsg)
    #
    #     # Handle mismatches with the population
    #     if self.cells is not None:
    #         pop_keys = set(self.cells.contacts.keys())
    #         if pop_keys != set(layer_keys):  # pragma: no cover
    #             if not len(pop_keys):
    #                 errormsg = f'Your population does not have any layer keys, but your simulation does {layer_keys}. If you called cv.People() directly, you probably need cv.make_people() instead.'
    #                 raise sc.KeyNotFoundError(errormsg)
    #             else:
    #                 errormsg = f'Please update your parameter keys {layer_keys} to match population keys {pop_keys}. You may find sim.reset_layer_pars() helpful.'
    #                 raise sc.KeyNotFoundError(errormsg)
    #
    #     return

    def validate_pars(self, validate_layers=True):
        '''
        Some parameters can take multiple types; this makes them consistent.

        Args:
            validate_layers (bool): whether to validate layer parameters as well via validate_layer_pars() -- usually yes, except during initialization
        '''

        # Handle types
        for key in ['pop_size']:
            try:
                self[key] = int(self[key])
            except Exception as E:
                errormsg = f'Could not convert {key}={self[key]} of {type(self[key])} to integer'
                raise ValueError(errormsg) from E

        # Handle start
        if self['start'] in [None, 0]:  # Use default start
            self['start'] = 2015

        # Handle end and n_years
        if self['end']:
            self['n_years'] = int(self['end'] - self['start'])
            if self['n_years'] <= 0:
                errormsg = f"Number of years must be >0, but you supplied start={str(self['start'])} and end={str(self['end'])}, which gives n_years={self['n_years']}"
                raise ValueError(errormsg)
        else:
            if self['n_years']:
                self['end'] = self['start'] + self['n_years']
            else:
                errormsg = f'You must supply one of n_years and end."'
                raise ValueError(errormsg)

        # Construct other things that keep track of time
        self.years = sc.inclusiverange(self['start'], self['end'])
        self.yearvec = sc.inclusiverange(start=self['start'], stop=self['end'], step=self['dt'])
        self.npts = len(self.yearvec)
        self.tvec = np.arange(self.npts)

        # Handle population network data
        network_choices = ['random', 'basic']
        choice = self['network']
        if choice and choice not in network_choices:  # pragma: no cover
            choicestr = ', '.join(network_choices)
            errormsg = f'Population type "{choice}" not available; choices are: {choicestr}'
            raise ValueError(errormsg)





        # Handle verbose
        if self['verbose'] == 'brief':
            self['verbose'] = -1
        if not sc.isnumber(self['verbose']):  # pragma: no cover
            errormsg = f'Verbose argument should be either "brief", -1, or a float, not {type(self["verbose"])} "{self["verbose"]}"'
            raise ValueError(errormsg)

        return

    def validate_init_conditions(self, init_hpv_prev):
        '''
        Initial prevalence values can be supplied with different amounts of detail.
        Here we flesh out any missing details so that the initial prev values are
        by age and genotype. We also check the prevalence values are ok.
        '''

        def validate_arrays(vals, n_age_brackets=None):
            ''' Little helper function to check prevalence values '''
            if n_age_brackets is not None:
                if len(vals) != n_age_brackets:
                    errormsg = f'The initial prevalence values must either be the same length as the age brackets: {len(vals)} vs {n_age_brackets}.'
                    raise ValueError(errormsg)
            else:
                if len(vals) != 1:
                    errormsg = f'No age brackets were supplied, but more than one prevalence value was supplied ({len(vals)}). An array of prevalence values can only be supplied along with an array of corresponding age brackets.'
                    raise ValueError(errormsg)
            if vals.any() < 0 or vals.any() > 1:
                errormsg = f'The initial prevalence values must either between 0 and 1, not {vals}.'
                raise ValueError(errormsg)

            return

        # If values have been provided, validate them
        sex_keys = {'m', 'f'}
        tot_keys = ['all', 'total', 'tot', 'average', 'avg']
        n_age_brackets = None

        if init_hpv_prev is not None:
            if sc.checktype(init_hpv_prev, dict):
                # Get age brackets if supplied
                if 'age_brackets' in init_hpv_prev.keys():
                    age_brackets = init_hpv_prev.pop('age_brackets')
                    n_age_brackets = len(age_brackets)
                else:
                    age_brackets = np.array([150])

                # Handle the rest of the keys
                var_keys = list(init_hpv_prev.keys())
                if (len(var_keys) == 1 and var_keys[0] not in tot_keys) or (
                        len(var_keys) > 1 and set(var_keys) != sex_keys):
                    errormsg = f'Could not understand the initial prevalence provided: {init_hpv_prev}. If supplying a dictionary, please use "m" and "f" keys or "tot". '
                    raise ValueError(errormsg)
                if len(var_keys) == 1:
                    k = var_keys[0]
                    init_hpv_prev = {sk: sc.promotetoarray(init_hpv_prev[k]) for sk in sex_keys}

                # Now set the values
                for k, vals in init_hpv_prev.items():
                    init_hpv_prev[k] = sc.promotetoarray(vals)

            elif sc.checktype(init_hpv_prev, 'arraylike') or sc.isnumber(init_hpv_prev):
                # If it's an array, assume these values apply to males and females
                init_hpv_prev = {sk: sc.promotetoarray(init_hpv_prev) for sk in sex_keys}
                age_brackets = np.array([150])

            else:
                errormsg = f'Initial prevalence values of type {type(var)} not recognized, must be a dict, an array, or a float.'
                raise ValueError(errormsg)

            # Now validate the arrays
            for sk, vals in init_hpv_prev.items():
                validate_arrays(vals, n_age_brackets)

        # If values haven't been supplied, assume zero
        else:
            init_hpv_prev = {'f': np.array([0]), 'm': np.array([0])}
            age_brackets = np.array([150])

        return init_hpv_prev, age_brackets

    def init_genotypes(self):
        ''' Initialize the genotypes '''
        if self._orig_pars and 'genotypes' in self._orig_pars:
            self['genotypes'] = self._orig_pars.pop('genotypes')  # Restore

        for i, genotype in enumerate(self['genotypes']):
            if isinstance(genotype, cellMass.genotype):
                if not genotype.initialized:
                    genotype.initialize(self)
            else:  # pragma: no cover
                errormsg = f'Genotype {i} ({genotype}) is not a hp.genotype object; please create using cv.genotype()'
                raise TypeError(errormsg)

        if not len(self['genotypes']):
            print('No genotypes provided, will assume only simulating HPV 16 by default')
            hpv16 = cellMass.genotype('hpv16')
            hpv16.initialize(self)
            self['genotypes'] = [hpv16]

        len_pars = len(self['genotype_pars'])
        len_map = len(self['genotype_map'])
        assert len_pars == len_map, f"genotype_pars and genotype_map must be the same length, but they're not: {len_pars} ≠ {len_map}"
        self['n_genotypes'] = len_pars  # Each genotype has an entry in genotype_pars

        return


    def init_results(self, frequency='annual'):
        '''
        Create the main results structure.
        We differentiate between flows, stocks, and cumulative results
        The prefix "new" is used for flow variables, i.e. counting new events (infections/deaths) on each timestep
        The prefix "n" is used for stock variables, i.e. counting the total number in any given state (sus/inf/etc) on any particular timestep
        The prefix "cum" is used for cumulative variables, i.e. counting the total number that have ever been in a given state at some point in the sim

        Arguments:
            sim         (hp.Sim)        : a sim
            frequency   (str or float)  : the frequency with which to save results: accepts 'annual', 'dt', or a float which is interpreted as a fraction of a year, e.g. 0.2 will save results every 0.2 years
        '''

        # Handle frequency
        if type(frequency) == str:
            if frequency == 'annual':
                resfreq = int(1 / self['dt'])
            elif frequency == 'dt':
                resfreq = 1
            else:
                errormsg = f'Result frequency not understood: must be "annual", "dt" or a float, but you provided {frequency}.'
                raise ValueError(errormsg)
        elif type(frequency) == float:
            if frequency < self['dt']:
                errormsg = f'You requested results with frequency {frequency}, but this is smaller than the simulation timestep {self["dt"]}.'
                raise ValueError(errormsg)
            else:
                resfreq = int(frequency / self['dt'])
        self.resfreq = resfreq

        # Construct the tvec that will be used with the results
        points_to_use = np.arange(0, self.npts, self.resfreq)
        res_yearvec = self.yearvec[points_to_use]
        res_npts = len(res_yearvec)
        res_tvec = np.arange(res_npts)

        # Function to create results
        def init_res(*args, **kwargs):
            ''' Initialize a single result object '''
            output = cellBase.Result(*args, **kwargs, npts=res_npts)
            return output

        ng = self['n_genotypes']
        results = dict()

        # Create new and cumulative flows TODO figure out fi this needs to be changed for my purposes or gotten rid of
        for key, lab in zip(['cum', 'new'], ['Cumulative', 'New']):  # key and label for new vs cumulative
            for lkey, llab, cstride, g in zip(['_total', ''], ['Total ', ''], [0.95, np.linspace(0.2, 0.8, ng)], [0,
                                                                                                                  ng]):  # key, label, and color stride by level (total vs genotype-specific)
                for flow, name, cmap, by_age in zip(cellDef.flow_keys, cellDef.flow_names, cellDef.flow_colors, cellDef.flow_by_age):
                    results[f'{key + lkey}_{flow}'] = init_res(f'{llab + lab.lower()} {name}', color=cmap(cstride),
                                                               n_rows=g)
                    if by_age in ['both', 'genotype'] and lkey == '':
                        results[f'{key}_{flow}_by_age'] = init_res(f'{lab} {name} by age', color=cmap(cstride),
                                                                   n_rows=g, n_copies=cellDef.n_age_brackets)
                    if by_age in ['both', 'total'] and lkey == '_total':
                        results[f'{key + lkey}_{flow}_by_age'] = init_res(f'{llab + lab.lower()} {name} by age',
                                                                          color=cmap(cstride),
                                                                          n_rows=cellDef.n_age_brackets)

        # Create stocks
        for lkey, llabel, cstride, g in zip(['_total', ''], ['Total number', 'Number'],
                                            [0.95, np.linspace(0.2, 0.8, ng)], [0, ng]):
            for stock, name, cmap, by_age in zip(cellDef.stock_keys, cellDef.stock_names, cellDef.stock_colors, cellDef.stock_by_age):
                results[f'n{lkey}_{stock}'] = init_res(f'{llabel} {name}', color=cmap(cstride), n_rows=g)
                if by_age in ['both', 'genotype'] and lkey == '':
                    results[f'n{lkey}_{stock}_by_age'] = init_res(f'{llabel} {name} by age', color=cmap(cstride),
                                                                  n_rows=g, n_copies=cellDef.n_age_brackets)
                if by_age in ['both', 'total'] and lkey == '_total':
                    results[f'n{lkey}_{stock}_by_age'] = init_res(f'{llabel} {name} by age', color=cmap(cstride),
                                                                  n_rows=cellDef.n_age_brackets)

        # Create incidence and prevalence results
        for lkey, llab, cstride, g in zip(['total_', ''], ['Total ', ''], [0.95, np.linspace(0.2, 0.8, ng)], [0,
                                                                                                              ng]):  # key, label, and color stride by level (total vs genotype-specific)
            for var, name, cmap, by_age in zip(cellDef.inci_keys, cellDef.inci_names, cellDef.inci_colors, cellDef.inci_by_age):
                for which in ['incidence', 'prevalence']:
                    results[f'{lkey + var}_{which}'] = init_res(llab + name + ' ' + which, color=cmap(cstride),
                                                                n_rows=g)
                    if by_age in ['both', 'genotype'] and lkey == '':
                        results[f'{lkey + var}_{which}_by_age'] = init_res(llab + name + ' ' + which + ' by age',
                                                                           color=cmap(cstride), n_rows=g,
                                                                           n_copies=cellDef.n_age_brackets)
                    if by_age in ['both', 'total'] and lkey == 'total_':
                        results[f'{lkey + var}_{which}_by_age'] = init_res(llab + name + ' ' + which + ' by age',
                                                                           color=cmap(cstride),
                                                                           n_rows=cellDef.n_age_brackets)

        # Create demographic flows
        for key, lab in zip(['cum', 'new'], ['Cumulative', 'New']):  # key and label for new vs cumulative
            for var, name, color in zip(cellDef.dem_keys, cellDef.dem_names, cellDef.dem_colors):
                results[f'{key}_{var}'] = init_res(f'{lab} {name}', color=color)

        # Create results by sex
        for key, lab in zip(['cum', 'new'], ['Cumulative', 'New']):  # key and label for new vs cumulative
            for var, name, color in zip(cellDef.by_sex_keys, cellDef.by_sex_colors, cellDef.by_sex_colors):
                results[f'{key}_{var}'] = init_res(f'{lab} {name}', color=color, n_rows=2)

        # Other results
        results['r_eff'] = init_res('Effective reproduction number', scale=False, n_rows=ng)
        results['doubling_time'] = init_res('Doubling time', scale=False, n_rows=ng)
        results['n_alive'] = init_res('Number alive', scale=True)
        results['n_alive_by_sex'] = init_res('Number alive by sex', scale=True, n_rows=2)
        results['n_alive_by_age'] = init_res('Number alive by age', scale=True, n_rows=cellDef.n_age_brackets)
        results['f_alive_by_age'] = init_res('Women alive by age', scale=True, n_rows=cellDef.n_age_brackets)

        # Time vector
        results['year'] = res_yearvec
        results['t'] = res_tvec

        self.results = results
        self.results_ready = False
        return

    def init_cells(self, popdict=None, init_states=False, reset=False, verbose=None, **kwargs):
        '''
        Create the people and the network.

        Use ``init_states=False`` for creating a fresh People object for use
        in future simulations

        Args:
            popdict         (any):  pre-generated people of various formats.
            init_states     (bool): whether to initialize states (default false when called directly)
            reset           (bool): whether to regenerate the people even if they already exist
            verbose         (int):  detail to print
            kwargs          (dict): passed to hp.make_people()
        '''

        # Handle inputs
        if verbose is None:
            verbose = self['verbose']
        if popdict is not None:
            self.popdict = popdict
        if verbose > 0:
            resetstr = ''
            if self.cells:
                resetstr = ' (resetting cells)' if reset else ' (warning: not resetting sim.cells'
            print(f'Initializing sim{resetstr} with {self["pop_size"]:0n} cells')
        if self.popfile and self.popdict is None:  # If there's a popdict, we initialize it
            self.load_population(init_cells=False)

        # Actually make the people
        microstructure = self['network']
        self.cells = cellPop.make_cells(self, reset=reset, verbose=verbose, microstructure=microstructure, **kwargs)
        self.cells.initialize(sim_pars=self.pars)  # Fully initialize the people
        #self.reset_layer_pars(force=False)  # Ensure that layer keys match the loaded population
        # if init_states:
        #     init_hpv_prev = sc.dcp(self['init_hpv_prev'])
        #     init_hpv_prev, age_brackets = self.validate_init_conditions(init_hpv_prev)
        #     self.init_states(age_brackets=age_brackets, init_hpv_prev=init_hpv_prev)

        return self


    def init_analyzers(self):
        ''' Initialize the analyzers '''
        if self._orig_pars and 'analyzers' in self._orig_pars:
            self['analyzers'] = self._orig_pars.pop('analyzers')  # Restore

        for analyzer in self['analyzers']:
            if isinstance(analyzer, cellA.Analyzer):
                analyzer.initialize(self)
        return

    def finalize_analyzers(self):
        for analyzer in self['analyzers']:
            if isinstance(analyzer, cellA.Analyzer):
                analyzer.finalize(self)

    # def reset_layer_pars(self, layer_keys=None, force=False):
    #     '''
    #     Reset the parameters to match the population.
    #
    #     Args:
    #         layer_keys (list): override the default layer keys (use stored keys by default)
    #         force (bool): reset the parameters even if they already exist
    #     '''
    #     if layer_keys is None:
    #         if self.cells is not None:  # If people exist
    #             layer_keys = self.cells.contacts.keys()
    #         elif self.popdict is not None:
    #             layer_keys = self.popdict['layer_keys']
    #     cellPar.reset_layer_pars(self.pars, layer_keys=layer_keys, force=force)
    #     return

    # def init_states(self, age_brackets=None, init_hpv_prev=None, init_cin_prev=None, init_cancer_prev=None):
    #     '''
    #     Initialize prior immunity and seed infections
    #     '''
    #
    #     # Shorten key variables
    #     ng = self['n_genotypes']
    #
    #     # Assign people to age buckets
    #     age_inds = np.digitize(self.cells.age, age_brackets)
    #
    #     # Assign probabilities of having HPV to each age/sex group
    #     hpv_probs = np.full(len(self.cells), np.nan, dtype=cellDef.default_float)
    #     hpv_probs[self.cells.f_inds] = init_hpv_prev['f'][age_inds[self.cells.f_inds]]
    #     hpv_probs[self.cells.m_inds] = init_hpv_prev['m'][age_inds[self.cells.m_inds]]
    #     hpv_probs[~self.cells.is_active] = 0  # Blank out people who are not yet sexually active
    #
    #     # # Get indices of people who have HPV (for now, split evenly between genotypes)
    #     # hpv_inds = cellUtil.true(cellUtil.binomial_arr(hpv_probs))
    #     # genotypes = np.random.randint(0, ng, len(hpv_inds))
    #
    #     # # Figure of duration of infection and infect people.
    #     # dur_hpv = cellUtil.sample(**self['dur']['none'], size=len(hpv_inds))
    #     # t_imm_event = np.floor(np.random.uniform(-dur_hpv, 0) / self['dt'])
    #     # _ = self.cells.infect(inds=hpv_inds, genotypes=genotypes, offset=t_imm_event, dur=dur_hpv,
    #     #                        layer='seed_infection')
    #
    #     # # Check for CINs
    #     # cin1_filters = (self.cells.date_cin1 < 0) * (self.cells.date_cin2 > 0)
    #     # self.cells.cin1[cin1_filters.nonzero()] = True
    #     # cin2_filters = (self.cells.date_cin2 < 0) * (self.cells.date_cin3 > 0)
    #     # self.cells.cin2[cin2_filters.nonzero()] = True
    #     # cin3_filters = (self.cells.date_cin3 < 0) * (self.cells.date_cancerous > 0)
    #     # self.cells.cin3[cin3_filters.nonzero()] = True
    #
    #     return

    def step(self):
        ''' Step through time and update values '''

        # Set the time and if we have reached the end of the simulation, then do nothing
        if self.complete:
            raise AlreadyRunError('Simulation already complete (call sim.initialize() to re-run)')

        # Shorten key variables
        dt = self['dt']  # Timestep
        t = self.t
        ng = self['n_genotypes']
        gen_pars = self['genotype_pars']


        print('step')

        self.cells.update_states_pre(t=t, year=self.yearvec[t], resfreq=self.resfreq) # changes states and has
            # cell process happening in it

        # Index for results
        idx = int(t / self.resfreq)

        # Update counts for this time step: flows
        for key, count in self.cells.total_flows.items():
            self.results[key][idx] += count
        for key, count in self.cells.flows.items():
            for genotype in range(ng):
                self.results[key][genotype][idx] += count[genotype]


        # # By-age flows
        # self.results['new_infections_by_age'][:, :, idx] += people.flows_by_age['new_infections_by_age']
        # for key, count in people.total_flows_by_age.items():
        #     self.results[key][:, idx] += count

        # Make stock updates every nth step, where n is the frequency of result output
        if t % self.resfreq == 0:

            # Create total stocks
            for key in cellDef.stock_keys:
                    for g in range(ng):
                        self.results[f'n_{key}'][g, idx] = self.cells.count_by_genotype(key, g)



            # # Do total CINs separately
            # for genotype in range(ng):
            #     self.results[f'n_cin'][genotype, idx] = self.results[f'n_cin1'][genotype, idx] + \
            #                                             self.results[f'n_cin2'][genotype, idx] + \
            #                                             self.results[f'n_cin3'][genotype, idx]
            # self.results[f'n_total_cin'][idx] = self.results[f'n_total_cin1'][idx] + self.results[f'n_total_cin2'][
            #     idx] + self.results[f'n_total_cin3'][idx]
            #
            # count_age_brackets_all = people.age_brackets * (people['cin1'] + people['cin2'] + people['cin3'])
            # age_inds, n_by_age = cellUtil.unique(count_age_brackets_all)  # Get the number infected
            # self.results[f'n_total_cin_by_age'][age_inds[1:] - 1, idx] = n_by_age[1:]
            #
            # # Save number alive
            # self.results['n_alive'][idx] = len(people.alive.nonzero()[0])
            # self.results['n_alive_by_sex'][0, idx] = len((people.alive * people.is_female).nonzero()[0])
            # self.results['n_alive_by_sex'][1, idx] = len((people.alive * people.is_male).nonzero()[0])

            # # Save number alive by age
            # count_age_brackets_alive = people.age_brackets * people.alive
            # age_inds, n_by_age = cellUtil.unique(count_age_brackets_alive)  # Get the number infected
            # self.results[f'n_alive_by_age'][age_inds[1:] - 1, idx] = n_by_age[1:]
            #
            # # Save number of women alive by age
            # count_age_brackets_alive = people.age_brackets * people.alive * people.is_female
            # age_inds, n_by_age = cellUtil.unique(count_age_brackets_alive)  # Get the number infected
            # self.results[f'f_alive_by_age'][age_inds[1:] - 1, idx] = n_by_age[1:]

            # Save number infected
            self.results['n_infect'][idx] = len(cells.infected.nonzero()[0])
            self.results['n_infect_by_type'][0, idx] = len((cells.is_basal * cells.infected).nonzero()[0])
            self.results['n_infect_by_type'][1, idx] = len((cells.is_parabasal * cells.infected).nonzero()[0])

            # Save number of differentiated
            self.results['n_diff'][idx] = len(cells.differentiated.nonzero()[0])
            self.results['n_diff_by_type'][0, idx] = len((cells.is_basal * cells.differentiated).nonzero()[0])
            self.results['n_diff_by_type'][1, idx] = len((cells.is_parabasal * cells.differentiated).nonzero()[0])

            # Save number of transformed
            self.results['n_transform'][idx] = len(cells.transformed.nonzero()[0])
            self.results['n_transform_by_type'][0, idx] = len((cells.is_basal * cells.transformed).nonzero()[0])
            self.results['n_transform_by_type'][1, idx] = len((cells.is_parabasal * cells.transformed).nonzero()[0])

        # Apply analyzers
        for i, analyzer in enumerate(self['analyzers']):
            analyzer(self)

        # Tidy up
        self.t += 1
        if self.t == self.npts:
            self.complete = True

        return



    '''
    EVENT DRIVEN PROCESS
    '''
    def draw_event_class_basal_normal(self, basals):
        '''
        Draws the type of event that could occur for each basal cell (division, or infection)

        Args:
            basals  (array): array of indices of normal basal cells

        Returns:
            (int): event class value
        '''

        basal_normal_bb_rate = 0  # draws the event class possibility
        basal_normal_pp_rate = 0
        basal_normal_bp_rate = 0
        infect_rate = 0


        for i in basals: # sample from distribution instead of looping
            basal_normal_bp_rate += cellPar.get_division_rate(i) #TODO access the split rates in parameters
            basal_normal_bb_rate += cellPar.get_division_rate(i)
            basal_normal_pp_rate += cellPar.get_division_rate(i)
            infect_rate += cellPar.get_infect_rate(i)



        basal_normal_bp_start = 0
        basal_normal_bp_end = basal_normal_bp_rate
        basal_normal_bb_end = basal_normal_bp_end + basal_normal_bb_rate
        basal_normal_pp_end = basal_normal_bb_end + basal_normal_pp_rate
        infect_end = basal_normal_pp_end + infect_rate

        normal_basal_events = np.zeros((len(basals))) # make array, multinomial pull in utilities?? or keep strings??
        for b in range(len(basals)):

            random_draw = random.uniform(basal_normal_bp_start, infect_end)

            if random_draw < basal_normal_bp_end:
                normal_basal_events.append("asymmetric BP")    #possible string change          # asymmetric normal split (BP) from basal
            elif (random_draw >= basal_normal_bp_end) & (random_draw < basal_normal_bb_end):
                normal_basal_events.append("symmetric BB")                      # symmetric normal split (BB) from basal
            elif (random_draw >= basal_normal_bb_end) & (random_draw < basal_normal_pp_end):
                normal_basal_events.append("symmetric PP")                       # symmetric normal split (PP) from basal
            elif (random_draw >= basal_normal_pp_end) & (random_draw < infect_end):
                normal_basal_events.append("infect")                     # infection event

        return normal_basal_events

    def draw_event_class_basal_infect(self, infected_basals):
        '''
        Draws the type of event that could occur for each basal cell (division, or transformation)

        Args:
              infected_basals   (array): array of indices of infected basal cells

        Returns:
            normal_basal_events (list): list of events
        '''


        basal_infect_bb_rate = 0
        basal_infect_pp_rate = 0
        basal_infect_bp_rate = 0
        transform_rate = 0

        # Construct the vectors TODO filter here
        basal_inds_infect = []


        for i in infected_basals:
            basal_infect_bb_rate += cellPar.get_division_rate(i)
            basal_infect_pp_rate += cellPar.get_division_rate(i)
            basal_infect_bp_rate += cellPar.get_division_rate(i)
            transform_rate += cellPar.get_transform_rate(i)

        basal_infect_bp_start = 0
        basal_infect_bb_end = basal_infect_bp_start + basal_infect_bb_rate
        basal_infect_bp_end = basal_infect_bb_end + basal_infect_bp_rate
        basal_infect_pp_end = basal_infect_bp_end + basal_infect_pp_rate
        transform_end = basal_infect_pp_end + transform_rate

        infected_basal_events = []

        for size in range(len(infected_basals)):
            random_draw = random.uniform(basal_infect_bp_start, transform_end)

            if random_draw < basal_infect_bb_end:
                infected_basal_events.append("symmetric infect BB")                      # symmetric infected split (BB) from basal
            elif (random_draw >= basal_infect_bb_end) & (random_draw < basal_infect_bp_end):
                infected_basal_events.append("asymmetric infect BP")                       # asymmetric infected split (BP) from basal
            elif (random_draw >= basal_infect_bp_end) & (random_draw < basal_infect_pp_end):
                infected_basal_events.append("symmetric infect PP")                       # symmetric infected split (PP) from basal
            elif(random_draw >= basal_infect_pp_end) & (random_draw < transform_end):
                infected_basal_events.append("transform")                        # transformation event
        return infected_basal_events

    def draw_event_class_parabasal_normal(self, parabasals):
        '''
        Draws the type of event that could occur for each parabasal cell (division, or differentiation)

        Args:
            parabasals   (array): array of indices of normal parabasal cells

        Returns:
            parabasal_events (list): list of events
        '''


        pbasal_normal_pp_rate = 0
        diff_normal_rate = 0


        for i in parabasals:
            pbasal_normal_pp_rate += cellPar.get_division_rate(i) #TODO access the split rates in parameters
            diff_normal_rate += cellPar.get_diff_rate(i)


        pbasal_normal_pp_start = 0
        pbasal_normal_pp_end = pbasal_normal_pp_rate
        diff_normal_end = pbasal_normal_pp_end + diff_normal_rate


        parabasal_events = []
        for size in range(len(parabasals)):
            random_draw = random.uniform(pbasal_normal_pp_start, diff_normal_end)

            if random_draw < pbasal_normal_pp_start:
                # Assign 5 to the attribute for that cell
                parabasal_events.append("symmetric PP")            # symmetric normal split (PP) from parabasal
            elif (random_draw >= pbasal_normal_pp_start) & (random_draw < diff_normal_end):
                parabasal_events.append("differentiate")

        return parabasal_events

    def draw_event_class_parabasal_infected(self, infected_parabasals):
        '''
        Draws the type of event that could occur for each parabasal cell (division, or differentiation)

        Args:
            infected_parabasals   (array): array of indices of infected parabasal cells

        Returns:
            infected_parabasal_events (list): list of events
        '''


        pbasal_infect_pp_rate = 0
        diff_infect_rate = 0


        for i in infected_parabasals:
            pbasal_infect_pp_rate += cellPar.get_division_rate(i) #TODO access the split rates in parameters
            diff_infect_rate += cellPar.get_diff_rate(i)


        pbasal_infect_pp_start = 0
        pbasal_infect_pp_end = pbasal_infect_pp_rate
        diff_infect_end = pbasal_infect_pp_end + diff_infect_rate


        infected_parabasal_events = []
        for size in range(len(infected_parabasals)):
            random_draw = random.uniform(pbasal_infect_pp_start, diff_infect_end)

            if random_draw < pbasal_infect_pp_start:
                infected_parabasal_events.append("symmetric infect PP")            # symmetric infected split (PP) from parabasal
            elif (random_draw >= pbasal_infect_pp_start) & (random_draw < diff_infect_end):
                infected_parabasal_events.append("differentiate infect")


    def draw_events(self, max_rate, event_list, cell_list):
        '''
        From the list of events that have the potential to happen to a certain type of cell, that event is either accepted for rejected according to the acceptance rate.
         Events that could occur are either cell splitting (basal cell to two new basal cells, two new parabasal cells, or a parabasal cell and a basal cell,
        or parabasal cell to two new parabasal cells,
        or parabasal cells differentiating,
        or basal cell becoming infected,
        or infected cell becoming transformed)

        Args:
            max_rate (float): largest rate in the vector
            event_list (list): vector of events that correspond with the cell_list
            cell_list (list): vector of cell indices that fit a certain criteria

        Returns:
            chosen_events (Dictionary): Keys are the indices of the cells having events happen to them
                                        Values are the events happening for that particular cell index
        '''
        chosen_events = {}
        for i in range(len(event_list)):
            accept_rate = cell_list[i].event_rate / max_rate
            random_draw = random.uniform(0,1)
            if random_draw < accept_rate:
                chosen_events[cell_list[i]] = event_list[i]
                # this will only show the indices that have events accepted


        return chosen_events

    def run(self, do_plot=False, until=None, restore_pars=True, reset_seed=True, verbose=None): #TODO this is where the event-driven piece needs to come into play
        ''' Run the model once '''
        # Initialization steps -- start the timer, initialize the sim and the seed, and check that the sim hasn't been run
        T = sc.timer()

        if not self.initialized:
            self.initialize()
            self._orig_pars = sc.dcp(
                self.pars)  # Create a copy of the parameters, to restore after the run, in case they are dynamically modified

        if verbose is None:
            verbose = self['verbose']

        if reset_seed:
            # Reset the RNG. If the simulation is newly created, then the RNG will be reset by sim.initialize() so the use case
            # for resetting the seed here is if the simulation has been partially run, and changing the seed is required
            self.set_seed()

        # Check for AlreadyRun errors
        errormsg = None
        until = self.npts if until is None else self.get_t(until)
        if until > self.npts:
            errormsg = f'Requested to run until t={until} but the simulation end is t={self.npts}'
        if self.t >= until:  # NB. At the start, self.t is None so this check must occur after initialization
            errormsg = f'Simulation is currently at t={self.t}, requested to run until t={until} which has already been reached'
        if self.complete:
            errormsg = 'Simulation is already complete (call sim.initialize() to re-run)'
        if self.people.t not in [self.t,
                                 self.t - 1]:  # Depending on how the sim stopped, either of these states are possible
            errormsg = f'The simulation has been run independently from the people (t={self.t}, people.t={self.people.t}): if this is intentional, manually set sim.people.t = sim.t. Remember to save the people object before running the sim.'
        if errormsg:
            raise AlreadyRunError(errormsg)

        # Main simulation loop
        while self.t < until: # WORK FROM HERE

            # Check if we were asked to stop
            elapsed = T.toc(output=True)
            if self['timelimit'] and elapsed > self['timelimit']:
                sc.printv(
                    f"Time limit ({self['timelimit']} s) exceeded; call sim.finalize() to compute results if desired",
                    1, verbose)
                return
            elif self['stopping_func'] and self['stopping_func'](self):
                sc.printv(
                    "Stopping function terminated the simulation; call sim.finalize() to compute results if desired", 1,
                    verbose)
                return

            # Print progress
            if verbose:
                simlabel = f'"{self.label}": ' if self.label else ''
                string = f'  Running {simlabel}{self.yearvec[self.t]} ({self.t:2.0f}/{self.npts}) ({elapsed:0.2f} s) '
                if verbose >= 2:
                    sc.heading(string)
                elif verbose > 0:
                    if not (self.t % int(1.0 / verbose)):
                        sc.progressbar(self.t + 1, self.npts, label=string, length=20, newline=True)

            # Do the heavy lifting -- actually run the model!
            self.step()

        # If simulation reached the end, finalize the results
        if self.complete:
            self.finalize(verbose=verbose, restore_pars=restore_pars)
            sc.printv(f'Run finished after {elapsed:0.2f} s.\n', 1, verbose)

        return self

    def finalize(self, verbose=None, restore_pars=True):
        ''' Compute final results '''

        if self.results_ready:
            # Because the results are rescaled in-place, finalizing the sim cannot be run more than once or
            # otherwise the scale factor will be applied multiple times
            raise AlreadyRunError('Simulation has already been finalized')

        # Calculate cumulative results
        for key, by_age in zip(cellDef.flow_keys, cellDef.flow_by_age):
            self.results[f'cum_total_{key}'][:] += np.cumsum(self.results[f'new_total_{key}'][:], axis=0)
            self.results[f'cum_{key}'][:] += np.cumsum(self.results[f'new_{key}'][:], axis=1)
            if by_age in ['both', 'total']:
                self.results[f'cum_total_{key}_by_age'][:] += np.cumsum(self.results[f'new_total_{key}_by_age'][:],
                                                                        axis=-1)
            if by_age in ['both', 'genotype']:
                self.results[f'cum_{key}_by_age'][:] += np.cumsum(self.results[f'new_{key}_by_age'][:], axis=-1)

        for key in cellDef.by_sex_keys:
            self.results[f'cum_{key}'][:] += np.cumsum(self.results[f'new_{key}'][:], axis=-1)

        self.results[f'cum_other_deaths'][:] += np.cumsum(self.results[f'new_other_deaths'][:], axis=-1)
        self.results[f'cum_births'][:] += np.cumsum(self.results[f'new_births'][:], axis=-1)

        # Finalize analyzers and interventions
        self.finalize_analyzers()
        # self.finalize_interventions()

        # Final settings
        self.results_ready = True  # Set this first so self.summary() knows to print the results
        self.t -= 1  # During the run, this keeps track of the next step; restore this be the final day of the sim

        # Perform calculations on results
        self.compute_results(verbose=verbose)  # Calculate the rest of the results
        self.results = sc.objdict(
            self.results)  # Convert results to a odicts/objdict to allow e.g. sim.results.diagnoses

        # Optionally print summary output
        if verbose:  # Verbose is any non-zero value
            if verbose > 0:  # Verbose is any positive number
                self.summarize()  # Print medium-length summary of the sim
            else:
                self.brief()  # Print brief summary of the sim

        return

    def compute_results(self, verbose=None):
        ''' Perform final calculations on the results '''
        self.compute_states()
        self.compute_summary()
        return

    def compute_states(self):
        '''
        Compute prevalence, incidence, and other states.
        '''
        res = self.results

        # Compute HPV incidence and prevalence
        self.results['total_hpv_incidence'][:] = res['new_total_infections'][:] / res['n_susceptible'][:].sum(axis=0)
        self.results['hpv_incidence'][:] = res['new_infections'][:] / res['n_susceptible'][:]
        self.results['total_hpv_prevalence'][:] = res['n_total_infectious'][:] / res['n_alive'][:]
        self.results['hpv_prevalence'][:] = res['n_infectious'][:] / res['n_alive'][:]

        # # Compute CIN and cancer prevalence
        # alive_females = res['n_alive_by_sex'][0, :]
        # self.results['total_cin1_prevalence'][:] = res['n_total_cin1'][:] / alive_females
        # self.results['total_cin2_prevalence'][:] = res['n_total_cin2'][:] / alive_females
        # self.results['total_cin3_prevalence'][:] = res['n_total_cin3'][:] / alive_females
        # self.results['total_cin_prevalence'][:] = res['n_total_cin'][:] / alive_females
        # self.results['total_cancer_prevalence'][:] = res['n_total_cancerous'][:] / alive_females
        # self.results['cin1_prevalence'][:] = res['n_cin1'][:] / alive_females
        # self.results['cin2_prevalence'][:] = res['n_cin2'][:] / alive_females
        # self.results['cin3_prevalence'][:] = res['n_cin3'][:] / alive_females
        # self.results['cin_prevalence'][:] = res['n_cin'][:] / alive_females
        # self.results['cancer_prevalence'][:] = res['n_cancerous'][:] / alive_females

        # # Compute CIN and cancer incidence. Technically the denominator should be number susceptible
        # # to CIN/cancer, not number alive, but should be small enough that it won't matter (?)
        # at_risk_females = alive_females - res['n_cancerous'].values.sum(axis=0)
        # scale_factor = 1e5  # Cancer and CIN incidence are displayed as rates per 100k women
        # demoninator = at_risk_females * scale_factor
        # self.results['total_cin1_incidence'][:] = res['new_total_cin1s'][:] / demoninator
        # self.results['total_cin2_incidence'][:] = res['new_total_cin2s'][:] / demoninator
        # self.results['total_cin3_incidence'][:] = res['new_total_cin3s'][:] / demoninator
        # self.results['total_cin_incidence'][:] = res['new_total_cins'][:] / demoninator
        # self.results['total_cancer_incidence'][:] = res['new_total_cancers'][:] / demoninator
        # self.results['cin1_incidence'][:] = res['new_cin1s'][:] / demoninator
        # self.results['cin2_incidence'][:] = res['new_cin2s'][:] / demoninator
        # self.results['cin3_incidence'][:] = res['new_cin3s'][:] / demoninator
        # self.results['cin_incidence'][:] = res['new_cins'][:] / demoninator
        # self.results['cancer_incidence'][:] = res['new_cancers'][:] / demoninator
        #
        # # Finally, add results by age
        # self.results['total_hpv_prevalence_by_age'][:] = res['n_total_infectious_by_age'][:] / self.results[
        #                                                                                            'n_alive_by_age'][:]
        # self.results['total_hpv_incidence_by_age'][:] = res['new_total_infections_by_age'][:] / self.results[
        #                                                                                             'n_total_susceptible_by_age'][
        #                                                                                         :]
        # cin_inci_denom = (self.results['f_alive_by_age'][:] - res['n_total_cancerous_by_age'][:]) * 1e5
        # self.results['total_cin_prevalence_by_age'][:] = res['n_total_cin_by_age'][:] / cin_inci_denom
        # self.results['total_cancer_prevalence_by_age'][:] = res['n_total_cancerous_by_age'][:] / cin_inci_denom
        # self.results['total_cin_incidence_by_age'][:] = res['new_total_cins_by_age'][:] / cin_inci_denom
        # self.results['total_cancer_incidence_by_age'][:] = res['new_total_cancers_by_age'][:] / cin_inci_denom

        return

    def compute_summary(self, t=None, update=True, output=False, require_run=False):
        '''
        Compute the summary dict and string for the sim. Used internally; see
        sim.summarize() for the user version.

        Args:
            t (int/str): day or date to compute summary for (by default, the last point)
            update (bool): whether to update the stored sim.summary
            output (bool): whether to return the summary
            require_run (bool): whether to raise an exception if simulations have not been run yet
        '''
        if t is None:
            t = -1

        # Compute the summary
        if require_run and not self.results_ready:
            errormsg = 'Simulation not yet run'
            raise RuntimeError(errormsg)

        summary = sc.objdict()
        for key in self.result_keys():
            summary[key] = self.results[key][t]

        # Update the stored state
        if update:
            self.summary = summary

        # Optionally return
        if output:
            return summary
        else:
            return

    def summarize(self, full=False, t=None, sep=None, output=False):
        '''
        Print a medium-length summary of the simulation, drawing from the last time
        point in the simulation by default. Called by default at the end of a sim run.
        point in the simulation by default. Called by default at the end of a sim run.
        See also sim.disp() (detailed output) and sim.brief() (short output).

        Args:
            full   (bool):    whether or not to print all results (by default, only cumulative)
            t      (int/str): day or date to compute summary for (by default, the last point)
            sep    (str):     thousands separator (default ',')
            output (bool):    whether to return the summary instead of printing it

        **Examples**::

            sim = cv.Sim(label='Example sim', verbose=0) # Set to run silently
            sim.run() # Run the sim
            sim.summarize() # Print medium-length summary of the sim
            sim.summarize(t=24, full=True) # Print a "slice" of all sim results on day 24
        '''
        # Compute the summary
        summary = self.compute_summary(t=t, update=False, output=True)

        # Construct the output string
        if sep is None: sep = cellOp.sep  # Default separator
        labelstr = f' "{self.label}"' if self.label else ''
        string = f'Simulation{labelstr} summary:\n'
        for key in self.result_keys():
            if full or key.startswith('cum_total') and 'by_sex' not in key and 'by_age' not in key:
                val = np.round(summary[key])
                string += f'   {val:10,.0f} {self.results[key].name.lower()}\n'.replace(',',
                                                                                        sep)  # Use replace since it's more flexible

        # Print or return string
        if not output:
            print(string)
        else:
            return string

    def plot(self, *args, **kwargs):
        ''' Plot the outputs of the model '''
        fig = cellPlt.plot_sim(sim=self, *args, **kwargs)
        return fig


class AlreadyRunError(RuntimeError):
    '''
    This error is raised if a simulation is run in such a way that no timesteps
    will be taken. This error is a distinct type so that it can be safely caught
    and ignored if required, but it is anticipated that most of the time, calling
    sim.run() and not taking any timesteps, would be an inadvertent error.
    '''
    pass
