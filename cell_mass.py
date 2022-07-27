'''
Defines the Cells class and functions associated with making people and handling
the transitions between states (e.g., from susceptible to infected).
'''

# %% Imports
import numpy as np
import sciris as sc
from collections import defaultdict
from . import version as cellV
from . import utils as cellUtil
from . import default as cellDef
from . import base as cellBase
from . import population as cellPop
from . import plotting as cellPlt
from . import parameters as cellPar


__all__ = ['Cells']


class Cells(cellBase.BaseCell):
    '''
    A class to perform all the operations on the cells -- usually not invoked directly.

    This class is usually created automatically by the sim. The only required input
    argument is the population size, but typically the full parameters dictionary
    will get passed instead since it will be needed before the People object is
    initialized. However, ages, contacts, etc. will need to be created separately --
    see ``hp.make_people()`` instead.

    Note that this class handles the mechanics of updating the actual cells, while
    ``cellBase.BaseCell`` takes care of housekeeping (saving, loading, exporting, etc.).
    Please see the BaseCell class for additional methods.

    Args:
        pars (dict): the sim parameters, e.g. sim.pars -- alternatively, if a number, interpreted as pop_size
        strict (bool): whether or not to only create keys that are already in self.meta.person; otherwise, let any key be set
        kwargs (dict): the actual data, e.g. from a popdict, being specified

    **Examples**::

        cell1 = cellBase.Cells(2000)

        sim = cellSim.Sim()
        cell2 = cellBase.Cells(sim.pars)
    '''

    def __init__(self, pars, strict=True, **kwargs):

        # Initialize the BaseCells, which also sets things up for filtering
        super().__init__()

        # Handle pars and population size
        self.set_pars(pars)
        self.version = cellV.__version__  # Store version info

        # Other initialization
        self.t = 0  # Keep current simulation time
        self._lock = False  # Prevent further modification of keys
        self.meta = cellDef.CellMeta()  # Store list of keys and dtypes


        # Set cell properties -- all floats except for UID
        for key in self.meta.cell:
            if key == 'uid':
                self[key] = np.arange(self.pars['pop_size'], dtype=cellDef.default_int)
            elif key == 'basal':
                self[key] = np.arange(self.pars['pop_size'], dtype=cellDef.default_int)
            elif key == 'parabasal':
                self[key] = np.arange(self.pars['pop_size'], dtype=cellDef.default_int)
            elif key == 'viral_load':
                self[key] = np.arange(self.pars['pop_size'], dtype=cellDef.default_int)
            elif key == 'possible_event':
                self[key] = np.arange(self.pars['pop_size'], dtype=cellDef.default_int )
            elif key == 'location':
                self[key] = np.arange(self.pars['pop_size'], dtype=tuple)
            else:
                self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cellDef.default_float)

        # Set cell states -- booleans except exposed by genotype which should return the genotype that ind is exposed to
        for key in self.meta.states:
            if key == 'infected':     # All false at the beginning TODO: adjust to construct percentages of the basal and parabasal cells
                self[key] = np.full(self.pars['pop_size'], False, dtype=bool)
            elif key == 'differentiated':                               # All false at the beginning
                self[key] = np.full(self.pars['pop_size'], False, dtype=bool)
            elif key == 'transformed':                                         # All false at the start
                self[key] = np.full(self.pars['pop_size'], False, dtype=bool)
            elif key == 'alive':                                             # All true at the start
                self[key] = np.full(self.pars['pop_size'], True, dtype=bool)
            else:
                self[key] = np.full((self.pars['n_genotypes'], self.pars['pop_size']), False, dtype=bool)

        # Set dates and durations -- both floats
        for key in self.meta.dates:
            if key == 'date_death':
                self[key] = np.full(self.pars['pop_size'], np.nan, dtype=cellDef.default_float)
            else:
                self[key] = np.full((self.pars['n_genotypes'], self.pars['pop_size']), np.nan, dtype=cellDef.default_float)



        # Store the dtypes used in a flat dict
        self._dtypes = {key: self[key].dtype for key in self.keys()}  # Assign all to float by default
        if strict:
            self.lock()  # If strict is true, stop further keys from being set (does not affect attributes)

        # Store flows to be computed during simulation
        self.init_flows()

        # Although we have called init(), we still need to call initialize()
        self.initialized = False


        return

    #TODO Ask about singular instance

    def init_flows(self):
        ''' Initialize flows to be zero '''
        ng = self.pars['n_genotypes']
        df = cellDef.default_float
        self.flows = {f'new_{key}': np.zeros(ng, dtype=df) for key in cellDef.flow_keys}
        self.total_flows = {f'new_total_{key}': 0 for key in cellDef.flow_keys}

        return



    def initialize(self, sim_pars=None):
        ''' Perform initializations '''
        self.validate(sim_pars=sim_pars)  # First, check that essential-to-match parameters match
        self.set_pars(sim_pars)  # Replace the saved parameters with this simulation's
        self.initialized = True
        return

    def update_states_pre(self, t, resfreq=None):
        ''' Perform all state updates at the current timestep '''

        # Initialize
        self.t = t
        self.dt = self.pars['dt']
        self.resfreq = resfreq if resfreq is not None else 1

        # Perform updates that are not genotype-specific
        if t % self.resfreq == 0:

            self.init_flows()  # Only reinitialize flows to zero every nth step, where n is the requested result frequency


            # Let cells die here
            hpv_cell_death, norm_cell_death = self.check_death()

            # # EVENT DRIVEN PROCESS HERE
            # # Let cells become infected, differentiated

            # Filtering for certain types of cells here

            active_basals = self.cells.filter((self.cells.is_basal() == True) & (self.cells.is_alive() == True) & (
                        self.cells.is_infected() == False))  # check alive or dead attribute

            active_infected_basals = self.cells.filter(
                (self.cells.is_basal() == True) & (self.cells.is_alive() == True) & (self.cells.is_infected() == True))

            active_parabasals = self.cells.filter(
                (self.cells.is_parabasal() == True) & (self.cells.is_alive() == True) & (
                            self.cells.is_infected() == False))

            active_infected_parabasals = self.cells.filter(
                (self.cells.is_parabasal() == True) & (self.cells.is_alive() == True) & (
                            self.cells.is_infected() == True))

            # Pass these values to the draw event classes TODO search for max rate

            # Choose events for the basal cells
            b_events = self.draw_event_class_basal_normal(active_basals)
            active_basal_events = self.draw_events(max_rate, b_events, active_basals)

            # Choose events for the infected basal cells
            i_b_events = self.draw_event_class_basal_infect(active_infected_basals)
            active_infected_basal_events = self.draw_events(max_rate, i_b_events, active_infected_basals)

            # Choose events for the parabasal cells
            p_events = self.draw_event_class_parabasal_normal(active_parabasals)
            active_parabasal_events = self.draw_events(max_rate, p_events, active_parabasals)

            # Choose events for the infected parabasal cells
            i_p_events = self.draw_event_class_parabasal_infected(active_infected_parabasals)
            active_infected_parabasal_events = self.draw_events(max_rate, i_p_events, active_infected_parabasals)

            # Make events happen randomly

            count = 0
            total_event_count = len(active_infected_parabasal_events) + len(active_infected_basal_events) + len(
                active_basal_events) + len(active_parabasal_events)

            #Use self.addtoself(new_cells) to add on the new cells to the arrayification
            while count < total_event_count:

                if len(active_basal_events) != 0:
                    # Active basals
                    pair = next(iter((active_basal_events.items())))

                    if pair[1] == "infect":
                        self.cells.infect(pair[0])
                    else:  # encapsulates the symmetric and asymmetric splits.
                        cell_to_make = self.cells.split(pair[0], pair[1])
                        self.addtoself(cell_to_make)

                    del pair[pair[0]]
                    count += 1

                if len(active_infected_basal_events) != 0:
                    # Active infected basals
                    pair = next(iter((active_infected_basal_events.items())))

                    if pair[1] == "transform":
                        self.cells.transform(pair[0])  # TODO check on the genotypes at play here
                    else:  # encapsulates the symmetric and asymmetric splits
                        cells_to_make = self.cells.split(pair[0], pair[1])
                        self.addtoself(cells_to_make)

                    del pair[pair[0]]
                    count += 1

                if len(active_parabasal_events) != 0:
                    # Active parabasals
                    pair = next(iter((active_parabasal_events.items())))
                    if pair[1] == "differentiate":
                        self.cells.differentiate(pair[0])
                    else:  # takes care of symmetric split
                        cells_to_make = self.cells.split(pair[0], pair[1])
                        self.addtoself(cells_to_make)

                    del pair[pair[0]]
                    count += 1

                if len(active_infected_parabasal_events) != 0:
                    # Active infected parabasals
                    pair = next(iter((active_infected_parabasal_events.items())))
                    if pair[1] == "differentiate infect":
                        self.cells.differentiate(pair[0])
                    else:
                        cells_to_make = self.cells.split(pair[0], pair[1])
                        self.addtoself(cells_to_make)

                    del pair[pair[0]]
                    count += 1

        #combine all the splits to make a bunch of new cells in one clump?



        # # Perform updates that are genotype-specific
        ng = self.pars['n_genotypes']
        for g in range(ng):
            self.flows['new_cin1s'][g] += self.check_cin1(g)
            self.flows['new_cin2s'][g] += self.check_cin2(g)
            self.flows['new_cin3s'][g] += self.check_cin3(g)
            if t % self.resfreq == 0:
                self.flows['new_cins'][g] += self.flows['new_cin1s'][g] + self.flows['new_cin2s'][g] + \
                                             self.flows['new_cin3s'][g]
            self.flows['new_cancers'][g] += self.check_cancer(g)
            self.flows['new_cancer_deaths'][g] += self.check_cancer_deaths(g)
            self.check_clearance(g)

        # Create total flows
        # self.total_flows['new_total_cin1s'] += self.flows['new_cin1s'].sum()
        # self.total_flows['new_total_cin2s'] += self.flows['new_cin2s'].sum()
        # self.total_flows['new_total_cin3s'] += self.flows['new_cin3s'].sum()
        # self.total_flows['new_total_cins'] += self.flows['new_cins'].sum()
        # self.total_flows['new_total_cancers'] += self.flows['new_cancers'].sum()
        # self.total_flows['new_total_cancer_deaths'] += self.flows['new_cancer_deaths'].sum()

        self.total_flows['total_infections'] += self.flows['infections'].sum()
        self.total_flows['total_basals'] += self.flows['basal'].sum()
        self.total_flows['total_parabasal'] += self.flows['parabasal'].sum()
        self.total_flows['total_viral_load'] += self.flows['viral_load'].sum()
        self.total_flows['total_transformed'] += self.flows['transformed'].sum()
        self.total_flows['total_dead'] += self.flows['dead'].sum()



        return new_cells

    # # %% Methods for updating partnerships
    # def dissolve_partnerships(self, t=None):
    #     ''' Dissolve partnerships '''
    #
    #     n_dissolved = dict()
    #
    #     for lno, lkey in enumerate(self.layer_keys()):
    #         dissolve_inds = cellUtil.true(
    #             self.t * self.pars['dt'] > self.contacts[lkey]['end'])  # Get the partnerships due to end
    #         dissolved = self.contacts[lkey].pop_inds(dissolve_inds)  # Remove them from the contacts list
    #
    #         # Update current number of partners
    #         unique, counts = cellUtil.unique(np.concatenate([dissolved['f'], dissolved['m']]))
    #         self.current_partners[lno, unique] -= counts
    #         n_dissolved[lkey] = len(dissolve_inds)
    #
    #     return n_dissolved  # Return the number of dissolved partnerships by layer

    # def create_partnerships(self, t=None, n_new=None, pref_weight=100):
    #     ''' Create new partnerships '''
    #
    #     new_pships = dict()
    #     for lno, lkey in enumerate(self.layer_keys()):
    #         new_pships[lkey] = dict()
    #
    #         # Define probabilities of entering new partnerships
    #         new_pship_probs = np.ones(
    #             len(self))  # Begin by assigning everyone equal probability of forming a new relationship
    #         new_pship_probs[~self.is_active] *= 0  # Blank out people not yet active
    #         underpartnered = cellUtil.true(self.current_partners[lno, :] < self.partners[lno,
    #                                                                   :])  # Indices of those who have fewer partners than desired
    #         new_pship_probs[underpartnered] *= pref_weight  # Increase weight for those who are underpartnerned
    #
    #         # Draw female and male partners separately
    #         new_pship_inds_f = cellUtil.choose_w(probs=new_pship_probs * self.is_female, n=n_new[lkey], unique=True)
    #         new_pship_inds_m = cellUtil.choose_w(probs=new_pship_probs * self.is_male, n=n_new[lkey], unique=True)
    #         new_pship_inds = np.concatenate([new_pship_inds_f, new_pship_inds_m])
    #         self.current_partners[lno, new_pship_inds] += 1
    #
    #         # Sort the new contacts by age so partners are roughly the same age
    #         sorted_f_inds = self.age[new_pship_inds_f].argsort()
    #         new_pship_inds_f = new_pship_inds_f[sorted_f_inds]
    #         sorted_m_inds = self.age[new_pship_inds_m].argsort()
    #         new_pship_inds_m = new_pship_inds_m[sorted_m_inds]
    #
    #         # Add everything to a contacts dictionary
    #         new_pships[lkey]['f'] = new_pship_inds_f
    #         new_pships[lkey]['m'] = new_pship_inds_m
    #         new_pships[lkey]['dur'] = cellUtil.sample(**self['pars']['dur_pship'][lkey], size=n_new[lkey])
    #         new_pships[lkey]['start'] = np.array([t * self['pars']['dt']] * n_new[lkey], dtype=cellDef.default_float)
    #         new_pships[lkey]['end'] = new_pships[lkey]['start'] + new_pships[lkey]['dur']
    #         new_pships[lkey]['acts'] = cellUtil.sample(**self['pars']['acts'][lkey], size=n_new[
    #             lkey])  # Acts per year for this pair, assumed constant over the duration of the partnership
    #
    #     self.add_contacts(new_pships)
    #
    #     return

    # %% Methods for updating state (updating the flows)
    def check_inds(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is false and which meet the date criterion '''
        if filter_inds is None:
            not_current = cellUtil.false(current)
        else:
            not_current = cellUtil.ifalsei(current, filter_inds)
        has_date = cellUtil.idefinedi(date, not_current)
        inds = cellUtil.itrue(self.t >= date[has_date], has_date)
        return inds

    def check_inds_true(self, current, date, filter_inds=None):
        ''' Return indices for which the current state is true and which meet the date criterion '''
        if filter_inds is None:
            current_inds = cellUtil.true(current)
        else:
            current_inds = cellUtil.itruei(current, filter_inds)
        has_date = cellUtil.idefinedi(date, current_inds)
        inds = cellUtil.itrue(self.t >= date[has_date], has_date)
        return inds








    # def check_cin2(self, genotype):
    #     ''' Check for new progressions to CIN2 '''
    #     filter_inds = self.true_by_genotype('cin1', genotype)
    #     inds = self.check_inds(self.cin2[genotype, :], self.date_cin2[genotype, :], filter_inds=filter_inds)
    #     self.cin2[genotype, inds] = True
    #     self.cin1[genotype, inds] = False  # No longer counted as CIN1
    #     return len(inds)

    # def check_cin3(self, genotype):
    #     ''' Check for new progressions to CIN3 '''
    #     filter_inds = self.true_by_genotype('cin2', genotype)
    #     inds = self.check_inds(self.cin3[genotype, :], self.date_cin3[genotype, :], filter_inds=filter_inds)
    #     self.cin3[genotype, inds] = True
    #     self.cin2[genotype, inds] = False  # No longer counted as CIN2
    #     return len(inds)

    # def check_cancer(self, genotype):
    #     '''
    #     Check for new progressions to cancer
    #     Once an individual has cancer they are no longer susceptible to new HPV infections or CINs and no longer infectious
    #     '''
    #     filter_inds = self.true_by_genotype('cin3', genotype)
    #     inds = self.check_inds(self.cancerous[genotype, :], self.date_cancerous[genotype, :], filter_inds=filter_inds)
    #     self.cancerous[genotype, inds] = True
    #     self.cin1[:,
    #     inds] = False  # No longer counted as CIN1 for this genotype.
    #     self.cin2[:, inds] = False  # No longer counted as CIN2
    #     self.cin3[:, inds] = False  # No longer counted as CIN3
    #     self.susceptible[:, inds] = False  #
    #     self.infectious[:, inds] = False  #
    #     return len(inds)

    # def check_cancer_deaths(self, genotype):
    #     '''
    #     Check for new progressions to cancer
    #     Once an individual has cancer they are no longer susceptible to new HPV infections or CINs and no longer infectious
    #     '''
    #     filter_inds = self.true_by_genotype('cancerous', genotype)
    #     inds = self.check_inds(self.dead_cancer[genotype, :], self.date_dead_cancer[genotype, :],
    #                            filter_inds=filter_inds)
    #     self.make_die(inds, genotype=genotype, cause='cancer')
    #     return len(inds)




    def check_death(self):
        '''
        Check for new deaths
        '''
        filter_inds = self.true('alive')
        inds = self.check_inds(self.differentiated, filter_inds=filter_inds)
        deaths_infected = len(cellUtil.true(self.is_infected[inds]))
        deaths_normal = len(inds) - deaths_infected
        return deaths_infected, deaths_normal
    # def check_clearance(self, genotype):
    #     '''
    #     Check for HPV clearance.
    #     '''
    #     filter_inds = self.true_by_genotype('infectious', genotype)
    #     inds = self.check_inds_true(self.infectious[genotype, :], self.date_clearance[genotype, :],
    #                                 filter_inds=filter_inds)
    #
    #     # Now reset disease states
    #     self.susceptible[genotype, inds] = True
    #     self.infectious[genotype, inds] = False
    #     self.cin1[genotype, inds] = False
    #     self.cin2[genotype, inds] = False
    #     self.cin3[genotype, inds] = False



        # return

    # def apply_death_rates(self):
    #     '''
    #     Apply death rates to remove people from the population
    #     NB people are not actually removed to avoid issues with indices
    #     '''
    #
    #     # Get age-dependent death rates.
    #     death_pars = self.pars['death_rates']
    #     age_inds = np.digitize(self.age, death_pars['f'][:, 0]) - 1
    #     death_probs = np.full(len(self), np.nan, dtype=cellDef.default_float)
    #     death_probs[self.f_inds] = death_pars['f'][age_inds[self.f_inds], 2] * self.dt
    #     death_probs[self.m_inds] = death_pars['m'][age_inds[self.m_inds], 2] * self.dt
    #
    #     # Get indices of people who die of other causes, removing anyone already dead
    #     death_inds = cellUtil.true(cellUtil.binomial_arr(death_probs))
    #     already_dead = self.dead_other[death_inds]
    #     death_inds = death_inds[~already_dead]  # Unique indices in deaths that are not already dead
    #
    #     deaths_female = len(cellUtil.true(self.is_female[death_inds]))
    #     deaths_male = len(cellUtil.true(self.is_male[death_inds]))
    #     # Apply deaths
    #     new_other_deaths = self.make_die(death_inds, cause='other')
    #     return new_other_deaths, deaths_female, deaths_male
    #
    # def add_births(self):
    #     ''' Method to add births '''
    #     this_birth_rate = sc.smoothinterp(self.t + self.pars['start'], self.pars['birth_rates'][0],
    #                                       self.pars['birth_rates'][1]) * self.dt
    #     new_births = round(this_birth_rate[0] * len(self) / 1000)  # Crude births per 1000
    #
    #     # Generate other characteristics of the new people
    #     uids, sexes, debuts, partners = cellPop.set_static(new_n=new_births, existing_n=len(self), pars=self.pars)
    #     pars = {
    #         'pop_size': new_births,
    #         'n_genotypes': self.pars['n_genotypes'],
    #         'n_partner_types': self.pars['n_partner_types']
    #     }
    #     new_people = Cells(pars=pars, uid=uids, age=np.zeros(new_births), sex=sexes, debut=debuts, partners=partners,
    #                         strict=False)
    #
    #     return new_births, new_people
    #
    # # %% Methods to make events occur (death, infection, others TBC)
    # def make_naive(self, inds):
    #     '''
    #     Make a set of people naive. This is used during dynamic resampling.
    #
    #     Args:
    #         inds (array): list of people to make naive
    #     '''
    #     for key in self.meta.states:
    #         if key in ['susceptible']:
    #             self[key][:, inds] = True
    #         elif key in ['other_dead']:
    #             self[key][inds] = False
    #         else:
    #             self[key][:, inds] = False
    #
    #     # Reset immunity
    #     for key in self.meta.imm_by_source_states:
    #         self[key][:, inds] = 0
    #
    #     # Reset dates
    #     for key in self.meta.dates + self.meta.durs:
    #         self[key][:, inds] = np.nan
    #
    #     return

    def infect(self, inds, genotypes = None):
        '''
        Infect basal cell in the array, only need to change the status of the cell and give the proper attributes for infection

        Args:
            inds        (int): Index of basal cell to be infected
            genotypes   (array): array of genotypes to pick from for infection

        Returns:
            None

        '''

        self.infected[genotypes, inds] = True # TODO access the genotype for that cell


    def transform(self, inds, genotypes = None):
        '''
        Transform infected cell in the array

        Args:
            inds (int): Index of infected cell to be transformed

        Return:
            None
        '''

        self.transformed[genotypes, inds] = True

    def differentiate(self, inds, genotypes = None):
        '''
        Differentiate a parabasal cell in the array. This sets the time and the dispersal of viral load as well.

        Args:
            inds (int): Index of parabasal cell to be differentiated

        Return:
            None
        '''

        self.differentiated[genotypes, inds] = True

    def split(self, ind_orignal, type): #TODO get the single or double cell creation going
        '''
        Split a cell in the array for cell division

        Args:
            ind_orignal (int): Index of cell to be divided'
            type        (string): String of which type of split is occurring

        Return:
            None
        '''
        # assign a new cell to the array that has all the same characteristics of the original

        if (type == "symmetric BB") or (type == "symmetric infect BB"):
            #make a new basal cell

        elif (type == "symmetric PP") or (type == "symmetric infect PP"):
            #make a new parabasal cell and possibly another one if originally came from a basal
            if ind_orignal.is_infected():
                #set the new cell to be infected as well
            else:
                return Cells()

            if ind_orignal.is_basal():
                #make the original basal cell dead and make another parabasal cell
                if ind_orignal.is_infected():
                    # set the new cell to be infected as well

        elif (type == "asymmetric BP") or (type == "asymmetric infect BP"):
            # make a parabasal cell
            return divided_cell

            if ind_orignal.is_infected():
                # set the new cell to be infected as well



    # def infect(self, inds, genotypes=None):
    #     '''
    #     Infect basal cell in the array
    #
    #     Args:
    #         inds      (array): array of people to infect
    #         genotypes (array): array of genotypes to infect people with
    #         source    (array): source indices of the people who transmitted this infection (None if an importation or seed infection)
    #         offset    (array): if provided, the infections will occur at the timepoint self.t+offset
    #         dur_inf   (array): if provided, the duration of the infections
    #         layer     (str):   contact layer this infection was transmitted on
    #
    #     Returns:
    #         count (int): number of people infected
    #     '''
    #
    #     if len(inds) == 0:
    #         return 0
    #
    #     dt = self.pars['dt']
    #
    #     # Deal with genotype parameters
    #     ng = self.pars['n_genotypes']
    #     genotype_keys = ['rel_cin1_prob', 'rel_cin2_prob', 'rel_cin3_prob', 'rel_cancer_prob', 'rel_death_prob']
    #     genotype_pars = self.pars['genotype_pars']
    #     genotype_map = self.pars['genotype_map']
    #     durpars = self.pars['dur']
    #     progpars = self.pars['prognoses']
    #     progprobs = [{k: self.pars[k] * genotype_pars[genotype_map[g]][k] for k in genotype_keys} for g in range(
    #         ng)]  # np.array([[self.pars[k] * genotype_pars[genotype_map[g]][k] for k in genotype_keys] for g in range(ng)])
    #
    #     # Set all dates
    #     base_t = self.t + offset if offset is not None else self.t
    #     self.date_infectious[genotypes, inds] = base_t
    #
    #     # Count reinfections
    #     for g in range(ng):
    #         self.flows['new_reinfections'][g] += len(
    #             (~np.isnan(self.date_clearance[g, inds[genotypes == g]])).nonzero()[-1])
    #     self.total_flows['new_total_reinfections'] += len(
    #         (~np.isnan(self.date_clearance[genotypes, inds])).nonzero()[-1])
    #     for key in ['date_clearance']:
    #         self[key][genotypes, inds] = np.nan
    #
    #     # Update states, genotype info, and flows
    #     new_total_infections = len(inds)  # Count the total number of new infections
    #     new_infections = np.array([len((genotypes == g).nonzero()[0]) for g in range(ng)],
    #                               dtype=np.float64)  # Count the number by genotype
    #     self.susceptible[genotypes, inds] = False  # Adjust states - set susceptible to false
    #     self.infectious[genotypes, inds] = True  # Adjust states - set infectious to true
    #
    #     # Add to flow results. Note, we only count these infectious in the results if they happened at this timestep
    #     if offset is None:
    #         # Create overall flows
    #         self.total_flows[
    #             'new_total_infections'] += new_total_infections  # Add the total count to the total flow data
    #         self.flows['new_infections'] += new_infections  # Add the count by genotype to the flow data
    #
    #         # Create by-age flows
    #         for g in range(ng):
    #             age_inds, infections = cellUtil.unique(self.age_brackets[inds[genotypes == g]])
    #             self.flows_by_age['new_infections_by_age'][age_inds - 1, g] += infections
    #         total_age_inds, total_infections = cellUtil.unique(self.age_brackets[inds])
    #         self.total_flows_by_age['new_total_infections_by_age'][total_age_inds - 1] += total_infections
    #
    #         # Create by-sex flows
    #         infs_female = len(cellUtil.true(self.is_female[inds]))
    #         infs_male = len(cellUtil.true(self.is_male[inds]))
    #         self.flows_by_sex['new_total_infections_by_sex'][0] += infs_female
    #         self.flows_by_sex['new_total_infections_by_sex'][1] += infs_male
    #
    #     # Determine the duration of the HPV infection without any dysplasia
    #     if dur is None:
    #         dur = cellUtil.sample(**durpars['none'], size=len(inds))  # Duration of infection without dysplasia in years
    #     else:
    #         if len(dur) != len(inds):
    #             errormsg = f'If supplying durations of infections, they must be the same length as inds: {len(dur)} vs. {len(inds)}.'
    #             raise ValueError(errormsg)
    #     dur_inds = np.digitize(dur, progpars['duration_cutoffs']) - 1  # Convert durations to indices
    #     self.dur_hpv[
    #         genotypes, inds] = dur  # Set the initial duration of infection as the length of the period without dysplasia - this is then extended for those who progress
    #
    #     # Use genotype-specific prognosis probabilities to determine what happens.
    #     # Only women can progress beyond infection.
    #     f_inds = self.is_female[inds]
    #
    #     for g in range(ng):
    #         # Apply filters so we only select females with this genotype
    #         filters = f_inds * (genotypes == g)
    #
    #         # Use prognosis probabilities to determine whether HPV clears or progresses to CIN1
    #         cin1_probs = progprobs[g]['rel_cin1_prob'] * progpars['cin1_probs'][dur_inds] * filters
    #         is_cin1 = cellUtil.binomial_arr(cin1_probs)
    #         cin1_inds = inds[is_cin1]
    #         no_cin1_inds = inds[~is_cin1]
    #
    #         # CASE 1: Infection clears without causing dysplasia
    #         self.date_clearance[g, no_cin1_inds] = self.date_infectious[g, no_cin1_inds] + np.ceil(self.dur_hpv[
    #                                                                                                    g, no_cin1_inds] / dt)  # Date they clear HPV infection (interpreted as the timestep on which they recover)
    #
    #         # CASE 2: Infection progresses to mild dysplasia (CIN1)
    #         self.dur_none2cin1[g, cin1_inds] = dur[is_cin1]  # Store the length of time before progressing
    #         excl_inds = cellUtil.true(
    #             self.date_cin1[g, cin1_inds] < self.t)  # Don't count CIN1s that were acquired before now
    #         self.date_cin1[g, cin1_inds[excl_inds]] = np.nan
    #         self.date_cin1[g, cin1_inds] = np.fmin(self.date_cin1[g, cin1_inds],
    #                                                self.date_infectious[g, cin1_inds] + np.ceil(self.dur_hpv[
    #                                                                                                 g, cin1_inds] / dt))  # Date they develop CIN1 - minimum of the date from their new infection and any previous date
    #         dur_cin1 = cellUtil.sample(**durpars['cin1'], size=len(cin1_inds))
    #         dur_cin1_inds = np.digitize(dur_cin1, progpars['duration_cutoffs']) - 1  # Convert durations to indices
    #
    #         # Determine whether CIN1 clears or progresses to CIN2
    #         cin2_probs = progprobs[g]['rel_cin2_prob'] * progpars['cin2_probs'][dur_cin1_inds]
    #         is_cin2 = cellUtil.binomial_arr(cin2_probs)
    #         cin2_inds = cin1_inds[is_cin2]
    #         no_cin2_inds = cin1_inds[~is_cin2]
    #
    #         # CASE 2.1: Mild dysplasia regresses and infection clears
    #         self.date_clearance[g, no_cin2_inds] = np.fmax(self.date_clearance[g, no_cin2_inds],
    #                                                        self.date_cin1[g, no_cin2_inds] + np.ceil(
    #                                                            dur_cin1[~is_cin2] / dt))
    #         self.dur_hpv[
    #             g, cin1_inds] += dur_cin1  # Duration of HPV is the sum of the period without dysplasia and the period with CIN1
    #
    #         # CASE 2.2: Mild dysplasia progresses to moderate (CIN1 to CIN2)
    #         self.dur_cin12cin2[g, cin2_inds] = dur_cin1[is_cin2]
    #         excl_inds = cellUtil.true(
    #             self.date_cin2[g, cin2_inds] < self.t)  # Don't count CIN2s that were acquired before now
    #         self.date_cin2[g, cin2_inds[excl_inds]] = np.nan
    #         self.date_cin2[g, cin2_inds] = np.fmin(self.date_cin2[g, cin2_inds], self.date_cin1[g, cin2_inds] + np.ceil(
    #             dur_cin1[
    #                 is_cin2] / dt))  # Date they get CIN2 - minimum of any previous date and the date from the current infection
    #         dur_cin2 = cellUtil.sample(**durpars['cin2'], size=len(cin2_inds))
    #         dur_cin2_inds = np.digitize(dur_cin2, progpars['duration_cutoffs']) - 1  # Convert durations to indices
    #
    #         # Determine whether CIN2 clears or progresses to CIN3
    #         cin3_probs = progprobs[g]['rel_cin3_prob'] * progpars['cin3_probs'][dur_cin2_inds]
    #         is_cin3 = cellUtil.binomial_arr(cin3_probs)
    #         no_cin3_inds = cin2_inds[~is_cin3]
    #         cin3_inds = cin2_inds[is_cin3]
    #
    #         # CASE 2.2.1: Moderate dysplasia regresses and the virus clears
    #         self.date_clearance[g, no_cin3_inds] = np.fmax(self.date_clearance[g, no_cin3_inds],
    #                                                        self.date_cin2[g, no_cin3_inds] + np.ceil(
    #                                                            dur_cin2[~is_cin3] / dt))  # Date they clear CIN2
    #         self.dur_hpv[
    #             g, cin2_inds] += dur_cin2  # Duration of HPV is the sum of the period without dysplasia and the period with CIN
    #
    #         # Case 2.2.2: CIN2 with progression to CIN3
    #         self.dur_cin22cin3[g, cin3_inds] = dur_cin2[is_cin3]
    #         excl_inds = cellUtil.true(
    #             self.date_cin3[g, cin3_inds] < self.t)  # Don't count CIN2s that were acquired before now
    #         self.date_cin3[g, cin3_inds[excl_inds]] = np.nan
    #         self.date_cin3[g, cin3_inds] = np.fmin(self.date_cin3[g, cin3_inds], self.date_cin2[g, cin3_inds] + np.ceil(
    #             dur_cin2[
    #                 is_cin3] / dt))  # Date they get CIN3 - minimum of any previous date and the date from the current infection
    #         dur_cin3 = cellUtil.sample(**durpars['cin3'], size=len(cin3_inds))
    #         dur_cin3_inds = np.digitize(dur_cin3, progpars['duration_cutoffs']) - 1  # Convert durations to indices
    #
    #         # Use prognosis probabilities to determine whether CIN3 clears or progresses to CIN2
    #         cancer_probs = progprobs[g]['rel_cancer_prob'] * progpars['cancer_probs'][dur_cin3_inds]
    #         is_cancer = cellUtil.binomial_arr(cancer_probs)  # See if they develop cancer
    #         cancer_inds = cin3_inds[is_cancer]
    #
    #         # Cases 2.2.2.1 and 2.2.2.2: HPV DNA is no longer present, either because it's integrated (& progression to cancer will follow) or because the infection clears naturally
    #         self.date_clearance[g, cin3_inds] = np.fmax(self.date_clearance[g, cin3_inds],
    #                                                     self.date_cin3[g, cin3_inds] + np.ceil(
    #                                                         dur_cin3 / dt))  # HPV is cleared
    #         self.dur_hpv[
    #             g, cin3_inds] += dur_cin3  # Duration of HPV is the sum of the period without dysplasia and the period with CIN
    #
    #         # Case 2.2.2.1: Severe dysplasia regresses
    #         self.dur_cin2cancer[g, cancer_inds] = dur_cin3[is_cancer]
    #         excl_inds = cellUtil.true(
    #             self.date_cancerous[g, cancer_inds] < self.t)  # Don't count cancers that were acquired before now
    #         self.date_cancerous[g, cancer_inds[excl_inds]] = np.nan
    #         self.date_cancerous[g, cancer_inds] = np.fmin(self.date_cancerous[g, cancer_inds],
    #                                                       self.date_cin3[g, cancer_inds] + np.ceil(dur_cin3[
    #                                                                                                    is_cancer] / dt))  # Date they get cancer - minimum of any previous date and the date from the current infection
    #
    #         # Record eventual deaths from cancer (NB, assuming no survival without treatment)
    #         dur_cancer = cellUtil.sample(**durpars['cancer'], size=len(cancer_inds))
    #         self.date_dead_cancer[g, cancer_inds] = self.date_cancerous[g, cancer_inds] + np.ceil(dur_cancer / dt)
    #
    #     return new_infections  # For incrementing counters

    # def make_die(self, inds, genotype=None, cause=None):
    #     ''' Make people die of all other causes (background mortality) '''
    #
    #     if cause == 'other':
    #         self.dead_other[inds] = True
    #     elif cause == 'cancer':
    #         self.dead_cancer[genotype, inds] = True
    #     else:
    #         errormsg = f'Cause of death must be one of "other" or "cancer", not {cause}.'
    #         raise ValueError(errormsg)
    #
    #     self.susceptible[:, inds] = False
    #     self.infectious[:, inds] = False
    #     self.cin1[:, inds] = False
    #     self.cin2[:, inds] = False
    #     self.cin3[:, inds] = False
    #     self.cancerous[:, inds] = False
    #
    #     # Remove dead people from contact network by setting the end date of any partnership they're in to now
    #     for contacts in self.contacts.values():
    #         m_inds = cellUtil.findinds(contacts['m'], inds)
    #         f_inds = cellUtil.findinds(contacts['f'], inds)
    #         pships_to_end = contacts.pop_inds(np.concatenate([f_inds, m_inds]))
    #         pships_to_end['end'] *= 0 + self.t  # Reset end date to now
    #         contacts.append(pships_to_end)
    #
    #     return len(inds)

    # %% Analysis methods

    def plot(self, *args, **kwargs):
        '''
        Plot statistics of the population -- age distribution, numbers of contacts,
        and overall weight of contacts (number of contacts multiplied by beta per
        layer).

        Args:
            bins      (arr)   : age bins to use (default, 0-100 in one-year bins)
            width     (float) : bar width
            font_size (float) : size of font
            alpha     (float) : transparency of the plots
            fig_args  (dict)  : passed to pl.figure()
            axis_args (dict)  : passed to pl.subplots_adjust()
            plot_args (dict)  : passed to pl.plot()
            do_show   (bool)  : whether to show the plot
            fig       (fig)   : handle of existing figure to plot into
        '''
        fig = cellPlt.plot_people(people=self, *args, **kwargs)
        return fig

    def story(self, uid, *args):
        '''
        Print out a short history of events in the life of the specified individual.

        Args:
            uid (int/list): the person or people whose story is being regaled
            args (list): these people will tell their stories too

        **Example**::

            sim = cv.Sim(pop_type='hybrid', verbose=0)
            sim.run()
            sim.people.story(12)
            sim.people.story(795)
        '''

        def label_lkey(lkey):
            ''' Friendly name for common layer keys '''
            if lkey.lower() == 'a':
                llabel = 'default contact'
            if lkey.lower() == 'r':
                llabel = 'regular'
            elif lkey.lower() == 'c':
                llabel = 'casual'
            else:
                llabel = f'"{lkey}"'
            return llabel

        uids = sc.promotetolist(uid)
        uids.extend(args)

        for uid in uids:

            p = self[uid]
            sex = 'female' if p.sex == 0 else 'male'

            intro = f'\nThis is the story of {uid}, a {p.age:.0f} year old {sex}.'
            intro += f'\n{uid} became sexually active at age {p.debut:.0f}.'
            if not p.susceptible:
                if ~np.isnan(p.date_infectious):
                    print(f'{intro}\n{uid} contracted HPV on timestep {p.date_infectious} of the simulation.')
                else:
                    print(f'{intro}\n{uid} did not contract HPV during the simulation.')

            total_contacts = 0
            no_contacts = []
            for lkey in p.contacts.keys():
                llabel = label_lkey(lkey)
                n_contacts = len(p.contacts[lkey])
                total_contacts += n_contacts
                if n_contacts:
                    print(f'{uid} is connected to {n_contacts} people in the {llabel} layer')
                else:
                    no_contacts.append(llabel)
            if len(no_contacts):
                nc_string = ', '.join(no_contacts)
                print(f'{uid} has no contacts in the {nc_string} layer(s)')
            print(f'{uid} has {total_contacts} contacts in total')

            events = []

            dates = {
                'date_HPV_clearance': 'HPV cleared',
            }

            for attribute, message in dates.items():
                date = getattr(p, attribute)
                if not np.isnan(date):
                    events.append((date, message))

            # for infection in self.infection_log:
            #     lkey = infection['layer']
            #     llabel = label_lkey(lkey)
            #     if infection['target'] == uid:
            #         if lkey:
            #             events.append((infection['date'], f'was infected with COVID by {infection["source"]} via the {llabel} layer'))
            #         else:
            #             events.append((infection['date'], 'was infected with COVID as a seed infection'))

            #     if infection['source'] == uid:
            #         x = len([a for a in self.infection_log if a['source'] == infection['target']])
            #         events.append((infection['date'],f'gave COVID to {infection["target"]} via the {llabel} layer ({x} secondary infections)'))

            if len(events):
                for timestep, event in sorted(events, key=lambda x: x[0]):
                    print(f'On timestep {timestep:.0f}, {uid} {event}')
            else:
                print(f'Nothing happened to {uid} during the simulation.')
        return

__all__ = ['genotype']


class genotype(sc.prettyobj):
    '''
    Add a new genotype to the sim
    Args:
        genotype (str): name of genotype
    **Example**::
        hpv16    = hp.genotype('16') # Make a sim with only HPV16
    '''

    def __init__(self, genotype):
        self.index     = None # Index of the genotype in the sim; set later
        self.label     = None # Genotype label (used as a dict key)
        self.p         = None # This is where the parameters will be stored
        self.parse(genotype=genotype) #
        self.initialized = False
        return


    def parse(self, genotype=None):
        ''' Unpack genotype information, which must be given as a string '''

        if isinstance(genotype, str):

            choices, mapping = cellPar.get_genotype_choices()
            known_genotype_pars = cellPar.get_genotype_pars()

            label = genotype.lower()

            if label in mapping:
                label = mapping[label]
                genotype_pars = known_genotype_pars[label]
            else:
                errormsg = f'The selected genotype "{genotype}" is not implemented; choices are:\n{sc.pp(choices, doprint=False)}'
                raise NotImplementedError(errormsg)


        else:
            errormsg = f'Could not understand {type(genotype)}, please specify as a predefined genotype:\n{sc.pp(choices, doprint=False)}'
            raise ValueError(errormsg)

        # Set label and parameters
        self.label = label
        self.p = genotype_pars

        return


    def initialize(self, sim):
        ''' Update genotype info in sim '''
        sim['genotype_pars'][self.label] = self.p  # Store the parameters
        self.index = list(sim['genotype_pars'].keys()).index(self.label) # Find where we are in the list
        sim['genotype_map'][self.index]  = self.label # Use that to populate the reverse mapping
        sim['imm_boost'] = list(sim['imm_boost']) + [self.p['imm_boost']]
        self.initialized = True
        return

