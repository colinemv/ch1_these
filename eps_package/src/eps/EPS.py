import pandas as pd
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from optimask import OptiMask

from eps.settings import *

@dataclass
class EPS:
    eps_to_compute: list = None
    eps_computed: list = None
    econo_computed: bool = True

    def __post_init__(self):
        self.eps_to_compute = [] if self.eps_to_compute is None else self.eps_to_compute        
        self.eps_computed = ['OCDE', 'BOD', 'GHG', 'GDP',  'BOD_3var', 'BOD_10var', 'BOD_50var',  'BOD_100var'] if self.eps_computed is None else self.eps_computed

    def __call__(self):
        if not self.econo_computed:
            self.load_databases()

            self.preprocess_databases()
            self.dc_EPS_total = {}
            self.compute_EPS()

            self.load_EPS_csv()
            self.create_EPS_total()
        
            self.create_econo_database()
            
        else:
            self.EPS_total = pd.read_csv(f"{Path(__file__).parents[2]}/data/EPS_total.csv")
            self.econo = pd.read_csv(f"{Path(__file__).parents[2]}/data/Econo.csv", index_col=0)

        self.dc_results = {}

    def load_databases(self):
        """Load all databases."""
        self._load_init_eps()
        self._load_gdp()
        self._load_pop()
        self._load_ghg()
        self._load_hdi()
        # self._load_rnd()

    def _load_init_eps(self):
        """Load and process EPS csv."""
        self.eps_init = (
            pd.concat(
                [
                    pd.read_csv(f"{Path(__file__).parents[2]}/inputs/EPS_{sector}.csv").assign(sector=sector)
                    for sector in sectors
                ]
            )
            .drop(col_to_drop, axis=1)
            .rename(columns=col_to_rename)
            .assign(EPS_value=lambda df: df.EPS_value.fillna(0))
            .assign(pol_code=lambda df: df['pol_code'].str.replace('LEV3_', ''))
            .assign(categorie=lambda df: np.where(df['pol_code'].isin(mbi_values), 'mbi', np.where(df['pol_code'].isin(nmbi_values), 'nmbi','nan')))
            .assign(sector_category=lambda df: df.apply(lambda row: f"{row.sector}_{row.categorie}", axis=1))
            .assign(
                ETS_E=lambda df: ((df['pol_code'] == 'ETS_E') & (df['EPS_value'] > 0)).astype(int),
                ETS_I=lambda df: ((df['pol_code'] == 'ETS_I') & (df['EPS_value'] > 0)).astype(int),
                ETS_B=lambda df: ((df['pol_code'] == 'ETS_B') & (df['EPS_value'] > 0)).astype(int),
                ETS_T=lambda df: ((df['pol_code'] == 'ETS_T') & (df['EPS_value'] > 0)).astype(int),
                ETS=lambda df: (
                ((df['pol_code'].isin(['ETS_E', 'ETS_I', 'ETS_B', 'ETS_T'])) & (df['EPS_value'] > 0))
                ).astype(int)
            )
            
        )

    def _load_gdp(self):
        """Load and process GDP & GDP manufacture csv."""
        gdp_base = (
            pd.read_csv(f"{Path(__file__).parents[2]}/inputs/GDP_OCDE.csv")
            .drop(col_to_drop_GDP, axis=1)
            .rename(columns=col_to_rename_GDP)
        )

        gdp_manufacture = (
            pd.read_csv(f"{Path(__file__).parents[2]}/inputs/GDP_manufacture.csv")
            .drop(col_to_drop_GDP_manufacture, axis=1)
            .rename(columns=col_to_rename_GDP_manufacture)
        )

        GDP = pd.concat([gdp_base, gdp_manufacture], ignore_index=True)

        ## Renommer les secteurs dans la base GDP
        GDP['sector'] = GDP['GDP_type'].map(sector_mapping_GDP)


        # Créer un DataFrame vide pour stocker les nouvelles lignes
        new_rows = []

        for iso3 in list(GDP.iso3.unique()):
            df_iso3 = GDP[GDP['iso3'] == iso3]
            for year in df_iso3['Year'].unique():
                df_year = df_iso3[df_iso3['Year'] == year]
                if df_year.shape[0] > 0 and all(GDP_type in df_year['GDP_type'].unique() for GDP_type in ["Industry, including energy (ISIC rev4)", "Activités de fabrication"]):
                    GDP_value = df_year[df_year['GDP_type'] == "Industry, including energy (ISIC rev4)"]['GDP_value'].sum() - df_year[df_year['GDP_type'] == "Activités de fabrication"]['GDP_value'].sum()
                    new_rows.append({
                        'iso3': iso3,
                        'GDP_type': 'Electricity',
                        'Year': year,
                        'GDP_value': GDP_value,
                        'sector': 'elec'
                    })

        new_rows_df = pd.DataFrame(new_rows)
        self.GDP = pd.concat([GDP, new_rows_df], ignore_index=True).copy()

    def _load_pop(self):
        """Load and process population csv."""
        self.POP = (
            pd.read_csv(f"{Path(__file__).parents[2]}/inputs/POP_OCDE.csv")
            .drop(col_to_drop_POP, axis=1)
            .rename(columns=col_to_rename_POP) 
        )

    def _load_ghg(self):
        """Load and process ghg csv."""
        GHG = (
            pd.read_csv(f"{Path(__file__).parents[2]}/inputs/GHG_OCDE.csv")
            .drop(col_to_drop_GHG, axis=1)
            .rename(columns=col_to_rename_GHG)
        )

        ## Renommer les secteurs dans la base GHG
        GHG['sector'] = GHG['GHG_source'].map(sector_mapping_GHG)

        # Liste des iso3 à modifier
        iso3_to_modify = ['MEX', 'CRI', 'COL', 'CHL', 'KOR', 'ISR', 'ARG']

        # Créer un DataFrame vide pour stocker les nouvelles lignes
        new_rows = []

        for iso3 in iso3_to_modify:
            df_iso3 = GHG[GHG['iso3'] == iso3]
            for year in df_iso3['Year'].unique():
                df_year = df_iso3[df_iso3['Year'] == year]
                if all(sector in df_year['sector'].unique() for sector in ['transport', 'elec', 'indus']):
                    GHG_value = df_year[df_year['GHG_source'] == '1 - Energy']['GHG_value'].sum() - df_year[df_year['GHG_source'].isin(['1A2 - Manufacturing industries and construction', '1A3 - Transport', '1B - Fugitive Emissions from Fuels', '1A1 - Energy Industries'])]['GHG_value'].sum()
                    if GHG_value < 0:
                        GHG_value = np.nan
                    new_rows.append({
                        'iso3': iso3,
                        'GHG_source': '1A4 - Residential and other sectors',
                        'Year': year,
                        'Unit': 'Tonnes of CO2 equivalent',
                        'PowerCode': 'Thousands',
                        'Reference Period': '',
                        'GHG_value': GHG_value,
                        'sector': 'building'
                    })

        new_rows_df = pd.DataFrame(new_rows)
        self.GHG = pd.concat([GHG, new_rows_df], ignore_index=True).copy()

    def _load_hdi(self):
        """Load and process HDI csv."""
        hdi_proc = ((
            pd.read_csv(f"{Path(__file__).parents[2]}/inputs/WB2.csv")
            )
            .drop(col_to_drop_HDI, axis=1)
            .rename(columns=col_to_rename_HDI)
        )

        # Remove the part between brackets in column names
        hdi_proc.columns = [col.split(' [')[0] for col in hdi_proc.columns]
        columns_order = ['iso3', 'Year', 'HDI_type', 'HDI_value']

        self.HDI = (hdi_proc
            .melt(id_vars=['iso3', 'HDI_type'], var_name='Year', value_name='HDI_value')
            .dropna(subset=['iso3'])
            .reindex(columns=columns_order)
            .assign(HDI_value = lambda df: df.HDI_value.replace('..', 'nan'))
            .assign(HDI_value = lambda df: df.HDI_value.astype(float))
        )

    # def _load_rnd(self):
    #     """load and process rnd csv."""
    #     rnd_proc = ((
    #         pd.read_csv(f"{Path(__file__).parents[2]}/inputs/RandD_OCDE.csv")
    #         )
    #         .drop(col_to_drop_rnd, axis=1)
    #         .rename(columns=col_to_rename_rnd)
    #     )

    #     self.rnd = rnd_proc.copy()

    def preprocess_databases(self):
        self._preprocess_EPS_GHG()
        self._preprocess_EPS_GDP()

    def _preprocess_EPS_GHG(self):
        """Merge EPS_init and GHG databases."""
        #filtrer la base ghg : 
        relevant_sectors = ['building', 'elec', 'transport', 'indus']
        GHG_rs = self.GHG[self.GHG['sector'].isin(relevant_sectors)]

        ## Créer un df pour les weigths
        GHG_weight = (GHG_rs
            .groupby(['iso3', 'Year'])
            .apply(lambda df:
                (df
                    .groupby('sector')
                    .agg(GHG_value = ('GHG_value', 'sum'))
                    .assign(GHG_ratio = lambda df: df.GHG_value / df.GHG_value.sum())
                )
            )
            .reset_index()
            .sort_values(['iso3', 'Year', 'sector'])
        )
        self.GHG_weight = GHG_weight
        self.GHG_weight.to_csv(f"{Path(__file__).parents[2]}/data/GHG_weight.csv")
        ##Créer la df merged avec l'ensemble 
        self.EPS_GHG_merged = (self.eps_init
            .merge(GHG_weight, on=['iso3', 'Year', 'sector'], how='inner')
        )

    def _preprocess_EPS_GDP(self):
        """Merge EPS_init and GDP databases."""
        relevant_sectors = ['building', 'elec', 'transport', 'indus']
        GDP_rs = self.GDP[self.GDP['sector'].isin(relevant_sectors)]

        ## Créer un df pour les weigths
        GDP_weight = (GDP_rs
            .groupby(['iso3', 'Year'])
            .apply(lambda df:
                (df
                    .groupby('sector')
                    .agg(GDP_value = ('GDP_value', 'sum'))
                    .assign(GDP_ratio = lambda df: df.GDP_value / df.GDP_value.sum())
                )
            )
            .reset_index()
            .sort_values(['iso3', 'Year', 'sector'])
        )

        self.EPS_GDP_merged = (self.eps_init
            .merge(GDP_weight, on=['iso3', 'Year', 'sector'], how='inner')
        )

    def compute_EPS(self):
        dc_eps_to_func = {
            'OCDE': self._compute_EPS_OCDE,
            'BOD': self._compute_EPS_BOD,
            'GHG': self._compute_EPS_GHG,
            'GDP': self._compute_EPS_GDP, 
            'BOD_3var' : self._compute_EPS_BOD_3var,
            'BOD_10var' : self._compute_EPS_BOD_10var,
            'BOD_50var' : self._compute_EPS_BOD_50var,
            'BOD_100var' : self._compute_EPS_BOD_100var
            #'BOD_85var' : self._compute_EPS_BOD_85var
        }
        
        for eps in self.eps_to_compute:
            dc_eps_to_func[eps]()

    def _compute_EPS_OCDE(self):
        self.EPS_OCDE = (
            pd.concat([
                # EPS OCDE categorie level
                (self.eps_init
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'))
                    .assign(EPS_value = lambda df: df.EPS_value / 8)
                ),
                # EPS OCDE sector level
                (self.eps_init
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'))
                    .assign(EPS_value = lambda df: df.EPS_value / 8)
                    .groupby(['Year', 'iso3', 'sector'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS OCDE total
                (self.eps_init
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'))
                    .assign(EPS_value = lambda df: df.EPS_value / 8)
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(sector = 'total', categorie = 'total')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                (self.eps_init
                    .groupby(['Year', 'iso3', 'categorie'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'mean'))
                    .assign(EPS_value=lambda df: df.EPS_value / 8)
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_mbi', categorie='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                (self.eps_init
                    .groupby(['Year', 'iso3', 'categorie'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'mean'))
                    .assign(EPS_value=lambda df: df.EPS_value / 8)
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_nmbi', categorie='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                )
            ])
            .rename(columns={'EPS_value': 'EPS_OCDE'})
        )

        self.EPS_OCDE.to_csv(f"{Path(__file__).parents[2]}/data/EPS_OCDE.csv")
        self.dc_EPS_total['EPS_OCDE'] = self.EPS_OCDE

    def _compute_EPS_BOD(self):
        EPS_BOD = None
        for iso3 in tqdm(list(self.eps_init.iso3.unique())):
            for year in list(self.eps_init.Year.unique()):
                subset = (self.eps_init
                    .query(f"Year == {year} and iso3 == '{iso3}'")
                    .reset_index(drop=True)
                )

                if subset.shape[0] > 0:
                    
                    # Définissez la fonction objective à maximiser
                    def objective_function(w):
                        return -sum(w[i] * subset['EPS_value'].iloc[i] for i in range(len(w)))

                    # Contrainte : la somme des poids doit être égale à 1
                    def constraint_sum_to_one(w):
                        return sum(w) - 1

                    # Contrainte : les poids doivent être positifs
                    def constraint_positive_weights(w):
                        return w + 1e-20

                    # Contrainte supplémentaire : la somme des poids par secteur divisée par la somme totale des poids
                    # ne doit pas dépasser la part des émissions carbone de ce secteur dans les émissions totales
                    # Dans la fonction constraint_emissions_ratio, ajuster le calcul de la somme
                    def constraint_sector_sum(w):
                        result = []
                        for sector in self.eps_init['sector_category'].unique():
                            subset_sector = subset.query(f"sector_category == '{sector}'")
                            weight_index = list(subset_sector.index)
                            sum_weight_sector = sum(w[i] for i in weight_index)
                            result.append(sum_weight_sector - 1/8)
                        return result
                    
                    constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                                {'type': 'ineq', 'fun': constraint_positive_weights},
                                {'type': 'ineq', 'fun': constraint_sector_sum}]


                    # Définition des valeurs initiales
                    initial_guess = [1.0 / len(subset)] * len(subset)

                    # Résolution du problème d'optimisation
                    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

                    EPS_BOD = pd.concat([EPS_BOD, subset.assign(weight = result.x)])

        EPS_BOD_proc = (EPS_BOD
            .assign(EPS_value_weighted = lambda df: abs(df.EPS_value * df.weight).round(4))
            .groupby(['iso3', 'Year', 'sector', 'categorie'])
            .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
            .reset_index()
        )


        self.EPS_BOD = (pd.concat([
                EPS_BOD_proc,
                (EPS_BOD_proc
                    .groupby(['iso3', 'Year', 'sector'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(categorie = 'total')
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_proc
                    .groupby(['iso3', 'Year'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(
                        categorie = 'total',
                        sector = 'total'
                    )
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_proc
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_mbi', categorie='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                (EPS_BOD_proc
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_nmbi', categorie='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                )

            ])
            .sort_values(['iso3', 'Year'])
            .rename(columns = {'EPS_value_weighted' : 'EPS_BOD'})
        )

        self.EPS_BOD.to_csv(f"{Path(__file__).parents[2]}/data/EPS_BOD.csv")
        self.dc_EPS_total['EPS_BOD'] = self.EPS_BOD

  
    def _compute_EPS_BOD_3var(self):
        EPS_BOD_3var = None
        for iso3 in tqdm(list(self.eps_init.iso3.unique())):
            for year in list(self.eps_init.Year.unique()):
                subset = (self.eps_init
                    .query(f"Year == {year} and iso3 == '{iso3}'")
                    .reset_index(drop=True)
                )

                if subset.shape[0] > 0:
                    
                    # Définissez la fonction objective à maximiser
                    def objective_function(w):
                        I = sum(w[i] * subset['EPS_value'].iloc[i] for i in range(len(w)))
                        var = np.var([w[i] * subset['EPS_value'].iloc[i] for i in range(len(w))])
                        return -(I-3*var)

                    # Contrainte : la somme des poids doit être égale à 1
                    def constraint_sum_to_one(w):
                        return sum(w) - 1

                    # Contrainte : les poids doivent être positifs
                    def constraint_positive_weights(w):
                        return w + 1e-20

                    # Contrainte supplémentaire : la somme des poids par secteur divisée par la somme totale des poids
                    # ne doit pas dépasser la part des émissions carbone de ce secteur dans les émissions totales
                    # Dans la fonction constraint_emissions_ratio, ajuster le calcul de la somme
                    def constraint_sector_sum(w):
                        result = []
                        for sector in self.eps_init['sector_category'].unique():
                            subset_sector = subset.query(f"sector_category == '{sector}'")
                            weight_index = list(subset_sector.index)
                            sum_weight_sector = sum(w[i] for i in weight_index)
                            result.append(sum_weight_sector - 1/8)
                        return result
                    
                    constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                                {'type': 'ineq', 'fun': constraint_positive_weights},
                                {'type': 'ineq', 'fun': constraint_sector_sum}]


                    # Définition des valeurs initiales
                    initial_guess = [1.0 / len(subset)] * len(subset)

                    # Résolution du problème d'optimisation
                    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

                    EPS_BOD_3var = pd.concat([EPS_BOD_3var, subset.assign(weight = result.x)])

        EPS_BOD_3var_proc = (EPS_BOD_3var
            .assign(EPS_value_weighted = lambda df: abs(df.EPS_value * df.weight).round(4))
            .groupby(['iso3', 'Year', 'sector', 'categorie'])
            .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
            .reset_index()
        )


        self.EPS_BOD_3var = (pd.concat([
                EPS_BOD_3var_proc,
                (EPS_BOD_3var_proc
                    .groupby(['iso3', 'Year', 'sector'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(categorie = 'total')
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_3var_proc
                    .groupby(['iso3', 'Year'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(
                        categorie = 'total',
                        sector = 'total'
                    )
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_3var_proc
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_mbi', categorie='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                (EPS_BOD_3var_proc
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_nmbi', categorie='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                )
            ])
            .sort_values(['iso3', 'Year'])
            .rename(columns = {'EPS_value_weighted' : 'EPS_BOD_3var'})
        )

        self.EPS_BOD_3var.to_csv(f"{Path(__file__).parents[2]}/data/EPS_BOD_3var.csv")
        self.dc_EPS_total['EPS_BOD_3var'] = self.EPS_BOD_3var   

    def _compute_EPS_BOD_10var(self):
        EPS_BOD_10var = None
        for iso3 in tqdm(list(self.eps_init.iso3.unique())):
            for year in list(self.eps_init.Year.unique()):
                subset = (self.eps_init
                    .query(f"Year == {year} and iso3 == '{iso3}'")
                    .reset_index(drop=True)
                )

                if subset.shape[0] > 0:
                    
                    # Définissez la fonction objective à maximiser
                    def objective_function(w):
                        I = sum(w[i] * subset['EPS_value'].iloc[i] for i in range(len(w)))
                        var = np.var([w[i] * subset['EPS_value'].iloc[i] for i in range(len(w))])
                        return -(I-10*var)

                    # Contrainte : la somme des poids doit être égale à 1
                    def constraint_sum_to_one(w):
                        return sum(w) - 1

                    # Contrainte : les poids doivent être positifs
                    def constraint_positive_weights(w):
                        return w + 1e-20

                    # Contrainte supplémentaire : la somme des poids par secteur divisée par la somme totale des poids
                    # ne doit pas dépasser la part des émissions carbone de ce secteur dans les émissions totales
                    # Dans la fonction constraint_emissions_ratio, ajuster le calcul de la somme
                    def constraint_sector_sum(w):
                        result = []
                        for sector in self.eps_init['sector_category'].unique():
                            subset_sector = subset.query(f"sector_category == '{sector}'")
                            weight_index = list(subset_sector.index)
                            sum_weight_sector = sum(w[i] for i in weight_index)
                            result.append(sum_weight_sector - 1/8)
                        return result
                    
                    constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                                {'type': 'ineq', 'fun': constraint_positive_weights},
                                {'type': 'ineq', 'fun': constraint_sector_sum}]


                    # Définition des valeurs initiales
                    initial_guess = [1.0 / len(subset)] * len(subset)

                    # Résolution du problème d'optimisation
                    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

                    EPS_BOD_10var = pd.concat([EPS_BOD_10var, subset.assign(weight = result.x)])

        EPS_BOD_10var_proc = (EPS_BOD_10var
            .assign(EPS_value_weighted = lambda df: abs(df.EPS_value * df.weight).round(4))
            .groupby(['iso3', 'Year', 'sector', 'categorie'])
            .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
            .reset_index()
        )


        self.EPS_BOD_10var = (pd.concat([
                EPS_BOD_10var_proc,
                (EPS_BOD_10var_proc
                    .groupby(['iso3', 'Year', 'sector'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(categorie = 'total')
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_10var_proc
                    .groupby(['iso3', 'Year'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(
                        categorie = 'total',
                        sector = 'total'
                    )
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_10var_proc
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_mbi', categorie='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                (EPS_BOD_10var_proc
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_nmbi', categorie='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                )
            ])
            .sort_values(['iso3', 'Year'])
            .rename(columns = {'EPS_value_weighted' : 'EPS_BOD_10var'})
        )

        self.EPS_BOD_10var.to_csv(f"{Path(__file__).parents[2]}/data/EPS_BOD_10var.csv")
        self.dc_EPS_total['EPS_BOD_10var'] = self.EPS_BOD_10var   

    def _compute_EPS_BOD_50var(self):
        EPS_BOD_50var = None
        for iso3 in tqdm(list(self.eps_init.iso3.unique())):
            for year in list(self.eps_init.Year.unique()):
                subset = (self.eps_init
                    .query(f"Year == {year} and iso3 == '{iso3}'")
                    .reset_index(drop=True)
                )

                if subset.shape[0] > 0:
                    
                    # Définissez la fonction objective à maximiser
                    def objective_function(w):
                        I = sum(w[i] * subset['EPS_value'].iloc[i] for i in range(len(w)))
                        var = np.var([w[i] * subset['EPS_value'].iloc[i] for i in range(len(w))])
                        return -(I-50*var)

                    # Contrainte : la somme des poids doit être égale à 1
                    def constraint_sum_to_one(w):
                        return sum(w) - 1

                    # Contrainte : les poids doivent être positifs
                    def constraint_positive_weights(w):
                        return w + 1e-20

                    # Contrainte supplémentaire : la somme des poids par secteur divisée par la somme totale des poids
                    # ne doit pas dépasser la part des émissions carbone de ce secteur dans les émissions totales
                    # Dans la fonction constraint_emissions_ratio, ajuster le calcul de la somme
                    def constraint_sector_sum(w):
                        result = []
                        for sector in self.eps_init['sector_category'].unique():
                            subset_sector = subset.query(f"sector_category == '{sector}'")
                            weight_index = list(subset_sector.index)
                            sum_weight_sector = sum(w[i] for i in weight_index)
                            result.append(sum_weight_sector - 1/8)
                        return result
                    
                    constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                                {'type': 'ineq', 'fun': constraint_positive_weights},
                                {'type': 'ineq', 'fun': constraint_sector_sum}]


                    # Définition des valeurs initiales
                    initial_guess = [1.0 / len(subset)] * len(subset)

                    # Résolution du problème d'optimisation
                    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

                    EPS_BOD_50var = pd.concat([EPS_BOD_50var, subset.assign(weight = result.x)])

        EPS_BOD_50var_proc = (EPS_BOD_50var
            .assign(EPS_value_weighted = lambda df: abs(df.EPS_value * df.weight).round(4))
            .groupby(['iso3', 'Year', 'sector', 'categorie'])
            .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
            .reset_index()
        )


        self.EPS_BOD_50var = (pd.concat([
                EPS_BOD_50var_proc,
                (EPS_BOD_50var_proc
                    .groupby(['iso3', 'Year', 'sector'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(categorie = 'total')
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_50var_proc
                    .groupby(['iso3', 'Year'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(
                        categorie = 'total',
                        sector = 'total'
                    )
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_50var_proc
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_mbi', categorie='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                (EPS_BOD_50var_proc
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value=('EPS_value', 'sum'))
                    .assign(sector='total_nmbi', categorie='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                )
            ])
            .sort_values(['iso3', 'Year'])
            .rename(columns = {'EPS_value_weighted' : 'EPS_BOD_50var'})
        )

        self.EPS_BOD_50var.to_csv(f"{Path(__file__).parents[2]}/data/EPS_BOD_50var.csv")
        self.dc_EPS_total['EPS_BOD_50var'] = self.EPS_BOD_50var   

    # def _compute_EPS_BOD_85var(self):
    #     EPS_BOD_85var = None
    #     for iso3 in tqdm(list(self.eps_init.iso3.unique())):
    #         for year in list(self.eps_init.Year.unique()):
    #             subset = (self.eps_init
    #                 .query(f"Year == {year} and iso3 == '{iso3}'")
    #                 .reset_index(drop=True)
    #             )

    #             if subset.shape[0] > 0:
                    
    #                 # Définissez la fonction objective à maximiser
    #                 def objective_function(w):
    #                     I = sum(w[i] * subset['EPS_value'].iloc[i] for i in range(len(w)))
    #                     var = np.var([w[i] * subset['EPS_value'].iloc[i] for i in range(len(w))])
    #                     return -(I-85*var)

    #                 # Contrainte : la somme des poids doit être égale à 1
    #                 def constraint_sum_to_one(w):
    #                     return sum(w) - 1

    #                 # Contrainte : les poids doivent être positifs
    #                 def constraint_positive_weights(w):
    #                     return w + 1e-20

    #                 # Contrainte supplémentaire : la somme des poids par secteur divisée par la somme totale des poids
    #                 # ne doit pas dépasser la part des émissions carbone de ce secteur dans les émissions totales
    #                 # Dans la fonction constraint_emissions_ratio, ajuster le calcul de la somme
    #                 def constraint_sector_sum(w):
    #                     result = []
    #                     for sector in self.eps_init['sector_category'].unique():
    #                         subset_sector = subset.query(f"sector_category == '{sector}'")
    #                         weight_index = list(subset_sector.index)
    #                         sum_weight_sector = sum(w[i] for i in weight_index)
    #                         result.append(sum_weight_sector - 1/8)
    #                     return result
                    
    #                 constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
    #                             {'type': 'ineq', 'fun': constraint_positive_weights},
    #                             {'type': 'ineq', 'fun': constraint_sector_sum}]


    #                 # Définition des valeurs initiales
    #                 initial_guess = [1.0 / len(subset)] * len(subset)

    #                 # Résolution du problème d'optimisation
    #                 result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

    #                 EPS_BOD_85var = pd.concat([EPS_BOD_85var, subset.assign(weight = result.x)])

    #     EPS_BOD_85var_proc = (EPS_BOD_85var
    #         .assign(EPS_value_weighted = lambda df: abs(df.EPS_value * df.weight).round(4))
    #         .groupby(['iso3', 'Year', 'sector', 'categorie'])
    #         .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
    #         .reset_index()
    #     )


    #     self.EPS_BOD_85var = (pd.concat([
    #             EPS_BOD_85var_proc,
    #             (EPS_BOD_85var_proc
    #                 .groupby(['iso3', 'Year', 'sector'])
    #                 .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
    #                 .reset_index()
    #                 .assign(categorie = 'total')
    #                 .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
    #             ),
    #             (EPS_BOD_85var_proc
    #                 .groupby(['iso3', 'Year'])
    #                 .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
    #                 .reset_index()
    #                 .assign(
    #                     categorie = 'total',
    #                     sector = 'total'
    #                 )
    #                 .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
    #             ),
    #             (EPS_BOD_85var_proc
    #                 .query("categorie == 'mbi'")
    #                 .groupby(['Year', 'iso3'], as_index=False)
    #                 .agg(EPS_value_weighted=('EPS_value_weighted', 'sum'))
    #                 .assign(sector='total_mbi', categorie='total_mbi')
    #                 .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value_weighted']]
    #             ),
    #             (EPS_BOD_85var_proc
    #                 .query("categorie == 'nmbi'")
    #                 .groupby(['Year', 'iso3'], as_index=False)
    #                 .agg(EPS_value_weighted=('EPS_value_weighted', 'sum'))
    #                 .assign(sector='total_nmbi', categorie='total_nmbi')
    #                 .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value_weighted']]
    #             )
    #         ])
    #         .sort_values(['iso3', 'Year'])
    #         .rename(columns = {'EPS_value_weighted' : 'EPS_BOD_85var'})
    #     )

    #     self.EPS_BOD_85var.to_csv(r"C:\Users\B25880\OneDrive - EDF\Documents drive\VSCode\eps_package\data\EPS_BOD_85var.csv")
    #     self.dc_EPS_total['EPS_BOD_85var'] = self.EPS_BOD_85var   

    def _compute_EPS_BOD_100var(self):
        EPS_BOD_100var = None
        for iso3 in tqdm(list(self.eps_init.iso3.unique())):
            for year in list(self.eps_init.Year.unique()):
                subset = (self.eps_init
                    .query(f"Year == {year} and iso3 == '{iso3}'")
                    .reset_index(drop=True)
                )

                if subset.shape[0] > 0:
                    
                    # Définissez la fonction objective à maximiser
                    def objective_function(w):
                        I = sum(w[i] * subset['EPS_value'].iloc[i] for i in range(len(w)))
                        var = np.var([w[i] * subset['EPS_value'].iloc[i] for i in range(len(w))])
                        return -(I-100*var)

                    # Contrainte : la somme des poids doit être égale à 1
                    def constraint_sum_to_one(w):
                        return sum(w) - 1

                    # Contrainte : les poids doivent être positifs
                    def constraint_positive_weights(w):
                        return w + 1e-20

                    # Contrainte supplémentaire : la somme des poids par secteur divisée par la somme totale des poids
                    # ne doit pas dépasser la part des émissions carbone de ce secteur dans les émissions totales
                    # Dans la fonction constraint_emissions_ratio, ajuster le calcul de la somme
                    def constraint_sector_sum(w):
                        result = []
                        for sector in self.eps_init['sector_category'].unique():
                            subset_sector = subset.query(f"sector_category == '{sector}'")
                            weight_index = list(subset_sector.index)
                            sum_weight_sector = sum(w[i] for i in weight_index)
                            result.append(sum_weight_sector - 1/8)
                        return result
                    
                    constraints = [{'type': 'eq', 'fun': constraint_sum_to_one},
                                {'type': 'ineq', 'fun': constraint_positive_weights},
                                {'type': 'ineq', 'fun': constraint_sector_sum}]


                    # Définition des valeurs initiales
                    initial_guess = [1.0 / len(subset)] * len(subset)

                    # Résolution du problème d'optimisation
                    result = minimize(objective_function, initial_guess, method='SLSQP', constraints=constraints)

                    EPS_BOD_100var = pd.concat([EPS_BOD_100var, subset.assign(weight = result.x)])

        EPS_BOD_100var_proc = (EPS_BOD_100var
            .assign(EPS_value_weighted = lambda df: abs(df.EPS_value * df.weight).round(4))
            .groupby(['iso3', 'Year', 'sector', 'categorie'])
            .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
            .reset_index()
        )


        self.EPS_BOD_100var = (pd.concat([
                EPS_BOD_100var_proc,
                (EPS_BOD_100var_proc
                    .groupby(['iso3', 'Year', 'sector'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(categorie = 'total')
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_100var_proc
                    .groupby(['iso3', 'Year'])
                    .agg(EPS_value_weighted = ('EPS_value_weighted', 'sum'))
                    .reset_index()
                    .assign(
                        categorie = 'total',
                        sector = 'total'
                    )
                    .loc[:, ['iso3', 'Year', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_100var_proc
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value_weighted=('EPS_value_weighted', 'sum'))
                    .assign(sector='total_mbi', categorie='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value_weighted']]
                ),
                (EPS_BOD_100var_proc
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value_weighted=('EPS_value_weighted', 'sum'))
                    .assign(sector='total_nmbi', categorie='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value_weighted']]
                )
            ])
            .sort_values(['iso3', 'Year'])
            .rename(columns = {'EPS_value_weighted' : 'EPS_BOD_100var'})
        )

        self.EPS_BOD_100var.to_csv(f"{Path(__file__).parents[2]}/data/EPS_BOD_100var.csv")
        self.dc_EPS_total['EPS_BOD_100var'] = self.EPS_BOD_100var   

    def _compute_EPS_GHG(self):
        self.EPS_GHG = (
            pd.concat([
                # EPS GHG categorie level
                (self.EPS_GHG_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GHG_ratio = ('GHG_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GHG_ratio / 2)
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GHG sector level
                (self.EPS_GHG_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GHG_ratio = ('GHG_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GHG_ratio / 2)
                    .groupby(['Year', 'iso3', 'sector'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GHG total
                (self.EPS_GHG_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GHG_ratio = ('GHG_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GHG_ratio / 2)
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total', sector='total')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GHG mbi
                (self.EPS_GHG_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GHG_ratio = ('GHG_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GHG_ratio / 2)
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total_mbi', sector='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GHG nmbi
                (self.EPS_GHG_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GHG_ratio = ('GHG_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GHG_ratio / 2)
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total_nmbi', sector='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                )
            ])
            .rename(columns={'EPS_value': 'EPS_GHG'})
        )

        self.EPS_GHG.to_csv(f"{Path(__file__).parents[2]}/data/EPS_GHG.csv")
        self.dc_EPS_total['EPS_GHG'] = self.EPS_GHG

    def _compute_EPS_GDP(self):
        self.EPS_GDP = (
            pd.concat([
                # EPS GDP categorie level
                (self.EPS_GDP_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GDP_ratio = ('GDP_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GDP_ratio / 2)
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GDP sector level
                (self.EPS_GDP_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GDP_ratio = ('GDP_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GDP_ratio / 2)
                    .groupby(['Year', 'iso3', 'sector'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GDP total
                (self.EPS_GDP_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GDP_ratio = ('GDP_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GDP_ratio / 2)
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total', sector='total')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GDP mbi
                (self.EPS_GDP_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GDP_ratio = ('GDP_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GDP_ratio / 2)
                    .query("categorie == 'mbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total_mbi', sector='total_mbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                ),
                # EPS GDP nmbi
                (self.EPS_GDP_merged
                    .groupby(['Year', 'iso3', 'sector', 'categorie'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'mean'), GDP_ratio = ('GDP_ratio', 'first'))
                    .assign(EPS_value = lambda df: df.EPS_value * df.GDP_ratio / 2)
                    .query("categorie == 'nmbi'")
                    .groupby(['Year', 'iso3'], as_index=False)
                    .agg(EPS_value = ('EPS_value', 'sum'))
                    .assign(categorie = 'total_nmbi', sector='total_nmbi')
                    .loc[:, ['Year', 'iso3', 'sector', 'categorie', 'EPS_value']]
                )
            ])
            .rename(columns={'EPS_value': 'EPS_GDP'})
        )
        self.EPS_GDP.to_csv(f"{Path(__file__).parents[2]}/data/EPS_GDP.csv")
        self.dc_EPS_total['EPS_GDP'] = self.EPS_GDP



    def load_EPS_csv(self):
        
        for eps in self.eps_computed:
            self.dc_EPS_total[f"EPS_{eps}"] = pd.read_csv(f"{Path(__file__).parents[2]}/data/EPS_{eps}.csv", index_col=0)

    def create_EPS_total(self):
        self.EPS_total = (self.dc_EPS_total['EPS_OCDE']
            .merge(self.dc_EPS_total['EPS_BOD'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer')                 
            .merge(self.dc_EPS_total['EPS_GHG'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer') 
            .merge(self.dc_EPS_total['EPS_GDP'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer')
            .merge(self.dc_EPS_total['EPS_BOD_3var'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer') 
            .merge(self.dc_EPS_total['EPS_BOD_10var'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer')
            .merge(self.dc_EPS_total['EPS_BOD_50var'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer')
            .merge(self.dc_EPS_total['EPS_BOD_100var'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer')
            #.merge(self.dc_EPS_total['EPS_BOD_85var'], on=['iso3', 'Year', 'sector', 'categorie'], how='outer') 
            .sort_values(['iso3', 'Year', 'sector', 'categorie'])               
        )

        self.EPS_total.to_csv(f"{Path(__file__).parents[2]}/data/EPS_total.csv")

    def create_econo_database(self):
        ETS_info = (self.eps_init[['iso3', 'Year', 'ETS_E', 'ETS_I', 'ETS_B', 'ETS_T', 'ETS']]
            .groupby(['iso3', 'Year'])
            .agg(
                ETS_E=('ETS_E', 'max'),  # 1 si ETS_E est présent, sinon 0
                ETS_I=('ETS_I', 'max'),  # 1 si ETS_I est présent, sinon 0
                ETS_B=('ETS_B', 'max'),  # 1 si ETS_B est présent, sinon 0
                ETS_T=('ETS_T', 'max'),
                ETS= ('ETS', 'max'), # 1 si ETS_T est présent, sinon 0
            )
            .reset_index()
        )

        GDP_econo = (self.GDP
            .query("GDP_type == 'Gross domestic product (output approach)' | sector.notna()")
            .assign(sector = lambda df: df.sector.fillna('total'))
            .pivot(index=['iso3', 'Year'], columns='sector', values='GDP_value')
            .pipe(lambda df: df.rename(columns={col: f"{col}_GDP" for col in df.columns}))
            .reset_index()
            .assign(Year = lambda df: df.Year.astype(int))
        )

        GHG_econo = (self.GHG
            .query("GHG_source == 'Total emissions excluding LULUCF' | sector.notna()")
            .assign(sector = lambda df: df.sector.fillna('total'))
            .groupby(['iso3', 'Year', 'sector'], as_index = False)
            .agg(GHG_value=("GHG_value", 'sum'))
            .pivot(index=['iso3', 'Year'], columns='sector', values='GHG_value')
            .pipe(lambda df: df.rename(columns={col: f"{col}_GHG" for col in df.columns}))
            .reset_index()
            .assign(Year = lambda df: df.Year.astype(int))
        )

        HDI_econo = (self.HDI
            .query(f"HDI_type.isin({list(HDI_var_to_keep.keys())})")
            .pivot(columns='HDI_type', values='HDI_value', index=['iso3', 'Year'])
            .rename(columns=HDI_var_to_keep)
            .reset_index()
            .assign(Year = lambda df: df.Year.astype(int))
        )

        def change_columns(df: pd.DataFrame) -> pd.DataFrame:
            df.columns = df.columns.to_flat_index()
            new_columns = ['_'.join([x for x in col if x != '']) for col in df.columns]
            df.columns = new_columns
            return df

        EPS_econo = (self.EPS_total
            .pivot(index=['Year', 'iso3'], values=['EPS_OCDE', 'EPS_BOD', 'EPS_GHG', 'EPS_GDP',  'EPS_BOD_3var', 'EPS_BOD_10var', 'EPS_BOD_50var',  'EPS_BOD_100var' ], columns=['sector', 'categorie'])
            .reset_index()
            .pipe(change_columns)
            .rename(columns=lambda x: x.replace('_total', ''))
            .assign(Year = lambda df: df.Year.astype(int))
        )

        self.econo = (EPS_econo
            .merge(GHG_econo, on=['iso3', 'Year'], how='left')
            .merge(GDP_econo, on=['iso3', 'Year'], how='left')
            .merge(self.POP, on=['iso3', 'Year'], how='left')
            .merge(HDI_econo, on=['iso3', 'Year'], how='left')
            # .merge(self.rnd, on=['iso3', 'Year'], how='left')
            .merge(ETS_info, on=['iso3', 'Year'], how='left')
            .reset_index(drop=True)
            .rename(columns=lambda x: x.replace('total_', '').replace('_value', ''))
            .query(f"~iso3.isin({country_to_drop_econo})" )
            # .query(f"~Year.isin({year_to_drop_econo})" )
            #.assign(GDP_WB = lambda df: df.GDP_WB * 10**-6)
            .pipe(lambda df:
                df.assign(**{
                    f'{column}_per_cap': np.where(df['POP'].isna(), 'nan', df[column].astype(float) / df['POP'].astype(float))
                    for column in df.columns if not(('EPS' in column) or ('ETS' in column) or (column in ['iso3', 'Year', 'urban_growth', 'POP', 'POP_growth', 'indus_0','agri_0']))
                })
            )
            # .pipe(lambda df:
            #     df.assign(**{
            #         column : df[column].astype(float).apply(lambda x: x * 10**3 if not(pd.isna(x)) else x)
            #         for column in df.columns if '_per_cap' in column and ('GHG' in column or 'GDP' in column)
            #     })
            # )
            .pipe(lambda df :
                df.assign(**{
                    'ln_' + column: df[column].astype(float).apply(lambda x: np.log(x) if not(pd.isna(x)) else x)
                    for column in list(df.columns) if not(('EPS' in column)  or ('ETS' in column) or (column in ['iso3', 'Year', 'urban_growth', 'indus_0','agri_0']))
                })
            )
            .pipe(lambda df:
                df.assign(**{
                    f'{column}_lag': df.groupby('iso3')[column].shift(3)
                    for column in df.columns if 'EPS' in column
                })
            )
            .assign(ln_GDP_2 = lambda df: df['ln_GDP'] ** 2)
            .assign(ln_GDP_per_cap_2 = lambda df: df['ln_GDP_per_cap'] ** 2)
            .assign(groupe1 = lambda df: np.where(df['iso3'].isin(country_eu), 'EU', 'NEU'))
            .assign(groupe2 = lambda df: np.where(df['iso3'].isin(developed_countries_iso3), 'Developed', 'Developing'))
            .assign(GDP_growth = lambda df: df.groupby('iso3').GDP.pct_change())
            .assign(GDP_growth_2 = lambda df: df.GDP_growth ** 2)
        )

        self.econo.to_csv(f"{Path(__file__).parents[2]}/data/Econo.csv")

    def clean_econo_database(
        self,
        model_name: str,
        selection_type: str,
        econo_var: list = None,
        save_dataframe: bool = True,
        already_computed: bool = False
    ) -> pd.DataFrame:

        if not already_computed:
            
            match selection_type:
                case 'max_year':
                    self.preprocessed_econo = (self.econo
                        .loc[:, ['iso3', 'Year'] + econo_var]
                        .query(f"~iso3.isin({countries_to_drop+country_to_drop_gdp}) and Year != 2021")
                        .assign(nan_value = lambda df: df.apply(lambda row: row.isna().sum() > 0, axis=1))
                        .pipe(lambda df: (df
                                .query(f"""iso3.isin({list(
                                    df.groupby('iso3', as_index=False)
                                    .agg(nan_value=('nan_value', 'sum'))
                                    .query("nan_value == 0")
                                    .iso3
                                )})""")
                            )
                        )
                    )

                case 'max_iso3':
                    self.preprocessed_econo = (self.econo
                        .loc[:, ['iso3', 'Year'] + econo_var]
                        .query(f"~iso3.isin({country_to_drop_max_iso3}) and Year != 2021")
                        .assign(nan_value = lambda df: df.apply(lambda row: row.isna().sum() > 0, axis=1))
                        .pipe(lambda df: (df
                                .query(f"""iso3.isin({list(
                                    df.groupby('iso3', as_index=False)
                                    .agg(nan_value=('nan_value', 'sum'))
                                    .query("nan_value <= 10")
                                    .iso3
                                )})""")
                            )
                        )
                        .pipe(lambda df: (df
                                .query(f"""Year.isin({list(
                                    df.groupby('Year', as_index=False)
                                    .agg(nan_value=('nan_value', 'sum'))
                                    .query("nan_value == 0")
                                    .Year
                                )})""")
                            )
                        )
                    )

                case 'max_obs':
                    matrix = (self.econo
                        .loc[:, ['iso3', 'Year'] + econo_var]
                        .assign(nan_value = lambda df: df.apply(lambda row: row.isna().sum() > 0, axis=1).astype(int))

                        .pivot(index='Year', columns='iso3', values='nan_value')
                        .applymap(lambda x: np.nan if x == 1.0 else x)
                    )
                    index, columns = OptiMask(n_tries=35).solve(matrix)
                    self.preprocessed_econo = (self.econo
                        .loc[:, ['iso3', 'Year'] + econo_var]
                        .query(f"Year.isin({list(index)}) & iso3.isin({list(columns)})")
                    )

                case 'reference':
                    self.preprocessed_econo = self.econo.copy()

            self._create_multi_index()

            if save_dataframe:
                self.preprocessed_econo.to_csv(f"{Path(__file__).parents[2]}/data/preprocessed_econo/Econo_{model_name}_{selection_type}.csv")

        else:
            self.preprocessed_econo = pd.read_csv(f"{Path(__file__).parents[2]}/data/preprocessed_econo/Econo_{model_name}_{selection_type}.csv", index_col=0)

        self.dc_results[f"{model_name}_{selection_type}"] = self.preprocessed_econo.copy()

    def _create_multi_index(self):
        Year = pd.Categorical(self.preprocessed_econo.Year)
        iso3 = pd.Categorical(self.preprocessed_econo.iso3)
        index_levels = ['iso3', 'Year']
        self.preprocessed_econo.set_index(index_levels, inplace=True)
        self.preprocessed_econo["Year"] = Year
        self.preprocessed_econo['iso3'] = iso3 