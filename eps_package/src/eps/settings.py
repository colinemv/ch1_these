
### LOAD AND PROCESS DATABASRES

sectors = [
    'building',
    'elec',
    'transport',
    'indus'
]

col_to_drop = [
    'STRUCTURE',
    'STRUCTURE_ID',
    'STRUCTURE_NAME',
    'ACTION',
    'Zone de référence',
    'FREQ',
    "Fréquence d'observation",
    'MEASURE',
    'Mesure',
    'UNIT_MEASURE',
    'Unité de mesure',
    'Période temporelle',
    "Valeur de l'observation",
    'OBS_STATUS',
    'UNIT_MULT',
    "Multiplicateur d'unités",
    'DECIMALS',
    'Décimales'   
]

col_to_rename = {
    'REF_AREA': 'iso3',
    'CLIM_ACT_POL': 'pol_code',
    'Actions et politiques en matière de climat': 'pol_nom',
    'TIME_PERIOD' : 'Year',
    'OBS_VALUE' : 'EPS_value',
    "État de l'observation" : 'obs_status'
}

# Liste des pol_codes à classer comme 'mbi'
mbi_values = ['EXCISETAX_I', 'FFS_I', 'FIN_MECH_I','ETS_I', 'CARBONTAX_I', 
              'AUCTION', 'ETS_E', 'CARBONTAX_E', 'FIT','EXCISETAX_E','FFS_E', 'RECS',
               'FFS_B' ,'EXCISETAX_B', 'ETS_B' ,'CARBONTAX_B', 'FIN_MECH_B',
                'EXCISETAX_T', 'ETS_T', 'CARBONTAX_T', 'FFS_T', 'CONG_CHARG']  

# Liste des pol_codes à classer comme 'nmbi'
nmbi_values = ['MEPS_MOTOR', 'EE_MANDATE',
               'EMIS_STD', 'BAN_PHOUT_COAL', 'RENEWABLE_EXP',
               'MEPS_APPL', 'BC', 'LABEL_APPL' ,'BAN_PHOUT_HEAT',
               'MEPS_T', 'EXP_RAIL', 'SPEED', 'LABEL_CAR', 'BAN_PHOUT_ICE']

col_to_drop_GDP= [
    'Country',
    'TRANSACT',
    'TIME',
    'Unit Code',
    'PowerCode Code',
    'MEASURE',
    'Measure',
    "Reference Period Code",
    'Flag Codes',
    'Flags',
    'Unit',
    'PowerCode',
    'Reference Period'
]

col_to_rename_GDP = {
    'LOCATION': 'iso3',
    'Transaction' : 'GDP_type',
    'Value' : 'GDP_value'
}

col_to_drop_GDP_manufacture=[
    'STRUCTURE',
    'STRUCTURE_ID',
    'STRUCTURE_NAME',
    'ACTION',
    'FREQ',
    "Fréquence d'observation",
    "Zone de référence",
    'SECTOR',
    "Secteur institutionnel",
    'COUNTERPART_SECTOR',
    'Secteur institutionnel de contrepartie',
    'TRANSACTION',
    'Transaction',
    'INSTR_ASSET',
    'Instruments financiers et actifs non financiers',
    'ACTIVITY',
    'EXPENDITURE',
    'Dépense',
    'UNIT_MEASURE',
    'Unité de mesure',
    'PRICE_BASE',
    'Type de prix',
    'TRANSFORMATION',
    'Transformation',
    'TABLE_IDENTIFIER',
    'Identifiant de tableau',
    'Période temporelle',
    "Valeur d'observation",
    'REF_YEAR_PRICE',
    "Prix ​​année de référence",
    'CONF_STATUS',
    'Statut de confidentialité',
    'DECIMALS',
    'Décimales',
    'OBS_STATUS',
    "Statut d'observation",
    'UNIT_MULT',
    "Multiplicateur d'unité",
    'CURRENCY',
    'Monnaie'
]

col_to_rename_GDP_manufacture = {
    'REF_AREA': 'iso3',
    'TIME_PERIOD' : 'Year',
    'Activité économique' : 'GDP_type',
    'OBS_VALUE' : 'GDP_value'
}

col_to_drop_POP= [
    'Pays',
    'TIME',
    'Sexe',
    'SEX',
    'AGE',
    'Âge',
    'Flag Codes',
    'Flags'  
]

col_to_rename_POP = {
    'LOCATION': 'iso3',
    'Temps': 'Year',
    'Value': 'POP_value'
}

col_to_drop_GHG= [
    'Country',
    'YEA',
    'POL',
    'Pollutant',
    'VAR',
    'Unit Code',
    'PowerCode Code',
    "Reference Period Code",
    'Flag Codes',
    'Flags'
]

col_to_rename_GHG = {
    'COU': 'iso3',
    'Value' : 'GHG_value',
    'Variable' : 'GHG_source'
}

col_to_drop_HDI= [
    'Country_Name',
    'Series_Code'
]

col_to_rename_HDI = {
    'Country_Code': 'iso3',
    'Series_Name' : 'HDI_type'
}

sector_mapping_GHG = {
    '1A4 - Residential and other sectors': 'building',
    '1A5 - Energy - Other': 'elec',
    '1B - Fugitive Emissions from Fuels': 'elec',
    '1A3 - Transport': 'transport',
    '1A1 - Energy Industries': 'elec',
    '1A2 - Manufacturing industries and construction': 'indus',
    '2- Industrial processes and product use' : 'indus'
}

sector_mapping_GDP = {
    "Real estate activities (ISIC rev4)": 'building',
    "Distributive trade, repairs; transport; accommod., food serv. (ISIC rev4)": 'transport',
    'Activités de fabrication': 'indus'
}

HDI_var_to_keep = {
    #'Foreign direct investment, net inflows (BoP, current US$)': 'FDI',
    'Population growth (annual %)' : 'POP_growth',
    'Urban population growth (annual %)': 'urban_growth',
    "Industry (including construction), value added (% of GDP)": 'indus_0',
    "Agriculture, forestry, and fishing, value added (% of GDP)": 'agri_0'
}

# col_to_drop_rnd = {
#     'STRUCTURE', 'STRUCTURE_ID', 'STRUCTURE_NAME', 'ACTION',
#        'Reference area', 'FREQ', 'Frequency of observation', 'MEASURE',
#        'Measure', 'UNIT_MEASURE', 'Unit of measure', 'PRICE_BASE',
#        'Price base', 'TRANSFORMATION', 'Transformation',
#        'Time period', 'Observation value', 'OBS_STATUS',
#        'Observation status', 'OBS_STATUS_2', 'Observation status 2',
#        'OBS_STATUS_3', 'Observation status 3', 'AUX_OBS_STATUS',
#        'Aux observation status', 'CONF_STATUS', 'Confidentiality status',
#        'UNIT_MULT', 'Unit multiplier', 'BASE_PER', 'Base year', 'DECIMALS',
#        'Decimals'
# }

# col_to_rename_rnd = {
#     'REF_AREA' : 'iso3',
#     'TIME_PERIOD' : 'Year',
#     'OBS_VALUE' : 'rnd'
# }

countries_to_drop = ['ZAF', 'IDN', 'MEX', 'COL', 'ISR', 'ARG', 'CHN', 'SAU', 'BRA', 'IND', 'USA']
country_to_drop_gdp = ['BGR', 'EST', 'HRV', 'HUN', 'ISL', 'LTU', 'LVA', 'MLT', 'ROU', 'RUS', 'SVK', 'SVN']
country_to_drop_max_iso3 = ['ZAF', 'IDN', 'IND', 'CHN', 'SAU', 'MDG']
country_to_drop_econo = ['USA']
year_to_drop_econo = [1990, 1991, 1992, 1993, 1994]
country_eu = ['AUT', 'BEL', 'BGR', 'HRV', 'GBR', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'GRC', 'FRA', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'SWE']
# Developed countries
developed_countries_iso3 = [
    'AUS',  # Australia
    'AUT',  # Austria
    'BEL',  # Belgium
    'CAN',  # Canada
    'CZE',  # Czech Republic
    'DNK',  # Denmark
    'FIN',  # Finland
    'FRA',  # France
    'DEU',  # Germany
    'ISL',  # Iceland
    'IRL',  # Ireland
    'ISR',  # Israel
    'ITA',  # Italy
    'JPN',  # Japan
    'KOR',  # Korea
    'LUX',  # Luxembourg
    'NLD',  # Netherlands
    'NZL',  # New Zealand
    'NOR',  # Norway
    'PRT',  # Portugal
    'SVK',  # Slovak Republic
    'SVN',  # Slovenia
    'ESP',  # Spain
    'SWE',  # Sweden
    'CHE',  # Switzerland
    'GBR',
    'USA',
    'EU'  # United Kingdom
]

# Developing countries
developing_countries_iso3 = [
    'ARG',  # Argentina
    'BRA',  # Brazil
    'BGR',  # Bulgaria
    'CHL',  # Chile
    'CHN',  # China
    'COL',  # Colombia
    'CRI',  # Costa Rica
    'HRV',  # Croatia
    'EST',  # Estonia
    'GRC',  # Greece
    'HUN',  # Hungary
    'IND',  # India
    'IDN',  # Indonesia
    'LVA',  # Latvia
    'LTU',  # Lithuania
    'MLT',  # Malta
    'MEX',  # Mexico
    'PER',  # Peru
    'POL',  # Poland
    'ROU',  # Romania
    'RUS',  # Russian Federation
    'SAU',  # Saudi Arabia
    'ZAF',  # South Africa
    'TUR'   # Türkiye
]

