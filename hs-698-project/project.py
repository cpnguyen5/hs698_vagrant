from api import db, app
from api.models import Report, Puf, Cancer
import os
import pandas as pd
import numpy as np
from pyzipcode import Pyzipcode as pz
import urllib3


def get_path():
    f_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'api', 'dataset')
    return f_name


def download(url, path):
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=False)
    with open(path, 'wb') as output:
        while True:
            data = r.read(1024)
            if not data:
                break
            output.write(data)

    r.release_conn()
    return


def readCSV():
    f='Medicare_Physician_and_Other_Supplier_National_Provider_Identifier__NPI__Aggregate_Report__Calendar_Year_2014.csv'
    f_path = os.path.join(get_path(), f)
    if not os.path.isfile(f_path):
        print "Downloading Report CSV file -- download may take awhile..."
        download('https://data.cms.gov/api/views/4a3h-46r6/rows.csv?accessType=DOWNLOAD', f_path)
        print "Report CSV download complete"
    columns = ["npi", "provider_last_name", "provider_first_name", "provider_middle_initial", "provider_credentials",
               "provider_gender", "provider_entity_type", "provider_street_address_1", "provider_street_address_2",
               "provider_city", "provider_zip_code", "provider_state_code", "provider_country_code", "provider_type",
               "medicare_participation_indicator", "number_of_HCPCS", "number_of_services", "number_of_medicare_beneficiaries",
               "total_submitted_charge_amount", "total_medicare_allowed_amount", "total_medicare_payment_amount",
               "total_medicare_standardized_payment_amount", "drug_suppress_indicator",
               "number_of_HCPCS_associated_with_drug_services", "number_of_drug_services",
               "number_of_medicare_beneficiaries_with_drug_services", "total_drug_submitted_charge_amount",
               "total_drug_medicare_allowed_amount", "total_drug_medicare_payment_amount", "total_drug_medicare_standardized_payment_amount",
               "medical_suppress_indicator", "number_of_HCPCS_associated_medical_services", "number_of_medical_services",
               "number_of_medicare_beneficiaries_with_medical_services", "total_medical_submitted_charge_amount",
               "total_medical_medicare_allowed_amount", "total_medical_medicare_payment_amount",
               "total_medical_medicare_standardized_payment_amount", "average_age_of_beneficiaries", "number_of_beneficiaries_age_less_65",
               "number_of_beneficiaries_age_65_to_74", "number_of_beneficiaries_age_75_to_84", "number_of_beneficiaries_age_greater_84",
               "number_of_female_beneficiaries", "number_of_male_beneficiaries", "number_of_non_hispanic_white_beneficiaries",
               "number_of_african_american_beneficiaries", "number_of_asian_pacific_islander_beneficiaries",
               "number_of_hispanic_beneficiaries", "number_of_american_indian_alaskan_native_beneficiaries",
               "number_of_beneficiaries_with_race_not_elsewhere_classified", "number_of_beneficiaries_with_medicare_only_entitlement",
               "number_of_beneficiaries_with_medicare_and_medicaid_entitlement", "percent_of_beneficiaries_identified_with_atrial_fibrillation",
               "percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia",
               "percent_of_beneficiaries_identified_with_asthma", "percent_of_beneficiaries_identified_with_cancer",
               "percent_of_beneficiaries_identified_with_heart_failure", "percent_of_beneficiaries_identified_with_chronic_kidney_disease",
               "percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease", "percent_of_beneficiaries_identified_with_depression",
               "percent_of_beneficiaries_identified_with_diabetes", "percent_of_beneficiaries_identified_with_hyperlipidemia",
               "percent_of_beneficiaries_identified_with_hypertension", "percent_of_beneficiaries_identified_with_ischemic_heart_disease",
               "percent_of_beneficiaries_identified_with_osteoporosis", "percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis",
               "percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders",
               "percent_of_beneficiaries_identified_with_stroke", "average_HCC_risk_score_of_beneficiaries"]
    types = [('npi', np.int64), ('provider_last_name', 'S20'), ('provider_first_name', 'S20'),
             ('provider_middle_initial', 'S20'), ('provider_credentials', 'S20'),
             ('provider_gender', 'S20'), ('provider_entity_type', 'S20'), ('provider_street_address_1', 'S20'),
             ('provider_street_address_2', 'S20'),
             ('provider_city', 'S20'), ('provider_zip_code', 'S20'), ('provider_state_code', 'S20'),
             ('provider_country_code', 'S20'), ('provider_type', 'S20'),
             ('medicare_participation_indicator', 'S20'), ('number_of_HCPCS', np.int64), ('number_of_services', np.float64),
             ('number_of_medicare_beneficiaries', np.int64),
             ('total_submitted_charge_amount', np.float64), ('total_medicare_allowed_amount', np.float64),
             ('total_medicare_payment_amount', np.float64),
             ('total_medicare_standardized_payment_amount', np.float64), ('drug_suppress_indicator', 'S20'),
             ('number_of_HCPCS_associated_with_drug_services', np.float64),
             ('number_of_drug_services', np.float64), ('number_of_medicare_beneficiaries_with_drug_services', np.float64),
             ('total_drug_submitted_charge_amount', np.float64),
             ('total_drug_medicare_allowed_amount', np.float64), ('total_drug_medicare_payment_amount', np.float64),
             ('total_drug_medicare_standardized_payment_amount', np.float64),
             ('medical_suppress_indicator', 'S20'), ('number_of_HCPCS_associated_medical_services', np.float64),
             ('number_of_medical_services', np.float64),
             ('number_of_medicare_beneficiaries_with_medical_services', np.float64),
             ('total_medical_submitted_charge_amount', np.float64),
             ('total_medical_medicare_allowed_amount', np.float64), ('total_medical_medicare_payment_amount', np.float64),
             ('total_medical_medicare_standardized_payment_amount', np.float64),
             ('average_age_of_beneficiaries', np.int64), ('number_of_beneficiaries_age_less_65', np.float64),
             ('number_of_beneficiaries_age_65_to_74', np.float64),
             ('number_of_beneficiaries_age_75_to_84', np.float64), ('number_of_beneficiaries_age_greater_84', np.float64),
             ('number_of_female_beneficiaries', np.float64),
             ('number_of_male_beneficiaries', np.float64), ('number_of_non_hispanic_white_beneficiaries', np.float64),
             ('number_of_african_american_beneficiaries', np.float64),
             ('number_of_asian_pacific_islander_beneficiaries', np.float64),
             ('number_of_hispanic_beneficiaries', np.float64),
             ('number_of_american_indian_alaskan_native_beneficiaries', np.float64),
             ('number_of_beneficiaries_with_race_not_elsewhere_classified', np.float64),
             ('number_of_beneficiaries_with_medicare_only_entitlement', np.float64),
             ('number_of_beneficiaries_with_medicare_and_medicaid_entitlement', np.float64),
             ('percent_of_beneficiaries_identified_with_atrial_fibrillation', np.float64),
             ('percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia', np.float64),
             ('percent_of_beneficiaries_identified_with_asthma', np.float64),
             ('percent_of_beneficiaries_identified_with_cancer', np.float64),
             ('percent_of_beneficiaries_identified_with_heart_failure', np.float64),
             ('percent_of_beneficiaries_identified_with_chronic_kidney_disease', np.float64),
             ('percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease', np.float64),
             ('percent_of_beneficiaries_identified_with_depression', np.float64),
             ('percent_of_beneficiaries_identified_with_diabetes', np.float64),
             ('percent_of_beneficiaries_identified_with_hyperlipidemia', np.float64),
             ('percent_of_beneficiaries_identified_with_hypertension', np.float64),
             ('percent_of_beneficiaries_identified_with_ischemic_heart_disease', np.float64),
             ('percent_of_beneficiaries_identified_with_osteoporosis', np.float64),
             ('percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis', np.float64),
             ('percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders', np.float64),
             ('percent_of_beneficiaries_identified_with_stroke', np.float64),
             ('average_HCC_risk_score_of_beneficiaries', np.float64)]
    rep_reader = pd.read_csv(f_path, sep=',', iterator=True, chunksize=200000, names=columns, header=0, dtype=types,
                     na_values='')

    #filter for only US states -- Convert to Numpy array
    state = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA',
             'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
             'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']

    terr = ['PR', 'GU', 'VI', 'AS', 'District of Columbia', 'MP', 'AA', 'AE', 'AP'] #USA territories
    usa = state+terr

    report_lst = []
    for chunk in rep_reader:
        data = chunk.as_matrix()
        US_data = np.array([row for row in data if row[12]=='US'])
        for row in US_data:
            if row[11] not in usa and len(str(row[10])) >=5:
                location=pz.get(int(str(row[10])[:5]),'US')
                if location != False:
                    row[10]=location['postal_code'] #correct ZIP code
                    row[11]=location['state_short'] #correct state code
        state_data= np.array([row for row in US_data if row[11] in state])

        #Convert to recarray -- transfer hetergeneous column dtypes to DataFrame
        state_recarray = np.core.records.fromarrays(np.transpose(state_data), dtype=types, names=columns)
        #Convert to Pandas DataFrame
        state_df = pd.DataFrame.from_records(state_recarray, columns=columns)
        state_df = state_df.replace(to_replace='', value=np.nan)
        report_lst += [state_df]
    return report_lst


def readPUF():
    puf_f ='Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2014.csv'
    puf_path = os.path.join(get_path(), puf_f)
    if not os.path.isfile(puf_path):
        print "Downloading PUF CSV file -- download may take awhile..."
        download('https://data.cms.gov/api/views/ee7f-sh97/rows.csv?accessType=DOWNLOAD', puf_path)
        print "PUF CSV download complete"
    puf_columns = ['npi', 'provider_last_name', 'provider_first_name', 'provider_middle_initial','provider_credentials',
               'provider_gender', 'provider_entity_type', 'provider_street_address_1', 'provider_street_address_2',
               'provider_city', 'provider_zip_code', 'provider_state_code', 'provider_country_code', 'provider_type',
               'medicare_participation_indicator', 'place_of_service', 'HCPCS_code', 'HCPCS_description',
               'identifies_HCPCS_as_drug_included_in_the_ASP_drug_list', 'number_of_services',
               'number_of_medicare_beneficiaries', 'number_of_distinct_medicare_beneficiary_per_day_services',
               'average_medicare_allowed_amount', 'average_submitted_charge_amount', 'average_medicare_payment_amount',
               'average_medicare_standardized_amount']

    puf_types = [('npi', np.uint64), ('provider_last_name', 'S20'), ('provider_first_name', 'S20'), ('provider_middle_initial', 'S20'),
             ('provider_credentials', 'S20'),('provider_gender', 'S20'),('provider_entity_type', 'S20'),
             ('provider_street_address_1', 'S20'),('provider_street_address_2', 'S20'),('provider_city', 'S20'),
             ('provider_zip_code', 'S20'),('provider_state_code', 'S20'),('provider_country_code', 'S20'),
             ('provider_type', 'S20'),('medicare_participation_indicator', 'S20'),('place_of_service', 'S20'),
             ('HCPCS_code', 'S20'),('HCPCS_description', 'S20'),('identifies_HCPCS_as_drug_included_in_the_ASP_drug_list', 'S20'),
             ('number_of_services', np.float64),('number_of_medicare_beneficiaries', np.float64),
             ('number_of_distinct_medicare_beneficiary_per_day_services', np.float64),
             ('average_medicare_allowed_amount', np.float64),('average_submitted_charge_amount', np.float64),
             ('average_medicare_payment_amount', np.float64),('average_medicare_standardized_amount', np.float64)]

    sel = ['npi', 'place_of_service', 'HCPCS_code', 'HCPCS_description',
           'identifies_HCPCS_as_drug_included_in_the_ASP_drug_list', 'number_of_services',
           'number_of_medicare_beneficiaries', 'number_of_distinct_medicare_beneficiary_per_day_services',
           'average_medicare_allowed_amount', 'average_submitted_charge_amount', 'average_medicare_payment_amount',
           'average_medicare_standardized_amount']

    #read in CSV in chunks -- chunks of rows
    csv_path= os.path.join(get_path(),
                           'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2014.csv')
    reader = pd.read_csv(csv_path, iterator=True, chunksize=2000000, na_values='', names=puf_columns, dtype=puf_types,
                         usecols=sel, header=0)
    #accumulate chunks in list
    pd_lst=[]
    for chunk in reader:
        pd_lst+=[chunk]
    return pd_lst


def readBCH():
    bch_f ='cancer_state.csv'
    bch_path = os.path.join(get_path(), bch_f)
    if not os.path.isfile(bch_path):
        print "Downloading Cancer CSV file -- download may take awhile..."
        download('https://opendata.socrata.com/api/views/mqh4-spnv/rows.csv?accessType=DOWNLOAD', bch_path)
        print "Cancer CSV download complete"
    columns = ['indicator', 'year', 'gender', 'race', 'value', 'place']
    # type = [('indicator', 'S10'), ('year', np.uint64), ('gender', 'S10'), ('race', 'S10'), ('value', np.float64), ('place'. 'S50')]
    df = pd.read_csv(os.path.join(get_path(), 'cancer_state.csv'), sep=',', names=columns, header=0, na_values='')
    df['place']=df['place'].apply(lambda x: x[-2:]) #filter for only state code
    return df


# def init_db():
#
#     #Create engine to store data in local directory's db file
#     db_name = os.path.basename(app.config['SQLALCHEMY_DATABASE_URI'])
#     db_path=os.path.join(get_path(), db_name) #hardcode
#
#     db_is_new = not os.path.exists(db_path)
#     if db_is_new:
#         print "Database created, creating table(s) schema..."
#         #Remove spontaneous quoting of column name
#         db.engine.dialect.identifier_preparer.initial_quote = ''
#         db.engine.dialect.identifier_preparer.final_quote = ''
#
#         #Create schema -- all tables in the engine -- equivalent to SQL "Create Table"
#         db.create_all() # sqlalchemy lib -- Base.metadata.create_all(bind=engine)
#         print "Table(s) schema created, inserting data..."
#
#         #Insert Data -- Bulk insert of DataFrame
#         df_report = readCSV()
#         report_lst = df_report.to_dict(orient='records')  # orient by records to align format
#         db.session.execute(Report.__table__.insert(), report_lst)
#         db.session.commit()
#         df_puf = readPUF()
#         for elem in df_puf[:5]:
#             puf_lst = elem.to_dict(orient='records')
#             db.session.execute(Puf.__table__.insert(), puf_lst)
#             db.session.commit()
#         df_can = readBCH()
#         can_lst = df_can.to_dict(orient='records')
#         db.session.execute(Cancer.__table__.insert(), can_lst)
#
#         db.session.commit() #Commit
#         db.session.close() #close session
#         print "Data insert successful...database initialization complete"
#         return
#
#     else:
#         print "Database exists; opened successfully"
#         db.session.commit()  # Commit
#         db.session.close()  # close session
#         return
