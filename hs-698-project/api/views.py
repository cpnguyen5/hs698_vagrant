from api import app, db
from flask import render_template, url_for
import os
import numpy as np
import pandas as pd
from .models import Report, Puf, Cancer
from sqlalchemy import func, desc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


def get_abs_path():
    """
    This function takes no parameters and returns the api root directory pathway.
    :return: api directory pathway
    """
    return os.path.abspath(os.path.dirname(__file__))


@app.route('/')
def home():
    return render_template("home.html", img_file=url_for('static', filename='img/cms_logo.jpg'))


@app.route('/prevalence')
def prevalence():
    rows = db.session.query(Report.provider_state_code,func.avg(Report.percent_of_beneficiaries_identified_with_cancer),
                            func.avg(Report.percent_of_beneficiaries_identified_with_atrial_fibrillation),
                            func.avg(Report.percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia),
                            func.avg(Report.percent_of_beneficiaries_identified_with_asthma),
                            func.avg(Report.percent_of_beneficiaries_identified_with_heart_failure),
                            func.avg(Report.percent_of_beneficiaries_identified_with_chronic_kidney_disease),
                            func.avg(Report.percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease),
                            func.avg(Report.percent_of_beneficiaries_identified_with_depression),
                            func.avg(Report.percent_of_beneficiaries_identified_with_diabetes),
                            func.avg(Report.percent_of_beneficiaries_identified_with_hyperlipidemia),
                            func.avg(Report.percent_of_beneficiaries_identified_with_hypertension),
                            func.avg(Report.percent_of_beneficiaries_identified_with_ischemic_heart_disease),
                            func.avg(Report.percent_of_beneficiaries_identified_with_osteoporosis),
                            func.avg(Report.percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis),
                            func.avg(Report.percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders),
                            func.avg(Report.percent_of_beneficiaries_identified_with_stroke)).\
        order_by(Report.provider_state_code).group_by(Report.provider_state_code).all()

    overall_prev = db.session.query(func.avg(Report.percent_of_beneficiaries_identified_with_cancer),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_atrial_fibrillation),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_alzheimers_disease_or_dementia),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_asthma),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_heart_failure),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_chronic_kidney_disease),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_chronic_obstructive_pulmonary_disease),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_depression),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_diabetes),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_hyperlipidemia),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_hypertension),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_ischemic_heart_disease),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_osteoporosis),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_rheumatoid_arthritis_osteoarthritis),
                                    func.avg(
                                        Report.percent_of_beneficiaries_identified_with_schizophrenia_other_psychotic_disorders),
                                    func.avg(Report.percent_of_beneficiaries_identified_with_stroke)).all()[0]
    state_avg = []
    for elem in rows:
        state_tup = tuple()
        state_tup += (elem[0],)
        i=1
        while i < len(elem):
            state_tup += (round(elem[i],2),)
            i+=1
        state_avg += [state_tup]
    overall_round = []
    for i in range(len(overall_prev)):
        overall_round += [round(float(overall_prev[i]), 2)]
    state_avg += [('Total Avg',) + tuple(overall_round)]
    diseases = ["Cancer", "A-Fib", "Alzheimers", "Asthma", "Heart Fail",
                "Kidney Dis", "Pulmonary Dis", "Depression", "Diabetes",
                "Hyperlipidemia", "Hypertension", "Ischemic Heart Dis", "Osteoporosis",\
                "Rheumatoid Arthritis", "Schizophrenia", "Stroke"]
    overall_freq = []
    for i in range(len(overall_round)):
        overall_freq += [round((overall_round[i] / 100), 5)]
    overall_bar = pd.DataFrame(np.column_stack((diseases, overall_freq)), columns=['diseases', 'frequency'])
    top_diseases = overall_bar.sort_values(['frequency'], ascending=False).as_matrix()[:5]
    top_perc=[]
    for elem in top_diseases:
        dis = [elem[0]]
        dis +=[str(float(elem[1])*100)]
        top_perc += [dis]
    tsv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'overall_prev.tsv')
    overall_bar.to_csv(tsv_path, sep='\t', header=['disease', 'frequency'])
    return render_template("state.html", rows=state_avg, top_disease = top_perc,
                           prev_js=url_for('static', filename='js/prevalence.v3.min.js'),
                           prev_js1=url_for('static', filename='js/prev.tip.v0.6.3.js'),
                           prev_file=url_for('static', filename='tmp/overall_prev.tsv'))


@app.route('/cancer')
def map():

    rows = db.session.query(Report.provider_state_code, func.avg(Report.percent_of_beneficiaries_identified_with_cancer)).\
        filter(Report.provider_state_code != 'DC').order_by(Report.provider_state_code).group_by(Report.provider_state_code).all()

    state_lst=[]
    for i in range(len(rows)):
        state = tuple()
        state += (str(rows[i][0]),)
        state += (round(rows[i][1],2),)
        state_lst+=[state]
    dict_state = dict(state_lst)

    us_avg = np.average(np.array(state_lst)[:,1].astype(np.float64))
    rows_cancer_cost = db.session.query(Report.provider_state_code,
                                        func.avg(Report.percent_of_beneficiaries_identified_with_cancer),
                                        func.avg(Report.total_medicare_standardized_payment_amount)).order_by(
        Report.provider_state_code). \
        group_by(Report.provider_state_code).all()


    state_costs_lst = []
    for i in range(len(rows_cancer_cost)):
        state_costs = tuple()
        state_costs += (str(rows_cancer_cost[i][0]),)
        state_costs += (round(float(rows_cancer_cost[i][1]) * float(rows_cancer_cost[i][2]), 2),)
        state_costs_lst += [state_costs]
    col = ['state', 'costs']
    state_costs_df = pd.DataFrame(state_costs_lst, columns=col)
    sorted_state_costs_df = state_costs_df.sort_values(['costs'], ascending=False)
    lowest_state_costs = sorted_state_costs_df.iloc[-5:].as_matrix()[::-1]
    highest_state_costs = sorted_state_costs_df.iloc[:5].as_matrix()
    max_state = [highest_state_costs[0][0], "%.2f" % (highest_state_costs[0][1])]
    min_state = [lowest_state_costs[0][0], "%.2f" % (lowest_state_costs[0][1])]
    cancer_cost_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cancer_costs.tsv')
    state_costs_df.to_csv(cancer_cost_path, sep='\t', header=col)

    #histogram
    plt.figure()
    rows_dist = db.session.query(Report.percent_of_beneficiaries_identified_with_cancer).all()
    dist_df = pd.DataFrame(rows_dist, columns=['cancer_distribution']).dropna()

    h = dist_df['cancer_distribution'].plot.hist(bins=50, figsize=(10, 7), color='green',
                                                 title='Histogram of Cancer Prevalence amongst CMS Beneficiaries')
    h.set_xlabel('Prevalence (%)')
    hist_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cancer_dist.png')
    h.figure.savefig(hist_path, transparent=True) #save figure
    plt.close()
    #calculate quartiles
    canc_dist = dist_df['cancer_distribution'].copy()
    canc_dist.sort_values(inplace=True)
    q1, q2, q3 =canc_dist.quantile([0.25, 0.5, 0.75]) #quartiles
    irq = q3 - q1
    outlier = {'upper': q3 + 1.5 * irq, 'lower': q3 - 1.5 * irq} #calculate upper & lower bound outliers

    return render_template("map.html", d_state=dict_state, rows=state_lst, us_avg=us_avg, outlier=outlier,
                           low_cost = lowest_state_costs, high_cost = highest_state_costs, max_state=max_state,
                           min_state=min_state,
                           chist_fig=url_for('static', filename='tmp/cancer_dist.png'),
                           js_file=url_for('static', filename='js/datamaps.usa.min.js'),
                           cancer_js = url_for('static', filename='js/cancer.v3.min.js'),
                           topo_js = url_for('static', filename='js/cancer_topojson.v1.min.js'),
                           cancer_costs_file=url_for('static', filename='tmp/cancer_costs.tsv'))


@app.route('/cancer/risk')
def risks():
    rows_age = db.session.query(Report.percent_of_beneficiaries_identified_with_cancer,
                                Report.number_of_beneficiaries_age_less_65, Report.number_of_beneficiaries_age_65_to_74,
                                Report.number_of_beneficiaries_age_75_to_84,
                                Report.number_of_beneficiaries_age_greater_84).all()
    col_c_age = ['prevalence', 'age_less_65', 'age_65_74', 'age_75_84', 'age_greater_84']
    canc_age_df = pd.DataFrame(rows_age, columns=col_c_age)

    #Extract categorical variables - age
    c_age_prev = canc_age_df['prevalence'] / 100
    prev_age_0 = (c_age_prev * canc_age_df['age_less_65']).dropna().as_matrix()
    prev_age_65 = (c_age_prev * canc_age_df['age_65_74']).dropna().as_matrix()
    prev_age_75 = (c_age_prev * canc_age_df['age_75_84']).dropna().as_matrix()
    prev_age_85 = (c_age_prev * canc_age_df['age_greater_84']).dropna().as_matrix()
    # b = map(lambda x: "age_less_65", canc_age_0)
    canc_age_0 = np.column_stack((['age_less_65']*len(prev_age_0), prev_age_0))
    canc_age_65 = np.column_stack((['age_65_74']*len(prev_age_65), prev_age_65))
    canc_age_75 = np.column_stack((['age_75_84']*len(prev_age_75), prev_age_75))
    canc_age_85 = np.column_stack((['age_greater_84']*len(prev_age_85), prev_age_85))
    canc_age = np.vstack((canc_age_0, canc_age_65, canc_age_75, canc_age_85))
    age_dist_df = pd.DataFrame({'age': canc_age[:,0],
                                'Num_with_Cancer': canc_age[:,1].astype(np.float64)})
    #determine outlier bounds
    age_dist = age_dist_df['Num_with_Cancer'].copy()
    age_dist.sort_values(inplace=True)
    aq1, aq2, aq3 =age_dist.quantile([0.25, 0.5, 0.75]) #quartiles
    airq = aq3 - aq1
    outlier = {'upper': aq3 + 1.5 * airq, 'lower': aq3 - 1.5 * airq}
    #box and whisker plot
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=2.0)
    age_plot = sns.boxplot(x='age', y='Num_with_Cancer', data=age_dist_df[age_dist_df['Num_with_Cancer'] < outlier['upper']+30])
    age_plot.set(title='Cancer Prevalence by Age Groups')
    age_box_path = os.path.join(get_abs_path(), 'static', 'tmp', 'age_prev.png')
    age_plot.figure.savefig(age_box_path, transparent=True) #save figure
    plt.close()
    ##race
    rows_race = db.session.query(Report.percent_of_beneficiaries_identified_with_cancer,
                                Report.number_of_non_hispanic_white_beneficiaries,
                                Report.number_of_african_american_beneficiaries,
                                Report.number_of_asian_pacific_islander_beneficiaries,
                                Report.number_of_hispanic_beneficiaries,
                                Report.number_of_american_indian_alaskan_native_beneficiaries,
                                Report.number_of_beneficiaries_with_race_not_elsewhere_classified).all()

    col_c_race = ['prevalence', 'white', 'african_am', 'api', 'hispanic', 'native_am', 'other_race']
    canc_race_df = pd.DataFrame(rows_race, columns=col_c_race)
    # Extract categorical variables - age
    c_race_prev = canc_race_df['prevalence'] / 100
    prev_white = (c_race_prev * canc_race_df['white']).dropna().as_matrix()
    prev_afric = (c_race_prev * canc_race_df['african_am']).dropna().as_matrix()
    prev_api = (c_race_prev * canc_race_df['api']).dropna().as_matrix()
    prev_hispanic = (c_race_prev * canc_race_df['hispanic']).dropna().as_matrix()
    prev_native = (c_race_prev * canc_race_df['native_am']).dropna().as_matrix()
    prev_other = (c_race_prev * canc_race_df['other_race']).dropna().as_matrix()
    canc_white = np.column_stack((['White'] * len(prev_white), prev_white))
    canc_afric = np.column_stack((['African_American'] * len(prev_afric), prev_afric))
    canc_api = np.column_stack((['API'] * len(prev_api), prev_api))
    canc_hispanic = np.column_stack((['Hispanic'] * len(prev_hispanic), prev_hispanic))
    canc_native = np.column_stack((['Native_American'] * len(prev_native), prev_native))
    canc_other = np.column_stack((['Other'] * len(prev_other), prev_other))
    canc_race = np.vstack((canc_white, canc_afric, canc_api, canc_hispanic, canc_native, canc_other))
    race_dist_df = pd.DataFrame({'race': canc_race[:, 0],
                                'Num_with_Cancer': canc_race[:, 1].astype(np.float64)})
    #determine outlier bounds
    race_dist = race_dist_df['Num_with_Cancer'].copy()
    race_dist.sort_values(inplace=True)
    rq1, rq2, rq3 = race_dist.quantile([0.25, 0.5, 0.75])  # quartiles
    rirq = rq3 - rq1
    outlier = {'upper': rq3 + 1.5 * rirq, 'lower': rq3 - 1.5 * rirq}
    # box and whisker plot
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=2.0)
    race_plot = sns.boxplot(x='race', y='Num_with_Cancer', data=race_dist_df[race_dist_df['Num_with_Cancer'] <
                                                                             outlier['upper'] + 20])
    race_plot.set(title='Cancer Prevalence by Race')
    race_box_path = os.path.join(get_abs_path(), 'static', 'tmp', 'race_prev.png')
    race_plot.figure.savefig(race_box_path, transparent=True)  # save figure
    plt.close()

    # mortality rate
    mort_rows = db.session.query(Cancer.race, func.avg(Cancer.value)).filter(Cancer.race != 'Multiracial').\
        filter(Cancer.race != 'All').filter(Cancer.race != 'American Indian/Alaska Native').\
        group_by(Cancer.race).order_by(Cancer.race).all()
    mort_df = pd.DataFrame(mort_rows, columns=['Race', 'Avg_Mortality_Rate'])
    plt.figure(figsize=(16,10))
    sns.set(font_scale=2.0)
    mortbar_plot = sns.barplot(x="Race", y="Avg_Mortality_Rate", data=mort_df)
    mortbar_plot.set(title='Average Mortality Rate by Race')
    mortbar_path = os.path.join(get_abs_path(), 'static', 'tmp', 'mort_bar.png')
    mortbar_plot.figure.savefig(mortbar_path, transparent=True)  # save figure
    plt.close()

    #annual mortality rate
    year_mort_rows = db.session.query(Cancer.year,Cancer.race, func.avg(Cancer.value)).\
        group_by(Cancer.year, Cancer.race).filter(Cancer.race != 'Multiracial').filter(Cancer.race != 'All').\
        filter(Cancer.race != 'American Indian/Alaska Native').all()
    annual_mort_df = pd.DataFrame(year_mort_rows, columns=['Year','Race', 'Mortality_Rate'])
    plt.figure(figsize=(16,10))
    sns.set(font_scale=1.8)
    annual_plot = sns.pointplot(x="Year", y="Mortality_Rate", hue="Race", data =annual_mort_df)
    plt.legend(bbox_to_anchor=(.90, 1), loc=2)
    annual_plot.set(title='Annual Average Mortality Rate by Race')
    annual_path = os.path.join(get_abs_path(), 'static', 'tmp', 'mort_year.png')
    annual_plot.figure.savefig(annual_path, transparent=True)  # save figure
    plt.close()
    return render_template('cancer_risks.html', risk_img=url_for('static', filename='img/riskfactor.png'),
                           age_boxfig=url_for('static', filename='tmp/age_prev.png'),
                           race_boxfig=url_for('static', filename='tmp/race_prev.png'),
                           mortbar_fig=url_for('static', filename='tmp/mort_bar.png'),
                           mort_fig=url_for('static', filename='tmp/mort_year.png'))


@app.route('/cost')
def cost():

    rows = db.session.query(Report.provider_state_code, func.avg(Report.total_medicare_standardized_payment_amount)).\
        filter(Report.provider_state_code != 'DC').order_by(Report.provider_state_code).group_by(Report.provider_state_code).all()

    state_lst=[]
    for i in range(len(rows)):
        state = tuple()
        state += (rows[i][0],)
        state += (round(rows[i][1],2),)
        state_lst+=[state]
    state_cost=pd.DataFrame(state_lst, dtype=int)
    csv_path = os.path.join(get_abs_path(), 'static', 'tmp', 'state_cost.csv')
    state_cost.to_csv(csv_path, index=False, header= ["name","value"])


    top_rows = db.session.query(Report.provider_state_code, func.avg(Report.total_medicare_standardized_payment_amount),
                            func.avg(Report.number_of_beneficiaries_age_less_65),
                            func.avg(Report.number_of_beneficiaries_age_65_to_74),
                            func.avg(Report.number_of_beneficiaries_age_75_to_84),
                            func.avg(Report.number_of_beneficiaries_age_greater_84)). \
        filter(Report.provider_state_code != 'DC').order_by(
        func.avg(Report.total_medicare_standardized_payment_amount).desc()). \
        group_by(Report.provider_state_code).limit(5).all()
    data = []
    for row in top_rows:
        state_sum=float(np.sum(row[2:]))
        state_cost=tuple()
        state_cost+=(row[0],) #state
        state_cost+=(round(row[1], 2),) #total payment amount
        state_cost+=( int(((float(row[2])) / state_sum) * row[1]),) #<65
        state_cost+=( int(((float(row[3])) / state_sum) * row[1]),) #65 to 74
        state_cost+=( int(((float(row[4])) / state_sum) * row[1]),) #75 to 84
        state_cost+=( int(((float(row[5])) / state_sum) * row[1]),) #>85
        data+=[state_cost]

    #costs by facility
    rows_total = db.session.query(Puf.place_of_service,
                                  func.avg(Report.total_medicare_standardized_payment_amount)).\
        join(Report, Report.npi == Puf.npi).group_by(Puf.place_of_service).all()
    rows_med = db.session.query(Puf.place_of_service,
                                  func.avg(Report.total_medical_medicare_standardized_payment_amount)). \
        join(Report, Report.npi == Puf.npi).group_by(Puf.place_of_service).all()
    rows_drug = db.session.query(Puf.place_of_service,
                                  func.avg(Report.total_drug_medicare_standardized_payment_amount)). \
        join(Report, Report.npi == Puf.npi).group_by(Puf.place_of_service).all()
    lst_total = []
    for elem in rows_total:
        if str(elem[0]) == 'F':
            new_row = ['facility'] + [elem[1]] + ['Total']
        else:
            new_row = ['office'] + [elem[1]] + ['Total']
        lst_total += [new_row]
    for elem in rows_med:
        if str(elem[0]) == 'F':
            new_row = ['facility'] + [elem[1]] + ['Medical']
        else:
            new_row = ['office'] + [elem[1]] + ['Medical']
        lst_total += [new_row]
    for elem in rows_drug:
        if str(elem[0]) == 'F':
            new_row = ['facility'] + [elem[1]] + ['Drug']
        else:
            new_row = ['office'] + [elem[1]] + ['Drug']
        lst_total +=[new_row]

    facil_df = pd.DataFrame(lst_total, columns=['location', 'amount', 'cost type'])
    facil_grp = facil_df.groupby(facil_df['location'])
    facil_mean = facil_grp.mean().as_matrix()
    facil_std = facil_grp.std().as_matrix()
    for i in range(len(facil_mean)):
        facil_mean[i] = round(facil_mean[i], 2)
        facil_std[i] = round(facil_std[i], 2)

    plt.figure()
    sns.set(font_scale=1.0)
    facil_plot = sns.factorplot(x='cost type', y='amount', hue='location', data = facil_df, kind='bar')
    facil_plot.set_ylabels("Average Costs ($)")
    facil_path = os.path.join(get_abs_path(), 'static', 'tmp', 'facil_cost.png')
    facil_plot.savefig(facil_path, transparent=True)  # save figure
    plt.close()
    #ratio of number of services amongst facility type
    row_ratio = db.session.query(Puf.place_of_service, func.sum(Report.number_of_services)).\
        join(Report, Report.npi == Puf.npi).group_by(Puf.place_of_service).all()
    total_services = row_ratio[0][1] + row_ratio[1][1]
    perc_f = (float(row_ratio[0][1]) / total_services) * 100
    perc_o = (float(row_ratio[1][1]) / total_services) * 100
    num_f = [round(perc_f, 2), row_ratio[0][1]]
    num_o = [round(perc_o, 2), row_ratio[1][1]]

    return render_template("state_cost.html", num_f=num_f, num_o=num_o, mean=facil_mean, std=facil_std,
                           data_file = url_for('static', filename='tmp/state_cost.csv'), data=data,
                           facil_fig = url_for('static', filename='tmp/facil_cost.png'),
                           cost_js = url_for('static', filename='js/cost.v3.min.js'))


@app.route('/cost/demo')
def demographics():
    ##age
    rows_age = db.session.query(func.sum(Report.total_medicare_standardized_payment_amount),
                                func.sum(Report.total_medical_medicare_standardized_payment_amount),
                                func.sum(Report.total_drug_medicare_standardized_payment_amount),
                                func.sum(Report.number_of_beneficiaries_age_less_65),
                                func.sum(Report.number_of_beneficiaries_age_65_to_74),
                                func.sum(Report.number_of_beneficiaries_age_75_to_84),
                                func.sum(Report.number_of_beneficiaries_age_greater_84)).all()
    rows_age = list(rows_age[0])
    total_age = sum(rows_age[3:])
    age_0_64 = float(rows_age[3]) / total_age
    age_65_74 = float(rows_age[4]) / total_age
    age_75_84 = float(rows_age[5]) / total_age
    age_85 = float(rows_age[6]) / total_age
    age = [age_0_64, age_65_74, age_75_84, age_85]
    medicare_amt_age = []
    medicare_medical_amt_age = []
    medicare_drug_amt_age = []
    for i in range(len(age)):
        medicare_amt_age += [round(rows_age[0] * age[i],2)]
        medicare_medical_amt_age += [round(rows_age[1] * age[i],2)]
        medicare_drug_amt_age += [round(rows_age[2] * age[i],2)]
    costs = ['Medicare Amount ($)', 'Medical Amount ($)', 'Drug Amount ($)']
    age_data = np.vstack((medicare_amt_age, medicare_medical_amt_age, medicare_drug_amt_age))
    age_df = pd.DataFrame({'costs': costs,
                           'age 0-64': age_data[:,0],
                           'age 65-74': age_data[:,1],
                           'age 75-84': age_data[:,2],
                           'age 84+': age_data[:,3]})
    age_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cost_age.csv')
    age_df.to_csv(age_path, sep=',', index=False)
    costs_age = np.column_stack((costs+['Total'], np.vstack((age_data, np.sum(age_data, axis=0)))))
    age_perc = np.round(age, 4) * 100
    age_ratio = np.column_stack((['Count (n)', 'Percentage(%)'],np.vstack((rows_age[3:], age_perc))))


    ##race
    rows_race = db.session.query(func.sum(Report.number_of_non_hispanic_white_beneficiaries),
                                 func.sum(Report.number_of_african_american_beneficiaries),
                                 func.sum(Report.number_of_asian_pacific_islander_beneficiaries),
                                 func.sum(Report.number_of_hispanic_beneficiaries),
                                 func.sum(Report.number_of_american_indian_alaskan_native_beneficiaries),
                                 func.sum(Report.number_of_beneficiaries_with_race_not_elsewhere_classified)).all()
    rows_race = rows_race[0]
    total_race = sum(rows_race)
    white = float(rows_race[0]) / total_race
    african_am = float(rows_race[1]) / total_race
    api = float(rows_race[2]) / total_race
    hispanic = float(rows_race[3]) / total_race
    native_am = float(rows_race[4]) / total_race
    other_race = float(rows_race[5]) / total_race
    race = [white, african_am, api, hispanic, native_am, other_race]
    medicare_amt_race = []
    medicare_medical_amt_race = []
    medicare_drug_amt_race = []
    for i in range(len(race)):
        medicare_amt_race += [round(rows_age[0] * race[i],2)]
        medicare_medical_amt_race += [round(rows_age[1] * race[i],2)]
        medicare_drug_amt_race += [round(rows_age[2] * race[i],2)]
    race_data = np.vstack((medicare_amt_race, medicare_medical_amt_race, medicare_drug_amt_race))
    race_df = pd.DataFrame({'costs': costs,
                            'White': race_data[:, 0],
                            'African-American': race_data[:, 1],
                            'Asian-Pacific Islander': race_data[:, 2],
                            'Hispanic': race_data[:, 3],
                            'Native American': race_data[:, 4],
                            'Other Race': race_data[:, 5]})
    race_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cost_race.csv')
    race_df.to_csv(race_path, sep=',', index=False)
    costs_race = np.column_stack((costs+['Total'], np.vstack((race_data, np.sum(race_data, axis=0)))))
    race_perc = np.round(race, 4) * 100
    race_ratio = np.column_stack((['Count (n)', 'Percentage(%)'],np.vstack((rows_race[:6], race_perc))))

    ##sex
    rows_sex = db.session.query(func.sum(Report.number_of_female_beneficiaries),
                                 func.sum(Report.number_of_male_beneficiaries)).all()

    rows_sex = rows_sex[0]
    total_sex = sum(rows_sex)
    female = float(rows_sex[0]) / total_sex
    male = float(rows_sex[1]) / total_sex
    sex = [female, male]
    medicare_amt_sex = []
    medicare_medical_amt_sex = []
    medicare_drug_amt_sex = []
    for i in range(len(sex)):
        medicare_amt_sex += [round(rows_age[0] * sex[i], 2)]
        medicare_medical_amt_sex += [round(rows_age[1] * sex[i], 2)]
        medicare_drug_amt_sex += [round(rows_age[2] * sex[i], 2)]
    sex_data = np.vstack((medicare_amt_sex, medicare_medical_amt_sex, medicare_drug_amt_sex))
    sex_df = pd.DataFrame({'costs': costs,
                           'Female': sex_data[:, 0],
                           'Male': sex_data[:, 1]})
    costs_sex = np.column_stack((costs+['Total'], np.vstack((sex_data, np.sum(sex_data, axis=0)))))
    sex_perc = np.round(sex, 4) * 100
    sex_ratio = np.column_stack((['Count (n)', 'Percentage (%)'],np.vstack((rows_sex[:2], sex_perc))))
    sex_path = os.path.join(get_abs_path(), 'static', 'tmp', 'cost_sex.csv')
    sex_df.to_csv(sex_path, sep=',', index=False)

    ##heatmap
    rows_heatmap = db.session.query(Report.total_medicare_standardized_payment_amount,
                                    Report.total_medical_medicare_standardized_payment_amount,
                                    Report.total_drug_medicare_standardized_payment_amount,
                                    Report.number_of_beneficiaries_age_less_65,
                                    Report.number_of_beneficiaries_age_65_to_74,
                                    Report.number_of_beneficiaries_age_75_to_84,
                                    Report.number_of_beneficiaries_age_greater_84,
                                    Report.number_of_non_hispanic_white_beneficiaries,
                                    Report.number_of_african_american_beneficiaries,
                                    Report.number_of_asian_pacific_islander_beneficiaries,
                                    Report.number_of_hispanic_beneficiaries,
                                    Report.number_of_american_indian_alaskan_native_beneficiaries,
                                    Report.number_of_beneficiaries_with_race_not_elsewhere_classified,
                                    Report.number_of_female_beneficiaries, Report.number_of_male_beneficiaries).all()
    col = ['medicare_amount', 'medicare_medical_amount', 'medicare_drug_amount', 'num_age_less_65', 'num_age_65_to_74',
           'num_age_75-84', 'num_age_greater_84', 'num_white', 'num_african_am', 'num_api', 'num_hispanic',
           'num_native_am', 'num_other_race', 'num_female', 'num_male']
    demo_df = pd.DataFrame(rows_heatmap, columns=col)
    demo_corr = demo_df.corr()
    plt.figure()
    sns.set(style='white')
    mask = np.zeros_like(demo_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(16, 12))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap_plot = sns.heatmap(demo_corr, mask=mask, cmap=cmap, ax=ax)
    heatmap_path = os.path.join(get_abs_path(), 'static', 'tmp', 'heatmap_demo.png')
    heatmap_plot.figure.savefig(heatmap_path, transparent=True)
    plt.close()
    return render_template("cost_demo.html", costs_age=costs_age, age_ratio=age_ratio,
                           costs_race=costs_race, race_ratio=race_ratio,
                           costs_sex=costs_sex, sex_ratio=sex_ratio,
                           cost_demo_js = url_for('static', filename='js/cost_demo.v3.min.js'),
                           age_file=url_for('static', filename='tmp/cost_age.csv'),
                           race_file=url_for('static', filename='tmp/cost_race.csv'),
                           sex_file=url_for('static', filename='tmp/cost_sex.csv'),
                           heatmap_fig=url_for('static',
                                       filename='tmp/heatmap_demo.png'))


@app.route('/cost/hcpcs')
def procedure():

    #histogram - unique hcpcs per npi
    rows_unique = db.session.query(Report.number_of_hcpcs).all()
    # rows_unique = list(db.engine.execute("SELECT number_of_HCPCS from report"))

    hcpcs_dist = pd.DataFrame(rows_unique, columns=['number_of_HCPCS'])
    hcpcs_dist = hcpcs_dist[hcpcs_dist['number_of_HCPCS'] < 200]
    plt.figure()
    h = hcpcs_dist['number_of_HCPCS'].plot.hist(bins=20, figsize=(10, 7), color='green')
    h.set_xlabel('Number of Unique HCPCS Services & Procedures',fontsize=18)
    h.set_ylabel('Frequency', fontsize=18)
    h.set_title('Histogram of Unique CMS Services & Procedures Provided', fontsize=18)
    h_path = os.path.join(get_abs_path(), 'static', 'tmp', 'hcpcs_dist.png')
    h.figure.savefig(h_path, transparent=True)  # save figure
    plt.close()
    hcpcs_median = int(hcpcs_dist['number_of_HCPCS'].median())
    hcpcs_mean = int(hcpcs_dist['number_of_HCPCS'].mean())
    hcpcs_mode = scipy.stats.mode(hcpcs_dist['number_of_HCPCS'].as_matrix().flatten())[0][0]

    #bar chart - number of services per cost category
    rows_total = db.session.query(func.sum(Report.number_of_services), func.sum(Report.number_of_medical_services),
                                  func.sum(Report.number_of_drug_services)).all()

    total_data = list(rows_total[0])
    pie_service = total_data[1:] + [total_data[0]-sum(total_data[1:])]
    total_df = pd.DataFrame({'type_serv': ['num_medical_services', 'num_drug_services', 'num_other_services'],
                             'num_serv': pie_service})
    serv_sum = total_df['num_serv'].groupby(total_df['type_serv']).sum()
    plt.figure()
    plt.axis=('equal')
    plt.pie(serv_sum, labels=serv_sum.index, autopct="%1.1f%%")
    plt.suptitle('CMS Service Distribution', fontsize=18)
    pie_path = os.path.join(get_abs_path(), 'static', 'tmp', 'num_pie.png')
    plt.savefig(pie_path, transparent=True)  # save figure
    plt.close()

    #leading hcpcs services
    ##most frequently utilized HCPCS
    rows_freq = db.session.query(Puf.hcpcs_code, func.count(Puf.hcpcs_code)).group_by(Puf.hcpcs_code).\
        order_by(desc(func.count(Puf.hcpcs_code))).limit(10).all()

    freq_serv = []
    for i in range(len(rows_freq)):
        code = str(rows_freq[i][0])
        code_info = db.session.query(Puf.hcpcs_description).filter(Puf.hcpcs_code == code).first()
        code_amt = db.session.query(func.avg(Puf.average_medicare_standardized_amount)).\
            filter(Puf.hcpcs_code == code).first()
        freq_row = (code,) + code_info + (int(rows_freq[i][1]),) + (round(float(code_amt[0]), 2),)
        freq_serv += [freq_row]
    freq_serv = np.array(freq_serv)

    ##most expensive HCPCS
    rows_exp = db.session.query(Puf.hcpcs_code, func.avg(Puf.average_medicare_standardized_amount)).\
        filter(Puf.hcpcs_code != '').group_by(Puf.hcpcs_code).\
        order_by(desc(func.avg(Puf.average_medicare_standardized_amount))).limit(10).all()

    exp_serv = []
    for i in range(len(rows_exp)):
        exp_code = str(rows_exp[i][0])
        exp_code_info = db.session.query(Puf.hcpcs_description).filter(Puf.hcpcs_code == exp_code).first()
        exp_code_count = db.session.query(func.count(Puf.hcpcs_code)).filter(Puf.hcpcs_code == exp_code).all()
        exp_row = (exp_code,) + exp_code_info + (int(exp_code_count[0][0]),) + (round(rows_exp[i][1], 2),)
        exp_serv += [exp_row]
    exp_serv = np.array(exp_serv)

    #correlation - cost & number of services
    rows_corr = db.session.query(Report.number_of_services, Report.number_of_hcpcs,
                                 Report.total_medicare_standardized_payment_amount,
                                 Report.total_medical_medicare_standardized_payment_amount,
                                 Report.total_drug_medicare_standardized_payment_amount).all()
    service_cost_df = pd.DataFrame(rows_corr, columns=['num_services', 'num_unique_HCPCS','total_overall_cost',
                                                       'total_medical_costs', 'total_drug_costs'])

    service_corr = service_cost_df.corr()
    plt.figure()
    sns.set(style='white', font_scale=1.5)
    mask = np.zeros_like(service_corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(16, 12))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    servicecorr_plot = sns.heatmap(service_corr, mask=mask, cmap=cmap, ax=ax)
    scorr_path = os.path.join(get_abs_path(), 'static', 'tmp', 'heatmap_service.png')
    servicecorr_plot.figure.savefig(scorr_path, transparent=True)
    plt.close()
    return render_template("cost_hcpcs.html", unique_fig=url_for('static', filename='tmp/hcpcs_dist.png'),
                           pie_fig=url_for('static', filename='tmp/num_pie.png'), total_serv =total_data,
                           median=hcpcs_median, avg=hcpcs_mean, mode=hcpcs_mode, freq_serv=freq_serv, exp_serv=exp_serv,
                           scorr_heatmap=url_for('static', filename='tmp/heatmap_service.png'))


@app.route('/data')
def data():
    return render_template("data.html", cms_img=url_for('static', filename='img/cms_logo.jpg'),
                           bchc_img=url_for('static', filename='img/bch_logo.png'))


@app.route('/data/report')
def report():
    return render_template("report_data.html")


@app.route('/data/puf')
def puf():
    return render_template("puf_data.html")


@app.route('/data/cancer')
def cancer():
    return render_template("cancer_data.html")


