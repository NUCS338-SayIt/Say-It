from analytics.covid19 import Covid19

from string import Template
import os
import random
import locale
import datetime
import pandas as pd
import numpy as np
locale.setlocale(locale.LC_ALL, 'en_US')


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '../template/')


transitionWords = {
    'similarity': ['likewise', 'similarly', 'together with'],
    'opposition': ['in contrast', 'on the contrary', 'however', 'while']
}

degreeAdverbs = {
    '-1': ['slightly', 'gradually', 'marginally', 'slightly'],
    '0': ['consistently', 'steadily'],
    '1': ['considerably', 'dramatically', 'enormously', 'remarkably']
}


def load_template(render_type, trend=None):
    dir = os.path.join(TEMPLATE_DIR, render_type)
    filename = os.path.join(dir, '{}_{}'.format(render_type, trend)) if trend else os.path.join(dir, render_type)
    with open(filename, 'r') as f:
        template_raw = f.read()
    return Template(template_raw)


def random_transition(key):
    return random.choice(transitionWords[key])


def random_degree(key):
    return random.choice(degreeAdverbs[key])

def report_sequence(date, state=None, county=None, my_span=1):
    covid = Covid19()

    # if the span and date is valid
    flag = covid.valid_span(date, state=state, county=county, span=my_span)
    if flag == False:
        print("Error. Date or span is out of range./n")
        return -1

    #Processing
    today = datetime.datetime.strptime(date, '%Y-%m-%d')
    delta = datetime.timedelta(days=my_span)
    pre_date = (today - delta).strftime('%Y-%m-%d')
    new_confirmed_cases, cumulative_confirmed_cases = covid.confirmed_cases(date, state=state, county=county, span=my_span)
    old_confirmed_cases, oldcumulative_confirmed_cases = covid.confirmed_cases(pre_date, state=state, county=county, span=my_span)
    new_death_cases, cumulative_death_cases = covid.death_cases(date, state=state, county=county, span=my_span)
    old_death_cases, oldcumulative_death_cases = covid.death_cases(pre_date, state=state, county=county, span=my_span)
    growth_rate = round(covid.growth_rate(date, state=state, county=county, span=my_span) * 100, 1)
    old_growth_rate = round(covid.growth_rate(pre_date, state=state, county=county, span=my_span) * 100, 1)
    death_rate = round(covid.death_rate(date, state=state, county=county, span=my_span)[0] * 100, 1)
    old_death_rate = round(covid.death_rate(pre_date, state=state, county=county, span=my_span)[0] * 100, 1)
    per_capita_scale = 100000
    most_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale,
                                                                            span=my_span)
    old_most_confirmed = covid.most_confirmed_cases_per_capita(pre_date, state=state, scale=per_capita_scale,
                                                                            span=my_span)
    least_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale,
                                                                             span=my_span, type='least')
    old_least_confirmed = covid.most_confirmed_cases_per_capita(pre_date, state=state, scale=per_capita_scale,
                                                                             span=my_span, type='least')
    ratio = round(most_confirmed_cases_per_capita[0] / least_confirmed_cases_per_capita[0], 1)
    old_ratio = round(old_most_confirmed[0] / old_least_confirmed[0], 1)
    print('Data Collected')

    # Correlation Matrix
    my_index = ['new confirmed cases', 'cumulative confirmed cases',
                'new death cases', 'cumulative death cases', 'growth rate', 'death rate',
                'most confirmed cases per capita', 'least confirmed cases per capita', 'ratio']
    ar = np.zeros((9,9))
    corr_df = pd.DataFrame(ar, index = my_index, columns=my_index)

    # Rate calculation
    new_data = [new_confirmed_cases, cumulative_confirmed_cases, new_death_cases, cumulative_death_cases
                , growth_rate, death_rate, most_confirmed_cases_per_capita, least_confirmed_cases_per_capita,
                ratio]
    old_data = [old_confirmed_cases, oldcumulative_confirmed_cases, old_death_cases, oldcumulative_death_cases
                , old_growth_rate, old_death_rate, old_most_confirmed, old_least_confirmed, old_ratio]
    # Zero division Exception is not handled
    change_rate = list(map(lambda x: (x[0] - x[1]) / x[1], zip(new_data, old_data)))

    # Output a list of map with key Names, Current Value, Previous Value, Change Rate
    sequence = []
    temp_list = map(list, zip(my_index, new_data, old_data, change_rate))
    for ele in temp_list:
        temp_dict = dict(zip(['Name', 'Current Value', 'Previous Value', 'Change Rate'], ele))
        sequence.append(temp_dict)

    # Score of changing rate
    temp = sorted(change_rate, key=abs, reverse=True)
    change_score = []
    for ele in change_rate:
        change_score.append(round(temp.index(ele) / 10.0 + 0.1, 1))

    # Decide the returning order
    res_order = []
    temp_index = my_index
    a1 = my_index[change_score.index(max(change_score))]
    res_order.append(a1)
    temp_index.remove(a1)
    while len(temp_index):
        previous = a1
        rest_score = []
        for ele in temp_index:
            score1 = change_score[my_index.index(ele)]
            score2 = corr_df.loc[previous][ele]
            score = 0.5 * score1 + 0.5 * score2
            rest_score.append(score)
        max_iloc = rest_score.index(max(rest_score))
        previous = temp_index[max_iloc]
        temp_index.remove(previous)
        res_order.append(previous)
    return sequence, res_order





def weekly_report(date, state=None, county=None):
    with open(os.path.join(TEMPLATE_DIR, 'weekly_report')) as f:
        template_raw = f.read()
    template = Template(template_raw)


    covid = Covid19()
    new_confirmed_cases, cumulative_confirmed_cases = covid.confirmed_cases(date, state=state, county=county, span=7)
    new_death_cases, cumulative_death_cases = covid.death_cases(date, state=state, county=county, span=7)
    growth_rate = round(covid.growth_rate(date, state=state, county=county, span=7) * 100, 1)
    death_rate = round(covid.death_rate(date, state=state, county=county, span=7)[0] * 100, 1)
    trend = 'higher' if covid.death_rate_comparison(date, state=state, county=county, scale='week') == 'increase' else 'lower'
    per_capita_scale = 100000
    most_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale, span=7)
    least_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale, span=7, type='least')
    ratio = round(most_confirmed_cases_per_capita[0] / least_confirmed_cases_per_capita[0], 1)

    d = {'date': date,
         'state': state if state else 'US',
         'new_confirmed_cases': locale.format_string('%d', new_confirmed_cases, grouping=True),
         'cumulative_confirmed_cases': locale.format_string('%d', cumulative_confirmed_cases, grouping=True),
         'new_death_cases': locale.format_string('%d', new_death_cases, grouping=True),
         'cumulative_death_cases': locale.format_string('%d', cumulative_death_cases, grouping=True),
         'growth_rate': growth_rate,
         'death_rate': death_rate,
         'trend': trend,
         'type': 'county' if state else 'state',
         'most_confirmed_cases_per_capita': most_confirmed_cases_per_capita[1],
         'most_confirmed_cases': most_confirmed_cases_per_capita[0],
         'per_capita_scale': locale.format_string('%d', per_capita_scale, grouping=True),
         'ratio': ratio}
    story = template.safe_substitute(d)
    print(story)

    with open('../{}-{}.txt'.format(date, state if state else 'us'), 'a+') as f:
        f.write(story)


if __name__ == '__main__':
    # weekly_report('2020-05-03')
    print(load_template('fatality_rate', 'upward').template)