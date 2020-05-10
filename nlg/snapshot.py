from analytics.covid19 import Covid19

from string import Template
import os
import random
import locale
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
