from analytics.covid19 import Covid19

from string import Template
import os
import locale
locale.setlocale(locale.LC_ALL, 'en_US')


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '../template/')


def weekly_report(date, state=None, county=None):
    with open(os.path.join(TEMPLATE_DIR, 'weekly_report.txt')) as f:
        template_raw = f.read()
    template = Template(template_raw)

    covid = Covid19()
    new_confirmed_cases = covid.confirmed_cases(date, state=state, county=county, span=7)
    cumulative_confirmed_cases = covid.confirmed_cases(date, state=state, county=county, span=999)
    new_death_cases = 123  # covid.death_cases(date, state=state, county=county, span=7)
    cumulative_death_cases = 1234  # covid.death_cases(date, state=state, county=county, span=999)
    growth_rate = 10.0  # covid.growth_rate(date, state=state, county=county, span=7)
    death_rate = 3.0  # covid.death_rate(date, state=state, county=county, span=7)
    trend = 'higher'
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


if __name__ == '__main__':
    weekly_report('2020-05-02', state='California')
