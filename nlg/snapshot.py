from analytics.covid19 import Covid19

from string import Template
import os
import random
import locale
import datetime
import pandas as pd
import numpy as np
import json
locale.setlocale(locale.LC_ALL, 'en_US')


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '../template/')
# TEMPLATE_DIR = r'C:\Users\86139\Desktop\西北大学\CS338\Project\Say-It\template'


transitionWords = {
    'similarity': ['likewise', 'similarly', 'together with',
                   'moreover', 'furthermore'],
    'opposition': ['in contrast', 'on the contrary', 'however', 'while', 'nevertheless',
                   'nonetheless']
}

degreeAdverbs = {
    '-1': ['slightly', 'gradually', 'marginally', 'slightly'],
    '0': ['consistently', 'steadily'],
    '1': ['considerably', 'dramatically', 'enormously', 'remarkably']
}

covid = Covid19()


def load_template(render_type, trend=None, beginning=False, explanation=False, ending=False):
    if beginning:
        filename = os.path.join(TEMPLATE_DIR, 'beginning.json')
        with open(filename, 'r') as f:
            data = json.load(f)
            # if render_type in data:
            #     template_raw_list = data[render_type][trend]
            # else:
            #     template_raw_list = data['others'][trend]
            template_raw_list = data[render_type][trend]
    elif explanation:
        filename = os.path.join(TEMPLATE_DIR, 'explanation.json')
        with open(filename, 'r') as f:
            data = json.load(f)
            template_raw_list = data[trend]
    elif ending:
        filename = os.path.join(TEMPLATE_DIR, 'ending.json')
        with open(filename, 'r') as f:
            data = json.load(f)
            template_raw_list = data[render_type]
    else:
        filename = os.path.join(TEMPLATE_DIR, '{}'.format(trend if trend and 'cumulative' not in render_type else 'cumulative'))
        with open(filename, 'r') as f:
            template_raw_list = f.read().splitlines()
    return Template(random.choice(template_raw_list))


def random_transition(key):
    return random.choice(transitionWords[key])


def random_degree(key):
    return random.choice(degreeAdverbs[key])

# Creating the Correlation Matrix
def create_corr_df():
    my_index = ['newly confirmed cases', 'cumulative confirmed cases',
                'newly death cases', 'cumulative death cases', 'growth rate', 'fatality rate']
    ar = np.zeros((6, 6))
    corr_df = pd.DataFrame(ar, index=my_index, columns=my_index)

    # corr_df[A][B] is the relation when B is after A.
    corr_df.loc['newly confirmed cases']['growth rate'] = 0.8
    corr_df.loc['newly confirmed cases']['cumulative confirmed cases'] = 0.6
    corr_df.loc['newly confirmed cases']['newly death cases'] = 0.4
    corr_df.loc['cumulative confirmed cases']['growth rate'] = 0.6
    corr_df.loc['cumulative confirmed cases']['cumulative death cases'] = 0.4
    corr_df.loc['newly death cases']['fatality rate'] = 0.8
    corr_df.loc['newly death cases']['cumulative death cases'] = 0.6
    corr_df.loc['newly death cases']['newly confirmed cases'] = 0.4
    corr_df.loc['cumulative death cases']['fatality rate'] = 0.6
    corr_df.loc['cumulative death cases']['cumulative confirmed cases'] = 0.4
    corr_df.loc['growth rate']['fatality rate'] = 0.2
    corr_df.loc['fatality rate']['growth rate'] = 0.2

    return corr_df


def report_sequence(date, state=None, county=None, my_span=1):
    # if the span and date is valid
    flag = covid.valid_span(date, state=state, county=county, span=my_span)
    if flag == False:
        print("Error. Date or span is out of range./n")
        return -1

    my_index = ['newly confirmed cases', 'cumulative confirmed cases',
                'newly death cases', 'cumulative death cases', 'growth rate', 'fatality rate']

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
    # most_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale,
    #                                                                         span=my_span)
    # old_most_confirmed = covid.most_confirmed_cases_per_capita(pre_date, state=state, scale=per_capita_scale,
    #                                                                         span=my_span)
    # least_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale,
    #                                                                          span=my_span, type='least')
    # old_least_confirmed = covid.most_confirmed_cases_per_capita(pre_date, state=state, scale=per_capita_scale,
    #                                                                          span=my_span, type='least')
    # ratio = round(most_confirmed_cases_per_capita[0] / least_confirmed_cases_per_capita[0], 1)
    # old_ratio = round(old_most_confirmed[0] / old_least_confirmed[0], 1)
    # print('Data Collected')

    # Correlation Matrix

    corr_df = create_corr_df()

    # Rate calculation

    new_data = [new_confirmed_cases, cumulative_confirmed_cases, new_death_cases, cumulative_death_cases
                , growth_rate, death_rate]
                # most_confirmed_cases_per_capita, least_confirmed_cases_per_capita, ratio]
    old_data = [old_confirmed_cases, oldcumulative_confirmed_cases, old_death_cases, oldcumulative_death_cases
                , old_growth_rate, old_death_rate]
                # old_most_confirmed, old_least_confirmed, old_ratio]
    # Zero division Exception is not handled
    change_rate = list(map(lambda x: (x[0] - x[1]) / x[1], zip(new_data, old_data)))

    # Output a list of map with key Names, Current Value, Previous Value, Change Rate
    sequence = []
    temp_list = map(list, zip(my_index, new_data, old_data, change_rate))
    for ele in temp_list:
        temp_dict = dict(zip(['Name', 'Current Value', 'Previous Value', 'Change Rate'], ele))
        sequence.append(temp_dict)

    # Score of changing rate
    temp = sorted(change_rate[:6], key=abs, reverse=True)
    change_score = []
    interval = 1 / len(my_index)
    for ele in change_rate[:6]:
        change_score.append(round((temp.index(ele) + 1) * interval, 2))

    # Choose the first attribute
    res_order = []
    temp_index = my_index
    # first = my_index[change_score.index(max(change_score))]
    # res_order.append(first)
    # temp_index.remove(first)

    # Generate the rest of the sequence order
    while len(temp_index) > 1:
        couple1 = ""
        couple2 = ""
        max_score = 0.0
        for ele1 in temp_index:
            for ele2 in temp_index:
                if ele2 == ele1:
                    continue
                current_score = change_score[my_index.index(ele1)] \
                                + change_score[my_index.index(ele2)] \
                                + 0.8 * corr_df.loc[ele1][ele2]
                if current_score > max_score:
                    couple1 = ele1
                    couple2 = ele2
                    max_score = current_score
        res_order.append(couple1)
        res_order.append(couple2)
        temp_index.remove(couple1)
        temp_index.remove(couple2)
    if len(temp_index) > 0:
        res_order.append(temp_index[0])

    sequence = sorted(sequence, key=lambda x: res_order.index(x['Name']), reverse=True)
    return sequence


def sentence_generate(data, date, state, county=None, span=7, beginning=False):
    attribute, current, previous, rate = data['Name'], data['Current Value'], data['Previous Value'], data['Change Rate']
    trend = 'upward' if rate > 0.0 else 'downward'
    template = load_template(attribute, trend=trend, beginning=beginning)

    if isinstance(current, np.int64 or np.int32):
        current = locale.format_string('%d', current, grouping=True)
        previous = locale.format_string('%d', previous, grouping=True)
    elif isinstance(current, float):
        # current = '{}%'.format(int(round(current)))
        # previous = '{}%'.format(int(round(previous)))
        current = '{}%'.format(round(current, 1))
        previous = '{}%'.format(round(previous, 1))
    # rate = abs(int(round(rate * 100)))
    rate = abs(round(rate * 100, 1))

    spans = {1: 'day', 7: 'week', 30: 'month'}

    if rate >= 5.0:
        degree = '1'
    elif rate >= 1.0:
        degree = '-1'
    else:
        degree = '0'
    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    d = {
        'date': date.strftime('%A %d %B %Y'),
        'attribute': attribute,
        'current': current,
        'previous': previous,
        'location': county if county else state,
        'span': spans[span],
        'trend': trend,
        'rate': '{}%'.format(rate),
        'adverb': random_degree(degree)
    }
    return template.safe_substitute(d)


def couple_generate(couple, date, state, county=None, span=7):
    data1, data2 = couple
    attribute1, attribute2 = data1['Name'], data2['Name']
    rate1, rate2 = data1['Change Rate'], data2['Change Rate']

    sent1 = sentence_generate(data1, date, state, county=county, span=span)
    sent2 = sentence_generate(data2, date, state, county=county, span=span)

    if rate1 * rate2 > 0.0:
        trans_word = random_transition('similarity')
    else:
        trans_word = random_transition('opposition')

    if rate1 < 0.0 and rate2 < 0.0:
        explanation = load_template('', 'downward', explanation=True).template
    elif rate1 > 0.0 and rate2 > 0.0:
        explanation = load_template('', 'upward', explanation=True).template
    else:
        explanation = ''

    para = '{}{}, {}{}'.format(sent1, trans_word, sent2, explanation)
    return para


def ratio_generate(date, state, county=None, span=7):
    per_capita_scale = 100000
    most_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale,
                                                                            span=span)
    least_confirmed_cases_per_capita = covid.most_confirmed_cases_per_capita(date, state=state, scale=per_capita_scale,
                                                                             span=span, type='least')
    ratio = int(round(most_confirmed_cases_per_capita[0] / least_confirmed_cases_per_capita[0], 1))
    template_raw = 'The coronavirus has spread at a widely different rate and pace from ${type} to ${type} this ${span}. ' \
                   '${most_confirmed_cases_per_capita} has the highest reported rate of confirmed cases, at nearly ' \
                   '${most_confirmed_cases} per ${per_capita_scale} residents -- over ${ratio} times the ' \
                   'rate of the ${type} with the lowest rate.'
    template = Template(template_raw)
    spans = {1: 'day', 7: 'week', 30: 'month'}
    d = {
        'type': 'county' if state else 'state',
        'most_confirmed_cases_per_capita': most_confirmed_cases_per_capita[1],
        'most_confirmed_cases': int(most_confirmed_cases_per_capita[0]),
        'per_capita_scale': locale.format_string('%d', per_capita_scale, grouping=True),
        'ratio': ratio,
        'span': spans[span]
    }

    return template.safe_substitute(d)


def story_ending(date, state, county=None, span=7):
    if county:
        highest = covid.highest_cases(date, state=state)
    else:
        highest = covid.highest_cases(date)
    my_rank = covid.growth_rate_rank(date, span=span, state=state, county=county)
    template_raw = 'Comparing with others, the growth rate of ${location} this ${span} ranked ${rank} among all ${scale1}. ' \
                   '${highest} has the most confirmed cases in ${scale2}wide.'
    template = Template(template_raw)
    spans = {1: 'day', 7: 'week', 30: 'month'}
    d = {
        'location': county if county else state,
        'span': spans[span],
        'scale1': 'counties' if county else 'states',
        'scale2': 'state' if county else 'nation',
        'rank': my_rank,
        'highest': highest
    }
    return template.safe_substitute(d)


def get_last_day_week(date, delta=1):
    today = datetime.datetime.strptime(date, '%Y-%m-%d')
    yesterday = (today - datetime.timedelta(days=delta)).strftime('%Y-%m-%d')
    return yesterday


def num_format(number):
    return locale.format_string('%d', number, grouping=True)


def beginning_generate(date, state, county=None, attribute='cases'):
    scale_dict = {
        'scale': 'county' if county else 'state',
        'larger_scale': 'state' if county else 'nation'
    }
    yesterday = get_last_day_week(date, 1)

    if attribute == 'cases':
        newly_cases, total_cases = covid.confirmed_cases(date, state=state, county=county, span=1)
        _, previous_total_cases = covid.confirmed_cases(yesterday, state=state, county=county, span=1)
    else:
        newly_cases, total_cases = covid.death_cases(date, state=state, county=county, span=1)
        _, previous_total_cases = covid.death_cases(yesterday, state=state, county=county, span=1)
    increase_ratio = int(round((total_cases - previous_total_cases) * 1.0 / previous_total_cases, 0))

    rank_data = covid.rank_among_peers(date, attribute, state, county=county, scale='cumulative')
    rank = rank_data['rank']
    aboves, tops = rank_data['aboves'], rank_data['tops']

    above1, above1_total_cases = aboves[0]
    above2, above2_total_cases = aboves[1]
    above3, above3_total_cases = aboves[2]

    top, top_total_cases = tops[0]
    diff_between_top = top_total_cases - total_cases

    date = datetime.datetime.strptime(date, '%Y-%m-%d')
    d = {
        'date': date.strftime('%A %d %B %Y'),
        'location': county if county else state,
        'newly_cases': num_format(newly_cases),
        'total_cases': num_format(total_cases),
        'previous_total_cases': num_format(previous_total_cases),
        'increase_ratio': increase_ratio,
        'scale': scale_dict['scale'],
        'larger_scale': scale_dict['larger_scale'],
        'total_cases_rank': rank,
        'top': top,
        'diff_between_top': num_format(diff_between_top),
        'above1': above1,
        'above2': above2,
        'above3': above3,
        'above2_total_cases': num_format(above2_total_cases),
        'above3_total_cases': num_format(above3_total_cases)
    }

    if rank == 1:
        section = 'top1'
    elif rank <= 3:
        section = 'top3'
    else:
        section = 'others'

    beginning_raw = load_template(attribute, section, beginning=True)
    beginning = beginning_raw.safe_substitute(d)

    return beginning


def ending_generate(date, state, county=None, attribute='cases'):
    if county:
        state_newly_cases, state_total_cases = covid.confirmed_cases(date, state=state, span=1)
        state_newly_deaths, state_total_deaths = covid.death_cases(date, state=state, span=1)
        state_rank_cases = covid.rank_among_peers(date, 'cases', state)['rank']
        state_rank_deaths = covid.rank_among_peers(date, 'deaths', state)['rank']

        coin = random.getrandbits(1)
        if coin == 0:
            date = datetime.datetime.strptime(date, '%Y-%m-%d')
            d = {
                'date': date.strftime('%A %d %B %Y'),
                'state': state,
                'state_newly_cases': num_format(state_newly_cases),
                'state_total_cases': num_format(state_total_cases),
                'state_newly_deaths': num_format(state_newly_deaths),
                'state_total_deaths': num_format(state_total_deaths),
                'state_rank_cases': state_rank_cases,
                'state_rank_deaths': state_rank_deaths
            }

            ending_raw = load_template('county_to_state', ending=True)
            ending = ending_raw.safe_substitute(d)
        else:
            ending_raw = 'Finally, at the state level, {} currently ranks {} in the nation for its number of ' \
                         'COVID-19 cases and {} for deaths.'.format(state, state_rank_cases, state_rank_deaths)
            ending = ending_raw + beginning_generate(date, state, attribute=attribute)
    else:
        transition_raw = load_template('transition_sentence', ending=True)

        main_raw = load_template('main', ending=True)
        main_data = covid.ending_main(date, state, attribute=attribute)

        county1, county1_new = main_data['county1'], main_data['county1_new']
        county2, county2_new = main_data['county2'], main_data['county2_new']
        county3, county3_new = main_data['county3'], main_data['county3_new']
        total = main_data['total']

        county1_ratio = int(round(county1_new * 100.0 / total, 0))
        diff_12 = county1_new - county2_new
        county2_previous_rank = covid.rank_among_peers(get_last_day_week(date, 1), attribute, state, county=county2, scale='new')['rank']

        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        attribute_dict = {
            'cases': 'newly confirmed cases',
            'deaths': 'newly fatality cases'
        }

        d = {
            'date': date.strftime('%A %d %B %Y'),
            'type': attribute_dict[attribute],
            'state': state,
            'county1': county1,
            'county1_new': num_format(county1_new),
            'county1_ratio': county1_ratio,
            'county2': county2,
            'county2_new': num_format(county2_new),
            'diff_12': num_format(diff_12),
            'county2_previous_rank': county2_previous_rank,
            'county3': county3,
            'county3_new': num_format(county3_new)
        }

        transition_sent = transition_raw.safe_substitute(d)
        main = main_raw.safe_substitute(d)

        conclusion_raw = load_template('conclusion', ending=True)
        conclusion = conclusion_raw.safe_substitute(d)

        ending = transition_sent + main + conclusion
    return ending


def capitalize_article(article):
    def capitalize_para(para):
        def capitalize_sentence(sentence):
            if len(sentence) > 1:
                return sentence[0].capitalize() + sentence[1:]
            if len(sentence) == 1:
                return sentence.capitalize()
            else:
                return ''

        originals = para.split('.')
        trimmed = [sentence.strip() for sentence in originals]
        uppercased = [capitalize_sentence(sentence) for sentence in trimmed]
        return '. '.join(uppercased)

    originals = article.split('\n')
    trimmed = [para.strip() for para in originals]
    uppercased = [capitalize_para(para) for para in trimmed]
    return '\n'.join(uppercased)


def story_generate(date, state, county=None, span=7):
    sequence = report_sequence(date, state=state, county=None, my_span=span)

    # Beginning
    # beginning = sentence_generate(sequence[0], date, state, county, span, beginning=True)
    beginning = beginning_generate(date, state, county=county, attribute=random.choice(['cases', 'deaths']))

    # Second paragraph: 1,2 + 3,4 + 5,6
    second_para = ""
    couple1 = couple_generate((sequence[0], sequence[1]), date, state, county=county, span=span)
    couple2 = couple_generate((sequence[2], sequence[3]), date, state, county=county, span=span)
    couple3 = couple_generate((sequence[4], sequence[5]), date, state, county=county, span=span)
    # single = sentence_generate(sequence[5], date, state, county=county, span=span)
    second_para = couple1 + couple2 + couple3

    # ratio
    # third_para = ratio_generate(date, state=state, span=span)

    # ending
    # ending = story_ending(date, state, county, span)
    ending = ending_generate(date, state, county=county, attribute=random.choice(['cases', 'deaths']))

    story = "{}\n\n{}\n\n{}\n".format(beginning, second_para, ending)
    story = capitalize_article(story)
    print(story)

    with open('../{}-{}.txt'.format(date, state if state else 'us'), 'w') as f:
        f.write(story)


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
    # print(load_template('fatality rate', 'upward').template)
    story_generate('2020-05-23', 'California', span=7)
    # print(report_sequence('2020-05-11', 'Ohio', my_span=7))