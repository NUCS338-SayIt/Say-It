import os
import requests

from statsmodels.tsa.stattools import adfuller as ADF
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')
# DATA_DIR = r'C:\Users\86139\Desktop\西北大学\CS338\Project\Say-It\analytics\data'
BASE_URL = 'https://api.census.gov/data/2019/pep/population'
API_KEY = 'ff51fda390a45a5085af02b99725c2963edf7931'


class Covid19(object):
    def __init__(self):
        self.df_us = pd.read_csv(os.path.join(DATA_DIR, 'us.csv'), dtype={'fips': str})
        self.df_states = pd.read_csv(os.path.join(DATA_DIR, 'us-states.csv'), dtype={'fips': str})
        self.df_counties = pd.read_csv(os.path.join(DATA_DIR, 'us-counties.csv'), dtype={'fips': str})
        self.df_counties = self.df_counties[self.df_counties['county'] != 'Unknown']

    def __str__(self):
        return 'us: {}\nus-states: {}\nus-counties: {}'.format(
            self.df_us.shape, self.df_states.shape, self.df_counties.shape)

    """
    Test if the input is valid
    """
    def valid_span(self, date, state=None, county=None, span=1):
        df = self.df_us.copy()
        if state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        df = df.reset_index(drop=True)
        idx = pd.Index(df['date']).get_loc(date)

        # 'Error, please input the right date.\n'
        if idx - 1 <= 0:
            return False
        # 'Error. The span is out of the range of dataset.\n'
        if idx - 2 * span <= 0:
            return False
        # Valid
        return True

    """
    Confirmed cases related
    """
    def confirmed_cases(self, date, state=None, county=None, span=1):
        """
        Confirmed cases for today or this week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param span: int, e.g. 1 for today, 7 for this week
        :return: int
        """
        df = self.df_us.copy()
        if state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        df = df.reset_index(drop=True)
        try:
            idx = pd.Index(df['date']).get_loc(date)
        except Exception:
            return 0, 0

        span_cases = df.iloc[idx]['cases'] - df.iloc[idx - span]['cases'] if idx - span > 0 else df.iloc[idx]['cases']
        total_cases = df.iloc[idx]['cases']
        return span_cases, total_cases

    def growth_rate(self, date, state=None, county=None, span=1):
        """
        Growth rate for today or this week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param span: int, e.g. 1 for today, 7 for this week
        :return: float
        """
        df = self.df_us.copy()
        if state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        df = df.reset_index(drop=True)
        idx = pd.Index(df['date']).get_loc(date)

        return df.iloc[idx]['cases'] / df.iloc[idx - span if idx - span > 0 else 0]['cases'] - 1

    def confirmed_cases_comparison(self, date, state=None, county=None, scale='day'):
        """
        Compare confirmed cases between today/yesterday, this week/last week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param scale: 'day' or 'week'
        :return: 'increase' or 'decrease'
        """
        span = 1 if scale == 'day' else 7

        from datetime import datetime, timedelta
        dt = datetime.strptime(date, '%Y-%m-%d')
        prev_dt = dt - timedelta(days=span)
        prev_date = prev_dt.strftime('%Y-%m-%d')

        return 'increase' if self.confirmed_cases(date, state, county, span) > self.confirmed_cases(prev_date, state, county, span) else 'decrease'

    """
        Death cases related
        """

    def death_cases(self, date, state=None, county=None, span=1):
        """
        Death cases for today or this week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param span: int, e.g. 1 for today, 7 for this week
        :return: int
        """
        if span <= 0:
            print('Span should be positive.\n')

        df = self.df_us.copy()
        if state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        df = df.reset_index(drop=True)
        idx = pd.Index(df['date']).get_loc(date)

        span_deaths = df.iloc[idx]['deaths'] - df.iloc[idx - span if idx - span > 0 else 0]['deaths']
        total_deaths = df.iloc[idx]['deaths']

        return span_deaths, total_deaths

    def death_rate(self, date, state=None, county=None, span=1):
        """
        Death rate for today or this week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param span: int, e.g. 1 for today, 7 for this week
        :return: 4 digits float
        """
        if span <= 0:
            print('Span should be positive.Default value is 1.\n')

        df = self.df_us.copy()
        if state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        df = df.reset_index(drop=True)
        idx = pd.Index(df['date']).get_loc(date)

        span_deaths = df.iloc[idx]['deaths'] - df.iloc[idx - span if idx - span > 0 else 0]['deaths']
        total_deaths = df.iloc[idx]['deaths']
        span_cases = df.iloc[idx]['cases'] - df.iloc[idx - span if idx - span > 0 else 0]['cases']
        total_cases = df.iloc[idx]['cases']
        span_death_rate = float(format(span_deaths / span_cases, '.4f'))
        total_death_rate = float(format(total_deaths / total_cases, '.4f'))

        return span_death_rate, total_death_rate

    def death_rate_comparison(self, date, state=None, county=None, scale='day'):
        """
        Compare death rate between today/yesterday, this week/last week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param scale: str, 'day' or 'week'
        :return:str, 'increase' or 'decrease'
        """
        df = self.df_us.copy()
        if state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        df = df.reset_index(drop=True)
        idx = pd.Index(df['date']).get_loc(date)
        if idx - 1 <= 0:
            return 'Error, please input the right date.\n'

        if scale == 'day':
            if idx - 2 <= 0:
                return 'Error. Please input the right date under scale day.\n'
            today_death_rate = df.iloc[idx]['deaths'] - df.iloc[idx - 1]['deaths']
            yesterday_death_rate = df.iloc[idx - 1]['deaths'] - df.iloc[idx - 2]['deaths']
            return 'increase' if today_death_rate > yesterday_death_rate else 'decrease'
        elif scale == 'week':
            if idx - 14 <= 0:
                return 'Error. Please input the right date under scale week.\n'
            thisweek_death_rate = df.iloc[idx]['deaths'] - df.iloc[idx - 7]['deaths']
            lastweek_death_rate = df.iloc[idx - 7]['deaths'] - df.iloc[idx - 14]['deaths']
            return 'increase' if thisweek_death_rate > lastweek_death_rate else 'decrease'
        else:
            return 'Scale should be either day or week.\n'

    """
    Rank related
    """
    def max_increase_per_day(self, state=None, county=None):
        """
        Max increase per day in history
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :return: tuple(date, number)
        """
        df = self.df_us.copy()
        if state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]

        max_increase = df.iloc[0]['cases']
        max_idx = 0
        for idx in range(1, len(df.index)):
            increase = df.iloc[idx]['cases'] - df.iloc[idx - 1]['cases']
            if increase > max_increase:
                max_increase = increase
                max_idx = idx

        return df.iloc[max_idx]['date'], max_increase

    def highest_cases(self, date, state=None, scale='new'):
        """
        The state/county that has the highest cases among US/state, new or accumulative
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param scale: 'new' or 'cumulative'
        :return: str, e.g. 'California' or 'Cook'
        """
        df = self.df_states.copy()
        if state:
            df = self.df_counties.copy()
            df = df[df['state'] == state]
        df = df[df['date'] == date]
        df = df.reset_index(drop=True)

        idx = df['cases'].idxmax()
        if state:
            return df.iloc[idx]['county']
        else:
            return df.iloc[idx]['state']

    def growth_rate_rank(self, date, span=7, state=None, county=None):
        """
        Growth rate rank of a state/county among US/state in the past x days
        :param date: str, e.g. '2020-04-23'
        :param span: int, e.g. 7
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :return: int
        """
        df = self.df_states.copy()
        if state and county:
            df = self.df_counties.copy()
            df = df[df['state'] == state]
        df = df[df['date'] == date]
        growth_rate = self.growth_rate(date, state, county, span)
        rank = 1

        if state and county:
            for idx in range(len(df.index)):
                county_rate = self.growth_rate(date, state, df.iloc[idx]['county'], span)
                if county_rate > growth_rate: rank += 1
        else:
            for idx in range(len(df.index)):
                state_rate =  self.growth_rate(date, df.iloc[idx]['state'], county, span)
                if state_rate > growth_rate: rank += 1

        return rank

    def rank_among_peers(self, date, attribute, state, county=None, scale='cumulative'):
        """
        Get rank among peers. If rank <= 10, return at most three peer in front of it, otherwise return top 3.
        :param date: str, e.g. '2020-04-23'
        :param attribute: str, e.g. 'cases' or 'deaths'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param scale: str, 'new' or 'cumulative'
        :return: dict, e.g. {'rank': 1, 'aboves': ['Illinois', 'California'], 'tops': ['New York', 'Illinois', 'California']}
        """
        df = self.df_states.copy()
        if county:
            df = self.df_counties.copy()
            df = df[df['state'] == state]

        data_list, ranks = [], []
        if scale == 'cumulative':
            df = df[df['date'] == date]
            df.sort_values(by=[attribute], inplace=True, ascending=False)
            ranks = df['county'].to_list() if county else df['state'].to_list()
            for idx, row in df.iterrows():
                data_list.append((row['county'] if county else row['state'], row[attribute]))
        elif scale == 'new':
            peers = df['county'].unique() if county else df['state'].unique()

            today = datetime.datetime.strptime(date, '%Y-%m-%d')
            yesterday = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            today = date

            data_list = []
            for peer in peers:
                data_today = df[(df['date'] == today) & (df['county'] == peer)] if county else df[(df['date'] == today) & (df['state'] == peer)]
                data_yesterday = df[(df['date'] == yesterday) & (df['county'] == peer)] if county else df[(df['date'] == yesterday) & (df['state'] == peer)]

                newly = data_today.iloc[0][attribute] - data_yesterday.iloc[0][attribute] if not data_yesterday.empty else 0  # no newly case
                data_list.append((peer, newly))
            data_list = sorted(data_list, key=lambda x: x[1], reverse=True)
            ranks = [peer for peer, newly in data_list]

        rank = ranks.index(county if county else state)
        # aboves = ranks[rank - 2 if rank >= 2 else 0: rank] if rank < 10 else ranks[0: 2]
        aboves = data_list[max(rank - 3, 0): rank]
        tops = data_list[0: min(len(data_list), 3)]

        for i in range(len(aboves), 3):
            aboves.insert(0, ('', 0))
        for i in range(len(tops), 3):
            tops.insert(0, ('', 0))

        return {'rank': rank + 1, 'aboves': aboves, 'tops': tops}

    def trend_description(self, date, attribute, state, county=None, scale='cumulative'):
        """
        Describe the trend
        :param date: str, e.g. '2020-04-23'
        :param attribute: str, e.g. 'cases' or 'deaths'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param scale: str, 'new' or 'cumulative'
        :return: str, a sentence
        """

        df = self.df_states.copy()
        df = df[df['state'] == state]
        if county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        df = df.set_index(df['date']).truncate(after=date)

        input_data = np.array(df[attribute])
        if scale == 'new':
            df['new'] = df[attribute].shift(-1) - df[attribute]
            df['new'] = df['new'].shift(-1)
            input_data = np.array(df['new'][:-2].values)

        if input_data.size < 13:
            return ""
        lastest = input_data[-12:]
        std_lastest = (lastest - np.min(lastest)) / (np.max(lastest) - np.min(lastest))
        # print(std_lastest)
        today = std_lastest[-1]
        last_ten = std_lastest[:10]
        index = np.arange(10)

        reg = LinearRegression().fit(index.reshape(-1, 1), last_ten.reshape(-1, 1))
        intercept = last_ten - reg.coef_ * index
        max_value = reg.coef_ * 11 + np.max(intercept)
        min_value = reg.coef_ * 11 + np.min(intercept)

        if reg.coef_ > 0.05:
            sign1 = "in a rising channel"
        elif reg.coef_ < -0.05:
            sign1 = "in a descending channel"
        else:
            ave = np.mean(lastest[:10])
            sign1 = "fluctuating around %d cases" % ave

        sentence1 = "Based on the data from the past ten days, %s %s of this region is %s " % (scale, attribute, sign1)

        if min_value <= today <= max_value:
            sentence2 = "and there is no sign of breakthrough."
        elif today < min_value:
            sentence2 = "and today's data shows a sign of downward breakthrough."
        else:
            sentence2 = "and today's data shows a sign of upward breakthrough."

        return sentence1 + sentence2

        # Total trend
        # n = input_data.shape[0]
        # sum_sgn = 0
        # for i in np.arange(n):
        #     if i <= (n - 1):
        #         for j in np.arange(i + 1, n):
        #             if input_data[j] > input_data[i]:
        #                 sum_sgn = sum_sgn + 1
        #             elif input_data[j] < input_data[i]:
        #                 sum_sgn = sum_sgn - 1
        #             else:
        #                 sum_sgn = sum_sgn
        # if n <= 10:
        #     z_value = sum_sgn / (n * (n - 1) / 2)
        # else:
        #     if sum_sgn > 0:
        #         z_value = (sum_sgn - 1) / np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
        #     elif sum_sgn == 0:
        #         z_value = 0
        #     else:
        #         z_value = (sum_sgn + 1) / np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
        # ADF_result = ADF(input_data, 0)
        # # 99% ——> +—2.576
        # # 95% ——> +—1.96
        # # 90% ——> +—1.645
        # if ADF_result[1] < 0.01:
        #     result = 'remaining relatively stable'
        # else:
        #     if 1.96 < np.abs(z_value) <= 2.576:
        #         result = 'rising' if z_value > 0 else 'descending'
        #     elif np.abs(z_value) > 2.576:
        #         result = 'significantly rising' if z_value > 0 else 'significantly descending'
        #     else:
        #         result = 'fluctuating'
        # return result

    """
    Additional Data needed
    """
    def confirmed_case_per_capita(self, date=None, state=None, county=None, scale=100000, span=999):
        """
        Confirmed case per capita
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param scale: int
        :param span: int, e.g. 1 for today, 7 for this week
        :return: float
        """
        population = self.get_population(state=state, county=county)
        confirmed_cases = self.confirmed_cases(date=date, state=state, county=county, span=span)[0]

        return round(confirmed_cases / population * scale, 1)

    def most_confirmed_cases_per_capita(self, date, state=None, scale=100000, span=999, type='most'):
        """
        The state/county with the most confirmed cases per capita
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param scale: int
        :param span: int, e.g. 1 for today, 7 for this week
        :param type: str, 'most' or 'least'
        :return: str
        """
        df = self.df_states.copy()
        if state:
            df = self.df_counties.copy()
            df = df[df['state'] == state]
        df = df.reset_index(drop=True)

        if state:
            candidates = df['county'].unique()
            cases_per_capita = {county: self.confirmed_case_per_capita(date, state=state, county=county, scale=scale, span=span) for county in candidates}
        else:
            candidates = df['state'].unique()
            cases_per_capita = {state: self.confirmed_case_per_capita(date, state=state, scale=scale, span=span) for state in candidates}

        if type == 'most':
            result = max(zip(cases_per_capita.values(), cases_per_capita.keys()))
        else:
            non_zeros = {k: v for k, v in cases_per_capita.items() if v > 0.0}
            result = min(zip(non_zeros.values(), non_zeros.keys()))
        return result

    def get_population(self, state=None, county=None):
        """
        Get population of a specific state/county
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :return: int
        """
        if state and county:
            df = self.df_counties.copy()
            df = df[(df['state'] == state) & (df['county'] == county)]
        elif state:
            df = self.df_states.copy()
            df = df[df['state'] == state]
        else:
            df = self.df_us.copy()
        df = df.reset_index(drop=True)
        fips = df['fips'][0] if state else None

        params = {
            'get': 'POP',
            'for': 'county:{}'.format(fips[2:]) if county else 'state:{}'.format(fips) if state else 'us:*',
            'in': 'state:{}'.format(fips[0:2]) if county else None,
            'key': API_KEY
        }
        response = requests.get(url=BASE_URL, params=params)
        if response.status_code != 200:
            import sys
            return sys.maxsize * 2 + 1
        data = response.json()
        return int(data[1][0])

    def ending_main(self, date, state, attribute='cases'):
        """
        Return some data used in the ending paragraph
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param attribute: str, e.g. 'cases' or 'deaths'
        :return:
        """
        df = self.df_counties.copy()
        df = df[df['state'] == state]
        df = df.reset_index(drop=True)

        peers = df['county'].unique()

        today = datetime.datetime.strptime(date, '%Y-%m-%d')
        yesterday = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        today = date

        data_list = []
        total_newly = 0
        for peer in peers:
            data_today = df[(df['date'] == today) & (df['county'] == peer)]
            data_yesterday = df[(df['date'] == yesterday) & (df['county'] == peer)]

            newly = data_today.iloc[0][attribute] - data_yesterday.iloc[0][attribute] if not data_yesterday.empty else 0
            if newly > 0:
                total_newly += newly
                data_list.append((peer, newly))
        data_list = sorted(data_list, key=lambda x: x[1], reverse=True)

        if len(data_list) < 3:
            raise IndexError
        county1, county1_new = data_list[0]
        county2, county2_new = data_list[1]
        county3, county3_new = data_list[-1]

        return {'county1': county1, 'county1_new': county1_new,
                'county2': county2, 'county2_new': county2_new,
                'county3': county3, 'county3_new': county3_new,
                'total': total_newly}
