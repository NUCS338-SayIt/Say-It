import os
import requests

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')
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
        raise NotImplementedError

    def highest_cases(self, date, state=None, scale='new'):
        """
        The state/county that has the highest cases among US/state, new or accumulative
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param scale: 'new' or 'cumulative'
        :return: str, e.g. 'California' or 'Cook'
        """
        raise NotImplementedError

    def growth_rate_rank(self, date, span=7, state=None, county=None):
        """
        Growth rate rank of a state/county among US/state in the past x days
        :param date: str, e.g. '2020-04-23'
        :param span: int, e.g. 7
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :return: int
        """
        raise NotImplementedError

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

    def cure_rate(self, date, state=None, county=None, span=1):
        """
        Cure rate for today or this week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param span: int, e.g. 1 for today, 7 for this week
        :return: float
        """
        raise NotImplementedError

    def cure_rate_comparison(self, date, state=None, county=None, scale='day'):
        """
        Compare cure rate between today/yesterday, this week/last week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param scale: 'day' or 'week'
        :return: 'increase' or 'decrease'
        """
        raise NotImplementedError

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
            non_zeros = {k: v for k, v in cases_per_capita.items() if v != 0.0}
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
