import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')


class Covid19(object):
    def __init__(self):
        self.df_us = pd.read_csv(os.path.join(DATA_DIR, 'us.csv'))
        self.df_states = pd.read_csv(os.path.join(DATA_DIR, 'us-states.csv')).drop(axis=1, labels=['fips'])
        self.df_counties = pd.read_csv(os.path.join(DATA_DIR, 'us-counties.csv')).drop(axis=1, labels=['fips'])

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
        idx = pd.Index(df['date']).get_loc(date)

        return df.iloc[idx]['cases'] - df.iloc[idx - span if idx - span > 0 else 0]['cases']

    def growth_rate(self, date, state=None, county=None, span=1):
        """
        Growth rate for today or this week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param span: int, e.g. 1 for today, 7 for this week
        :return: float
        """
        raise NotImplementedError

    def confirmed_cases_comparison(self, date, state=None, county=None, scale='day'):
        """
        Compare confirmed cases between today/yesterday, this week/last week
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: str, e.g. 'Cook'
        :param scale: 'day' or 'week'
        :return: 'increase' or 'decrease'
        """
        raise NotImplementedError

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
    def confirmed_case_per_capita(self, date=None, state=None, county=None, scale='million'):
        """
        Confirmed case per capita
        :param date: str, e.g. '2020-04-23'
        :param state: str, e.g. 'Illinois'
        :param county: 'new' or 'cumulative'
        :param scale: 'million'
        :return: int
        """
        raise NotImplementedError

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

    def most_confirmed_cases_per_capita(self, state=None):
        """
        The state/county with the most confirmed cases per capita
        :param state: str, e.g. 'Illinois'
        :return: str
        """
        raise NotImplementedError
