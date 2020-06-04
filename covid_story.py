import argparse
from nlg.snapshot import story_generate


def interface():
    parser = argparse.ArgumentParser(description="""

        This script generates a snapshot story about COVID-19.

    """)
    parser.add_argument('-d', '--date', help='date in format "YYYY-MM-dd", e.g. 2020-05-28', required=True)
    parser.add_argument('-s', '--state', help='state name, e.g. Illinois', required=True)
    parser.add_argument('-c', '--county', help='county name, e.g. Cook')
    parser.add_argument('--save', nargs='?', const=1, help='save story in current directory')
    args = parser.parse_args()

    date, state, county = args.date, args.state, args.county
    save = True if args.save == 1 else False

    story_generate(date, state, county=county, save=save)


if __name__ == '__main__':
    interface()
