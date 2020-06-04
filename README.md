# Say-It

*Say-It* is a system that generates a snapshot story about COVID-19 for a specific place and date.

## Usage

*Say-It* should be run in the command line with Python3 environment.

### Named Arguments

|Arguments|Explanation|
-|-
-d, --date|date<br>e.g. 2020-06-01
-s, --state|state name<br>e.g. Illinois
-c, --county|county name(optional)<br>e.g. Cook
--save|save to current directory(optional)


### Example

Illinois state on 2020-06-01 without saving:

```shell script
python3 covid_story.py -d 2020-06-01 -s Illinois
```

Cook county Illinois state on 2020-06-01 with saving:

```shell script
python3 covid_story.py -d 2020-06-01 -s Illinois -c Cook --save
```

## Structure

```
- Say-It
  ├─ analytics
  |  ├─ __init__.py
  |  └─ covid19.py            # covid data analysis
  ├─ data                     # covid data
  ├─ nlg
  |  ├─ __init__.py
  |  └─ snapshot.py           # story generation
  ├─ template
  |  ├─ beginning.json        # beginning paragraph template
  |  ├─ ending.json           # ending paragraph template
  |  ├─ explanation.json      # beginning paragraph template
  |  ├─ upward                # upward trend template
  |  └─ downward              # downward trend template
  ├─ covid_story.py           # command line interface
  ├─ requirements.txt         # python package requirements
  └─ update.sh                # shell script to update dataset
```
