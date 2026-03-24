"""Using mesonet and agrimet modules to investigate properties of station networks."""

import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

from chmdata.agrimet import Agrimet, load_stations
from chmdata.mesonet import Mesonet, stns_metadata

# # Finding average length of period of record for Montana Agrimet stations

all_stns = load_stations()

thing = Agrimet(lat=46.5889579, lon=-112.0152353)  # WRD building in Helena, MT
print()
print(thing.station)

installs = pd.DataFrame()
i = 0
for key in all_stns.keys():
    install = all_stns[key]['properties']['install']
    if (len(install) > 0) and (all_stns[key]['properties']['state'] == 'MT'):
        installs.at[i, 'ID'] = key
        installs.at[i, 'Install'] = dt.datetime.strptime(install, '%m/%d/%Y')
        installs.at[i, 'POR'] = dt.date.today() - installs.at[i, 'Install'].date()
        i += 1
print(i)
print(installs)
print(installs['POR'].mean())
print(7963 / 365)  # 22 years on 8/19/24

years = pd.date_range('1984-01-01', '2025-01-01', freq='YS')

plt.figure(figsize=(12, 5))
plt.suptitle('Agrimet Station Period of Record Summary')

plt.subplot(121)
plt.xlabel('year of install')
plt.ylabel('number of stations')
plt.grid(axis='y', zorder=1)
plt.hist(installs['Install'], bins=years, rwidth=0.9, align='left', zorder=3)

plt.subplot(122)
plt.xlabel('year')
plt.ylabel('fraction of stations with data')
plt.grid(axis='y', zorder=1)
plt.hist(installs['Install'], bins=years, rwidth=0.9, align='left', zorder=3, cumulative=True, density=True)

plt.tight_layout()
plt.show()

# Both Agrimet and Mesonet

# prepping mesonet
metadata = stns_metadata(False)  # No install date for inactive stations... :(
installs_mn = pd.DataFrame(index=metadata.keys(), columns=['Install Date'])
for key, val in metadata.items():
    installs_mn.loc[key] = val['date_installed']
installs_mn['Install Date'] = pd.to_datetime(installs_mn['Install Date'])

# plotting
plt.figure(figsize=(12, 5))
plt.suptitle('Weather Station Period of Record Summary')

plt.subplot(121)
plt.xlabel('year of install')
plt.ylabel('number of stations')
plt.grid(axis='y', zorder=1)
plt.hist([installs['Install'], installs_mn['Install Date']], bins=years,
         rwidth=0.9, align='left', zorder=3, stacked=True)
plt.legend(['Agrimet', 'Mesonet'])

plt.subplot(122)
plt.xlabel('year')
plt.ylabel('fraction of stations with data')
plt.grid(axis='y', zorder=1)
plt.hist([installs['Install'], installs_mn['Install Date']], bins=years, rwidth=0.9, align='left', zorder=3,
         cumulative=True, stacked=True)
plt.tight_layout()
plt.show()

# # From Mesonet.py
thing = Mesonet(lat=46.5889579, lon=-112.0152353)  # WRD building in Helena, MT
print()
print(thing.station)

one = Mesonet(stn_id='acecrowa')
one.get_data(elems=['air_temp', 'wind_dir'], der_elems=['etr', 'feels_like'],
             start='2023-05-01', end='2023-05-06', time_step='hourly')
print(one.data)
print(one.data.columns)
print()

metadata = stns_metadata(False)  # No install date for inactive stations... :(
print(metadata['aceabsar'])
installs = pd.DataFrame(index=metadata.keys(), columns=['Install Date'])
for key, val in metadata.items():
    installs.loc[key] = val['date_installed']
installs['Install Date'] = pd.to_datetime(installs['Install Date'])
# print(installs)
print(len(installs[installs['Install Date'].dt.strftime('%Y') < '2021']))
# No, none of the ones installed in 2021 are early enough in the year to count for a full growing season.

years = pd.date_range('2016-01-01', '2025-01-01', freq='YS')

plt.figure(figsize=(12, 5))
plt.suptitle('Mesonet Station Period of Record Summary')

plt.subplot(121)
plt.xlabel('year of install')
plt.ylabel('number of stations')
plt.grid(axis='y', zorder=1)
plt.hist(installs['Install Date'], bins=years, rwidth=0.9, align='left', zorder=3)

plt.subplot(122)
plt.xlabel('year')
plt.ylabel('fraction of stations with data')
plt.grid(axis='y', zorder=1)
plt.hist(installs['Install Date'], bins=years, rwidth=0.9, align='left', zorder=3, cumulative=True, density=True)

plt.tight_layout()
plt.show()
