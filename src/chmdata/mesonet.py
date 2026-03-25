"""Functions to retrieve Mesonet-related data from the Montana Climate Office's API.

For more information, see https://climate.umt.edu/mesonet/ or https://mesonet.climate.umt.edu/api/v2/docs#/

This module currently only deals with active stations. Inactive stations do not have a valid install date,
and I cannot get the data downloads to work.

I have attempted to make this module as generically useful as possible. However, there are many
opportunities to fine-tune the output from these queries, and many subtle variations in the data available from
different stations. If something more specific is required, it's likely possible! Check out the documentation above.

For example, data at a finer resolution than hourly can be asked for, and different aggregation functions
(other than the default average) can be specified for observations.

Default output is in US customary units. SI mostly works, but GDD needs other new inputs in C to work properly.

Hannah Haugen, hannah.haugen@mt.gov, 08/14/2024
"""

from datetime import date
import json

import pandas as pd
import requests

from chmdata.met_utils import great_circle_distance

# Available variables from the observations endpoints.
# Generic values like 'soil temp' will fetch soil temp at all available levels.
# Valid generic values are towards the end of this list.
# Are air temp max and min just not available? Then why are they listed here?
OBSERVATIONS = [
    "air_temp_0200",
    "air_temp_0244",
    "air_temp_logger_0244",
    "air_temp_max_0200",
    "air_temp_min_0200",
    "batt_per",
    "batt_vol",
    "bp",
    "bp_0150",
    "bp_logger_0244",
    "door",
    "irra_ni",
    "irra_red",
    "lightning_count",
    "lightning_dist",
    "pluv_fill",
    "ppt",
    "ppt_max_rate",
    "radi_ni",
    "radi_red",
    "rh",
    "rhs_temp",
    "snow_depth",
    "snow_depth_q",
    "soil_ec_blk_0005",
    "soil_ec_blk_0010",
    "soil_ec_blk_0020",
    "soil_ec_blk_0050",
    "soil_ec_blk_0070",
    "soil_ec_blk_0076",
    "soil_ec_blk_0091",
    "soil_ec_blk_0100",
    "soil_ec_perm_0005",
    "soil_ec_perm_0010",
    "soil_ec_perm_0020",
    "soil_ec_perm_0050",
    "soil_ec_perm_0100",
    "soil_ec_por_0005",
    "soil_ec_por_0010",
    "soil_ec_por_0020",
    "soil_ec_por_0050",
    "soil_ec_por_0100",
    "soil_temp_0005",
    "soil_temp_0010",
    "soil_temp_0020",
    "soil_temp_0050",
    "soil_temp_0070",
    "soil_temp_0076",
    "soil_temp_0091",
    "soil_temp_0100",
    "soil_vwc_0005",
    "soil_vwc_0010",
    "soil_vwc_0020",
    "soil_vwc_0050",
    "soil_vwc_0070",
    "soil_vwc_0076",
    "soil_vwc_0091",
    "soil_vwc_0100",
    "sol_rad",
    "swe",
    "vpd_atmo",
    "well_eco",
    "well_lvl",
    "well_tmp",
    "wind_dir_0244",
    "wind_dir_1000",
    "wind_dir_sd_1000",
    "windgust_0244",
    "windgust_1000",
    "wind_spd_0244",
    "wind_spd_1000",
    "x_lvl_at",
    "y_lvl_at",
    "air_temp",
    "air_temp_logger",
    "air_temp_max",
    "air_temp_min",
    "bp_logger",
    "soil_ec_blk",
    "soil_ec_perm",
    "soil_ec_por",
    "soil_temp",
    "soil_vwc",
    "wind_dir",
    "wind_dir_sd",
    "windgust",
    "wind_spd",
    "soil",
    "wind",
]

# print(len(OBSERVATIONS))  # 89

# Available variables from the derived endpoints
# Note: 'gdd' (growing degree days) only available at daily time steps, not hourly.
DERIVED = ["wind_chill", "heat_index", "feels_like", "wet_bulb", "etr", "gdd", "swp", "cci", "frost_depth"]

OBS_LONG_NAMES = [
    "Air Temperature @ 2 m",
    "Air Temperature @ 8 ft",
    "Logger Temperature",
    "Maximum Air Temperature @ 2 m",
    "Minimum Air Temperature @ 2 m",
    "Battery Percent",
    "Battery Voltage",
    "Atmospheric Pressure",
    "Atmospheric Pressure @ 1.5 m",
    "Logger Reference Pressure",
    "Enclosure Door Status",
    "810 nm Irradiance",
    "650 nm Irradiance",
    "Lightning Activity",
    "Lightning Distance",
    "Pluviometer Fill Level",
    "Precipitation",
    "Max Precip Rate",
    "810 nm Radiance",
    "650 nm Radiance",
    "Relative Humidity",
    "RH Sensor Temp",
    "Snow Depth",
    "Snow Depth Quality",
    "Bulk EC @ -5 cm",
    "Bulk EC @ -10 cm",
    "Bulk EC @ -20 cm",
    "Bulk EC @ -50 cm",
    "Bulk EC @ 70 cm",
    "Bulk EC @ 76 cm",
    "Bulk EC @ -91 cm",
    "Bulk EC @ -100 cm",
    "Rel. Permittivity @ -5 cm",
    "Rel. Permittivity @ -10 cm",
    "Rel. Permittivity @ -20 cm",
    "Rel. Permittivity @ -50 cm",
    "Rel. Permittivity @ -100 cm",
    "Pore EC @ -5 cm",
    "Pore EC @ -10 cm",
    "Pore EC @ -20 cm",
    "Pore EC @ -50 cm",
    "Pore EC @ -100 cm",
    "Soil Temperature @ -5 cm",
    "Soil Temperature @ -10 cm",
    "Soil Temperature @ -20 cm",
    "Soil Temperature @ -50 cm",
    "Soil Temperature @ -70 cm",
    "Soil Temperature @ -76 cm",
    "Soil Temperature @ -91 cm",
    "Soil Temperature @ -100 cm",
    "Soil VWC @ -5 cm",
    "Soil VWC @ -10 cm",
    "Soil VWC @ -20 cm",
    "Soil VWC @ -50 cm",
    "Soil VWC @ -70 cm",
    "Soil VWC @ -76 cm",
    "Soil VWC @ -91 cm",
    "Soil VWC @ -100 cm",
    "Solar Radiation",
    "SWE",
    "VPD",
    "Well EC",
    "Well Water Level",
    "Well Water Temperature",
    "Wind Direction @ 8 ft",
    "Wind Direction @ 10 m",
    "Wind Direction SD @ 10 m",
    "Gust Speed @ 8 ft",
    "Gust Speed @ 10 m",
    "Wind Speed @ 8 ft",
    "Wind Speed @ 10 m",
    "X-axis Level",
    "Y-axis Level",
]
DER_LONG_NAMES = [
    "Reference ET (a=0.23)",
    "GDDs",
    "Wind Chill",
    "Heat Index",
    "Wet Bulb Temperature",
    "Comprehensive Climate Index",
    "Frost Depth",
    "Feels Like Temperature",
    "Cumulative GDDs",
]

# Dictionary for referencing column names/variable abreviations
ALL_ELEMS = OBSERVATIONS + DERIVED
ALL_LONG_NAMES = OBS_LONG_NAMES + DER_LONG_NAMES
DICT_ALL2 = dict(zip(ALL_LONG_NAMES, ALL_ELEMS))
# print(DICT_ALL2)


class Mesonet:
    """Mesonet data from the Montana Climate Office."""

    def __init__(self, stn_id=None, stn_name=None, lat=None, lon=None):
        """Initializes instance with station id, station name, or a set of lat and lon.

        Priority is in order listed. Establishes which station an instance is, as well as basic info.
        For data downlaod, see get_data() method below.
        """

        # Each instance will store the information for all stations.
        # Definition is outside class so users can look at that info separately too.
        self.all_metadata = stns_metadata()

        self.lat = lat
        self.lon = lon

        # If station id is given, and it's valid, that's the station
        if stn_id is not None and stn_id in self.all_metadata:
            self.station = stn_id
            self.lat = self.all_metadata[self.station]["latitude"]
            self.lon = self.all_metadata[self.station]["longitude"]
            self.dist_from_stn = None
        # Else, if station name is given, and it's valid, that's the station
        elif stn_name is not None and self.find_stn_abr(stn_name) is not None:
            self.station = self.find_stn_abr(stn_name)
            self.lat = self.all_metadata[self.station]["latitude"]
            self.lon = self.all_metadata[self.station]["longitude"]
            self.dist_from_stn = None
        # Else, if lat and lon given, and coordinates are in/near Montana, use the closest station
        elif lat is not None and lon is not None and self.closest_station() < 300:  # km
            self.dist_from_stn = self.closest_station()
        else:
            # If all of the above fail, tell the user why.
            if stn_id is not None:
                raise ValueError("This station ID does not correspond to any active Mesonet stations.")
            if stn_name is not None:
                raise ValueError("This station name does not correspond to any active Mesonet stations.")
            if lat is not None and lon is not None:
                raise ValueError("This location is not in Montana.")
            if stn_id is None and stn_name is None and (lat is None or lon is None):
                raise ValueError("Not enough information to identify the appropriate Mesonet station.")

        self.name = self.all_metadata[self.station]["name"]
        self.station_page = f"https://mesonet.climate.umt.edu/dash/{stn_id}"
        # self.var_names = self.station_vars()  # Is this necessary?
        self.data = pd.DataFrame()

    def station_vars(self) -> list[str]:
        """Retrieves avilable Mesonet variables for this station.

        Availability by station is determined by what network the station is a part of: hydromet or agrimet.

        Returns:
            var_names: A list of available station variables.
        """
        url = f"https://mesonet.climate.umt.edu/api/v2/elements/{self.station}/?type=json"
        r = requests.get(url, timeout=20)
        vars_info = json.loads(r.text)
        var_names = [i["element"] for i in vars_info]
        return var_names

    def closest_station(self) -> float:
        """Determines which Mesonet station is the closest given a set of coordinates.

        Sets self.station, returns great circle distance to that station in km.

        Returns:
            distances[k]: distance to closest station in km
        """
        distances = {}
        station_coords = {}
        for k, feat in self.all_metadata.items():
            stn_site_id = k
            lat_stn = feat["latitude"]
            lon_stn = feat["longitude"]
            dist = great_circle_distance((self.lat, self.lon), (lat_stn, lon_stn))
            distances[stn_site_id] = dist
            station_coords[stn_site_id] = lat_stn, lon_stn
        k = min(distances, key=distances.get)
        # self.dist_from_stn = distances[k]
        # return k
        self.station = k
        return distances[k]

    @staticmethod
    def find_stn_abr(target_name: str) -> str | None:
        """Takes a known station name, returns the abreviated key used to identify it."""
        info = stns_metadata()  # should this be self.all_metadata?
        for k, v in info.items():
            # Remove case sensitivity
            if v["name"].lower() == target_name.lower():
                return k
        # print("Sorry, that name does not corespond to any active Mesonet stations.")
        return None

    def get_data(self, elems="", der_elems=None, start="", end="", time_step="daily", units="us", public=True):
        """Downloads Mesonet data for a single station from Montana Climate Office and saves to self.data.

        This version just overwrites the previous data, no advanced duplicate/inclusion checking.
        Start date is inclusive, end date is exclusive.
        Hourly data is downloaded by the day, so for each day in the date range, 24 observations will be reported.

        Args:
            elems: list of str, desired variables from OBSERVATIONS to fetch. By default, all variables will be
              downloaded. Pass 'None' to download no observation data.
            der_elems: list of str, desired variables from DERIVED to fetch. By default, no variables will be
              downloaded. Pass '' (empty str) to download all derived data.
            start: str, optional; YYYY-MM-DD format (inclusive)
            end: str, optional; YYYY-MM-DD format (exclusive)
            time_step: str, optional; either 'daily' or 'hourly' to determine time period over which to aggregate data
            units: str, optional; either 'us' or 'si' to determine the output units.
              NOTE: issue with gdd var and si units.
            public: bool, optional; whether to include more obscure/maintenance associated variables like battery
            and sensor temp. If True, a smaller subset of variables will be downloaded.
        """

        # Check for correct inputs
        if time_step not in ["daily", "hourly"]:
            raise ValueError('Invalid time step, please choose either "daily" or "hourly".')
        if units not in ["us", "si"]:
            raise ValueError('Invalid unit system, please choose either "us" or "si".')
        if (elems is None) and (der_elems is None):
            raise ValueError(
                'Variables to download set to "None". Please select variables '
                "or use empty strs to download all data."
            )

        # converting bool to string for url
        if public:
            public = "true"
        else:
            public = "false"

        stn_metadata = self.all_metadata[self.station]

        # If start not provided, find install date of station
        if start == "":
            start = stn_metadata["date_installed"]
            if type(start) is int:
                start = date.fromtimestamp(start / 1000)  # they switched it to milliseconds.
                start = start.strftime("%Y-%m-%d")
            print(f"{self.station} install date: {start}")
        # If end not provided, choose one (which logic to use?)
        if end == "":
            # End of previous calendar year
            end = f"{date.today().year}-01-01"
            # # Today
            # end = date.today().strftime('%Y-%m-%d')

        # Downloading observations, if requested
        url_elems = ""
        data = pd.DataFrame()
        if elems is not None:
            # list of variables prepared for url
            for v in elems:
                url_elems += f"&elements={v}"
            url = (
                "https://mesonet.climate.umt.edu/api/v2/observations/{}/?na_info=false&premade=false&"
                "latest=true&type=csv&rm_na=false&active=true&public={}&wide=true&units={}&tz=America%2FDenver&"
                "simple_datetime=false&end_time={}T00%3A00%3A00&start_time={}T00%3A00%3A00&level=1{}&"
                "stations={}".format(time_step, public, units, end, start, url_elems, self.station)
            )
            data = pd.read_csv(url, index_col="datetime")
            if time_step == "daily":
                data.index = [j[:10] for j in data.index]  # remove time component, leaving only date
            data.index = pd.to_datetime(data.index)

        # Downloading derived metrics, if requested
        url_der_elems = ""
        datad = pd.DataFrame()
        if der_elems is not None:
            # list of variables prepared for url
            for v in der_elems:
                url_der_elems += f"&elements={v}"
            url = (
                "https://mesonet.climate.umt.edu/api/v2/derived/{}/?crop=corn&high=86&low=50&alpha=0.23&"
                "na_info=false&rm_na=false&premade=true&wide=true&keep=false&units={}&type=csv&tz=America%2FDenver&"
                "simple_datetime=false&time=daily&end_time={}T00%3A00%3A00&start_time={}T00%3A00%3A00&"
                "level=1&stations={}{}".format(time_step, units, end, start, self.station, url_der_elems)
            )
            datad = pd.read_csv(url, index_col="datetime")
            if time_step == "daily":
                datad.index = [j[:10] for j in datad.index]  # remove time component, leaving only date
            datad.index = pd.to_datetime(datad.index)

        # this line is not working? Need indices to line up.
        all_data = pd.concat([data, datad], axis=1)
        all_data = all_data.loc[:, ~all_data.columns.duplicated()].copy()

        self.data = all_data
        return self.data

    def get_data1(self, elems="", der_elems=None, start="", end="", time_step="daily", units="us", public=True):
        """Downloads Mesonet data for a single station from Montana Climate Office and saves to self.data.

        This version is trying to be dynamically updating, and that is proving really tricky.
        Start date is inclusive, end date is exclusive.
        Hourly data is downloaded by the day, so for each day in the date range, 24 observations will be reported.

        Args:
            elems: list of str, desired variables from OBSERVATIONS to fetch. By default, all variables will be
              downloaded. Pass 'None' to download no observation data.
            der_elems: list of str, desired variables from DERIVED to fetch. By default, no variables will be
              downloaded. Pass '' (empty str) to download all derived data.
            start: str, optional; YYYY-MM-DD format (inclusive)
            end: str, optional; YYYY-MM-DD format (exclusive)
            time_step: str, optional; either 'daily' or 'hourly' to determine time period over which to aggregate data.
            units: str, optional; either 'us' or 'si' to determine the output units.
              NOTE: issue with gdd var and si units.
            public: bool, optional; whether to include more obscure/maintenance associated variables like battery
              and sensor temp.
        """

        # Check for correct inputs
        if time_step not in ["daily", "hourly"]:
            raise ValueError('Invalid time step, please choose either "daily" or "hourly".')
        if units not in ["us", "si"]:
            raise ValueError('Invalid unit system, please choose either "us" or "si".')
        if (elems is None) and (der_elems is None):
            raise ValueError(
                'Variables to download set to "None". Please select variables '
                "or use empty strs to download all data."
            )

        # converting bool to string for url
        if public:
            public = "true"
        else:
            public = "false"

        # Check that variables are new - this kind of works. The generic names are a problem.
        # convert column names to valid variable names
        existing = []
        for i in self.data.columns[1:]:
            existing.append(i.split("[")[0][:-1])  # remove the bracketed units at the end
        # existing = []
        existing = {DICT_ALL2[i] for i in existing}  # turn back into abbreviated variables
        if existing:
            existing.add("station")
            # then just do the difference of existing and elems and der_elems
            print("existing columns:", existing)
            print("Elems updating:")
            print("old", elems)
            set(elems).difference_update(existing)
            # set(der_elems).difference_update(existing)
            # also remove generic words for specific variables? But what if you only downloaded some of the soil data?
            print("new", elems)

        stn_metadata = self.all_metadata[self.station]

        # If start not provided, find install date of station
        if start == "":
            start = stn_metadata["date_installed"]
            if type(start) is int:
                start = date.fromtimestamp(start / 1000)  # they switched it to milliseconds.
                start = start.strftime("%Y-%m-%d")
            print(f"{self.station} install date: {start}")
        # If end not provided, choose one (which logic to use?)
        if end == "":
            # End of previous calendar year
            end = f"{date.today().year}-01-01"
            # # Today
            # end = date.today().strftime('%Y-%m-%d')

        # Downloading observations, if requested
        url_elems = ""
        data = pd.DataFrame()
        if elems is not None:
            # list of variables prepared for url
            for v in elems:
                url_elems += f"&elements={v}"
            url = (
                "https://mesonet.climate.umt.edu/api/v2/observations/{}/?na_info=false&premade=false&"
                "latest=true&type=csv&rm_na=false&active=true&public={}&wide=true&units={}&tz=America%2FDenver&"
                "simple_datetime=false&end_time={}T00%3A00%3A00&start_time={}T00%3A00%3A00&level=1{}&"
                "stations={}".format(time_step, public, units, end, start, url_elems, self.station)
            )
            data = pd.read_csv(url, index_col="datetime")
            if time_step == "daily":
                data.index = [j[:10] for j in data.index]  # remove time component, leaving only date
            data.index = pd.to_datetime(data.index)

        # Downloading derived metrics, if requested
        url_der_elems = ""
        datad = pd.DataFrame()
        if der_elems is not None:
            # list of variables prepared for url
            for v in der_elems:
                url_der_elems += f"&elements={v}"
            url = (
                "https://mesonet.climate.umt.edu/api/v2/derived/{}/?crop=corn&high=86&low=50&alpha=0.23&"
                "na_info=false&rm_na=false&premade=true&wide=true&keep=false&units={}&type=csv&tz=America%2FDenver&"
                "simple_datetime=false&time=daily&end_time={}T00%3A00%3A00&start_time={}T00%3A00%3A00&"
                "level=1&stations={}{}".format(time_step, units, end, start, self.station, url_der_elems)
            )
            datad = pd.read_csv(url, index_col="datetime")
            if time_step == "daily":
                datad.index = [j[:10] for j in datad.index]  # remove time component, leaving only date
            datad.index = pd.to_datetime(datad.index)

        # this line is not working? Need indices to line up.
        all_data = pd.concat([data, datad], axis=1)

        self.data = pd.concat([self.data, all_data], axis=1)
        return self.data

    def asce_ref_et(self):
        """Calculate ASCE alfalfa reference ET, including downloading required variables if missing."""
        # TODO: write this function
        return self.data

    def save_data(self, save_loc: str) -> None:
        """save_loc: str, filepath to save csv to."""
        self.data.to_csv(save_loc)


def stns_metadata(active: bool = True) -> dict:
    """Retrieves metadata for Mesonet stations.

    Args:
        active: bool, optional; if True, retrieve only currently active stations,
          if False, retrieve info for all stations.

    Returns:
        stns_dict: A dictionary of station metadata.
    """

    if active:
        active = "true"  # currently 155
    else:
        active = "false"  # currently 220

    # Error in retrieving elements from inactive stations, so just deal with active ones for now.
    # How to handle this, especially when it is noted by changing the install date to 'None'?

    url = f"https://mesonet.climate.umt.edu/api/v2/stations/?public=true&active={active}&type=json"
    r = requests.get(url, timeout=20)
    stations = json.loads(r.text)

    stns_dict = {}
    for stn in stations:
        # retrieve station identifier
        temp = stn["station"]
        if type(stn["date_installed"]) is int:  # they switched it to milliseconds.
            stn["date_installed"] = date.fromtimestamp(stn["date_installed"] / 1000)
            stn["date_installed"] = stn["date_installed"].strftime("%Y-%m-%d")
        # remove station identifier from existing dictionary
        stn.pop("station")
        # store info as values in dictionary by station identifier keys
        stns_dict[temp] = stn

    return stns_dict

# ========================= EOF ====================================================================
