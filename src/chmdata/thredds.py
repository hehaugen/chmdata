"""Module for accessing Unidata's Thematic Real-time Environmental Distributed Data Services (THREDDS)

 =============================================================================================
 Copyright 2017 dgketchum

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =============================================================================================

This module contains the following classes:
 - Thredds
 - TopoWX
 - GridMet
 - BBox

Changes that are a departure from the original module under the Apache-2.0 license are annotated
with 'Addition' in the following code.

These changes were made by Todd Blythe, MTDNRC, CNB968
 - 1/28/2025 CHANGE_01: Edited the GeoTransform/CRS attributes for xarray Dataset returned by GridMet.subset_nc()
   so that they are compatible with GDAL (previously were not recognizable).
 - 12/5/2025 CHANGE_02: Added static method to BBox class to import bounds from other package bounding box types
 - 12/5/2025 CHANGE_03: Updated Doc Strings to Google Style and added type hints.
"""

import os
import copy
import warnings
from shutil import rmtree
from tempfile import mkdtemp
from datetime import datetime
from urllib.parse import urlunparse
# Addition - imports from typing
from typing import Union, Optional

import numpy as np
from rasterio import open as rasopen
# Addition - for type hints
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.warp import calculate_default_transform as cdt
from xarray import open_dataset
from xarray import Dataset
import pandas as pd

warnings.simplefilter(action="ignore", category=DeprecationWarning)


class BBox(object):
    # Addition - Added type hints and docstring
    """Bounding box used to subset requested data.

    Attributes:
        west:
            The western (left) limit of the bounding box
        east:
            The eastern (right) limit of the bounding box
        north:
            The northern (top) limit of the bounding box
        south:
            The southern (bottom) limit of the bounding box
    """

    def __init__(self, west: float, east: float, north: float, south: float):
        self.west = west
        self.east = east
        self.north = north
        self.south = south
        # Addition - added docstring
        """Initializes the instance based on bound values.

        Arguments:
            west:
                Float used for west bound of instance.
            east:
                Float used for east bound of instance.
            north:
                Float used for north bound of instance.
            south:
                Float used for south bound of instance.
        """

    # Addition - method to return as tuple so this is a compatible input for Thredds Parent class
    def as_tuple(self):
        """Returns a tuple of the BBox bounds in order WSEN"""
        tup = self.west, self.south, self.east, self.north

        return tup

    # Addition - Static method to initialize instance from list, array, pandas DataFrame, or rasterio BoundingBox
    @staticmethod
    def import_bounds(bnds: Union[list, np.ndarray, pd.DataFrame, BoundingBox]):
        """Function to initilize a BBox instance from other source.

        This function is meant to streamline the creation of a BBox instance from other common representations of
        spatial bounds including a list, array, pandas DataFrame (returned from GeoPandas bounds method), or
        rasterio BoundingBox.

        Args:
            bnds:
                The geospatial bounds to create a BBox bounding instance.

                If input is list or np.ndarry, the order of the bounds is interpreted as:
                 - [x_min, y_min, x_max, y_max] or [west, south, east, north]

                If input is a pd.DataFrame, this is meant to process direct output from GeoPandas.GeoDataFrame.bounds
                attribute. If a custom dataframe is created it must mirror this output format, must contain columns:
                 - minx
                 - miny
                 - maxx
                 - maxy
                Also, if there is more than one row in the DataFrame, the total bounds will be used, (e.g.,the
                minimum of the 'minx' column will be used as 'west', and so on).

        Returns:
            An instance of this class.

        """
        if isinstance(bnds, (list, np.ndarray)):
            box = BBox(bnds[0], bnds[2], bnds[3], bnds[1])
        elif isinstance(bnds, pd.DataFrame):
            box = BBox(bnds["minx"].min(), bnds["maxx"].max(), bnds["maxy"].max(), bnds["miny"].min())
        elif isinstance(bnds, BoundingBox):
            box = BBox(bnds.left, bnds.right, bnds.top, bnds.bottom)
        else:
            raise ValueError(
                "The input bounds are of an unrecognized type, must be list, numpy array, rasterio"
                "BoundingBox, or pandas DataFrame."
            )

        return box


# TODO: address missing(?) attributes in Thredds
class Thredds:
    # Addition - Edited class docstring
    """  Unidata's Thematic Real-time Environmental Distributed Data Services (THREDDS)

    Parent class for accessing different datasets through THREDDS.

    Attributes:
        start:
            The start date, or first date, of the requested time-dimensioned data.
        end:
            The end date, or last date, of the requested time-dimensioned data.
        date:
            The date for a single day of the requested time-dimensioned data.
        src_bounds_wsen:
            A tuple of the extent/bounds over which data is requested, ordered West, South, East, North.
        target_profile:
            A dictionary of raster metadata returned from rasterio.DatasetReader.profile attribute to mimic
            (i.e., conform requested data to this target raster).
        bbox:
            A bounding box object that houses the spatial bounds for which data is requested.
        lat:
            A latitude value for requesting data at a single point.
        lon:
            A longitude value for requesting data at a single point.
        clip_feature:
            A geojson style dictionary representing a shape/polygon to extract data from that area.
        temp_dir:
            Temporary directory created by some internal methods/processes.
        projection:
            Path to temporary projection geotiff used by internal methods/processes.
        reprojection:
            Path to temporary reprojected geotiff used by internal methods/processes.
        mask:
            Path to temporary raster mask geotiff for clipping to a feature.
    """

    # Addition - added type hints
    def __init__(self,
                 start: Optional[datetime] = None,
                 end: Optional[datetime] = None,
                 date: Optional[datetime] = None,
                 bounds: Optional[BBox] = None,
                 target_profile: Optional[dict] = None,
                 lat: Optional[float] = None,
                 lon: Optional[float] = None,
                 clip_feature: Optional[dict] = None):
        # Addition - added docstring
        """Initializes the instance based on optional inputs

        Args:
            start:
                Sets start date of instance.
            end:
                Sets end date of instance.
            date:
                Sets date of instance.
            bounds:
                Defines the bounding box used by the instance to request data.
            target_profile:
                Sets a target dataset for data returned by the instance to mimic.
            lat:
                Sets the latitude of the instance.
            lon:
                Sets the longitude of the instance.
            clip_feature:
                Defines a geometry used by the instance to extract data.

                This is limited by the rasterio.mask.mask() input types which are geojson style dict or
                a shapely Polygon object.
        """
        self.start = start
        self.end = end
        self.date = date

        self.src_bounds_wsen = None

        self.target_profile = target_profile
        self.bbox = bounds
        self.lat = lat
        self.lon = lon
        self.clip_feature = clip_feature
        self._is_masked = False

    def conform(self,
                subset: np.ndarray,
                out_file: Optional[str] = None) -> np.ndarray:
        # Addition - added docstring
        """Conforms raster dataset to target raster dataset and clip feature.

        Args:
            subset:
                The data array returned raster data.
            out_file:
                A path-like string of where to save the conformed dataset.

        Returns:
            The conformed (reprojected, clipped, and resampled raster data array).
        """
        if subset.dtype != np.float32:
            subset = np.array(subset, dtype=np.float32)
        self._project(subset)
        self._warp()
        self._mask()
        result = self._resample()
        if out_file:
            self.save_raster(result, self.target_profile, output_filename=out_file)
        return result

    def _project(self, subset):

        proj_path = os.path.join(self.temp_dir, "tiled_proj.tif")
        setattr(self, "projection", proj_path)

        profile = copy.deepcopy(self.target_profile)
        profile["dtype"] = np.float32
        bb = self.bbox.as_tuple()

        if self.src_bounds_wsen:
            bounds = self.src_bounds_wsen
        else:
            bounds = (bb[0], bb[1], bb[2], bb[3])

        dst_affine, dst_width, dst_height = cdt(
            CRS({"init": "epsg:4326"}),
            CRS({"init": "epsg:4326"}),
            subset.shape[1],
            subset.shape[2],
            *bounds,
        )

        profile.update(
            {
                "crs": CRS({"init": "epsg:4326"}),
                "transform": dst_affine,
                "width": dst_width,
                "height": dst_height,
                "count": subset.shape[0],
            }
        )

        with rasopen(proj_path, "w", **profile) as dst:
            dst.write(subset)

    def _warp(self):

        reproj_path = os.path.join(self.temp_dir, "reproj.tif")
        setattr(self, "reprojection", reproj_path)

        with rasopen(self.projection, "r") as src:
            src_profile = src.profile
            src_bounds = src.bounds
            src_array = src.read()

        dst_profile = copy.deepcopy(self.target_profile)
        dst_profile["dtype"] = np.float32
        bounds = src_bounds
        dst_affine, dst_width, dst_height = cdt(
            src_profile["crs"], dst_profile["crs"], src_profile["width"], src_profile["height"], *bounds
        )

        dst_profile.update(
            {
                "crs": dst_profile["crs"],
                "transform": dst_affine,
                "width": dst_width,
                "height": dst_height,
                "count": src_array.shape[0],
            }
        )

        with rasopen(reproj_path, "w", **dst_profile) as dst:
            dst_array = np.empty((src_array.shape[0], dst_height, dst_width), dtype=np.float32)

            reproject(
                src_array,
                dst_array,
                src_transform=src_profile["transform"],
                src_crs=src_profile["crs"],
                dst_crs=self.target_profile["crs"],
                dst_transform=dst_affine,
                resampling=Resampling.bilinear,
                num_threads=2,
            )

            dst.write(dst_array)

    def _mask(self):

        mask_path = os.path.join(self.temp_dir, "masked.tif")
        with rasopen(self.reprojection) as src:
            out_arr, out_trans = mask(src, self.clip_feature, crop=True, all_touched=True)
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_arr.shape[1],
                    "width": out_arr.shape[2],
                    "transform": out_trans,
                    "count": out_arr.shape[0],
                }
            )

        with rasopen(mask_path, "w", **out_meta) as dst:
            dst.write(out_arr)

        self._is_masked = True

        setattr(self, "mask", mask_path)

    def _resample(self):

        # home = os.path.expanduser('~')
        # resample_path = os.path.join(home, 'images', 'sandbox', 'thredds', 'resamp_twx_{}.tif'.format(var))

        resample_path = os.path.join(self.temp_dir, "resample.tif")

        if self._is_masked:
            ras_obj = self.mask
        else:
            ras_obj = self.reprojection

        with rasopen(ras_obj, "r") as src:
            array = src.read()
            profile = src.profile
            res = src.res
            try:
                target_affine = self.target_profile["affine"]
            except KeyError:
                target_affine = self.target_profile["transform"]
            target_res = target_affine.a
            res_coeff = res[0] / target_res

            new_array = np.empty(
                shape=(array.shape[0], round(array.shape[1] * res_coeff), round(array.shape[2] * res_coeff)),
                dtype=np.float32,
            )
            aff = src.transform
            new_affine = Affine(aff.a / res_coeff, aff.b, aff.c, aff.d, aff.e / res_coeff, aff.f)

            profile.update(
                {
                    "transform": self.target_profile["transform"],
                    "width": self.target_profile["width"],
                    "height": self.target_profile["height"],
                    "dtype": str(new_array.dtype),
                    "count": new_array.shape[0],
                }
            )

            try:
                delattr(self, "mask")
            except AttributeError:
                pass
            delattr(self, "reprojection")

            with rasopen(resample_path, "w", **profile) as dst:
                reproject(
                    array,
                    new_array,
                    src_transform=aff,
                    dst_transform=new_affine,
                    src_crs=src.crs,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest,
                )

                dst.write(new_array)

            with rasopen(resample_path, 'r') as rsrc:
                arr = rsrc.read()

            return arr

    def _date_index(self):
        date_ind = pd.date_range(self.start, self.end, freq="d")

        return date_ind

    @staticmethod
    def _dtime_to_dtime64(dtime):
        dtnumpy = np.datetime64(dtime).astype(np.datetime64)
        return dtnumpy

    @staticmethod
    def save_raster(arr, geometry, output_filename):
        # Addition - added docstring
        """Saves output raster data array."""
        try:
            arr = arr.reshape(1, arr.shape[1], arr.shape[2])
        except IndexError:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        geometry["dtype"] = str(arr.dtype)

        with rasopen(output_filename, "w", **geometry) as dst:
            dst.write(arr)
        return None


class TopoWX(Thredds):
    # Addition - edited docstring
    """TopoWX Surface Temperature, return as numpy array in daily stack unless modified.

    Available variables: [ 'tmmn', 'tmmx']

    ----------
    Observation elements to access. Currently available elements:
    - 'tmmn' : daily minimum air temperature [K]
    - 'tmmx' : daily maximum air temperature [K]

    variables:
        List of available variables = ['tmin', 'tmax'].
    service:
        A string used to construct data download URL = 'cida.usgs.gov'
    scheme:
        A string used to construct data download URL = 'https'
    year:
        The year attribute of the start time datetime object.
    tmin:
        Returned minimum temperature data as a numpy array, after running get_data_subset() with var='tmin'
    tmax:
        Returned maximum temperature data as a numpy array, after running get_data_subset() with var='tmax'
"""

    def __init__(self, **kwargs):
        """Initializes the instance, see parent class for possible key-word arguments."""
        Thredds.__init__(self)

        self.temp_dir = mkdtemp()

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.service = "cida.usgs.gov"
        self.scheme = "https"
        self.variables = ["tmin", "tmax"]

        if self.date:
            self.start = self.date
            self.end = self.date

        self.year = self.start.year

    # Addition - added type hints
    def get_data_subset(self,
                        var: str = 'tmax',
                        temp_units_out: str = 'C',
                        grid_conform: bool = False,
                        out_file: Optional[str] = None) -> Optional[np.ndarray]:
        # Addition - Added docstring
        """Function to get a subset of TopoWX temperature data.

        Subsets data based on time window and spatial constraints specified for the instance.

        Args:
            var:
                String of which dataset to acquire, either 'tmin' or 'tmax'
            temp_units_out:
                String for which units to use for returned temperature data, either 'C' for Celcius or 'K' for Kelvin
            grid_conform:
                Boolean to determine whether to conform the returned data to a clipped region and target dataset,
                default = False.
            out_file:
                A file path for saving conformed data. Ignored if grid_conform = False.

        Returns:
            numpy array of the subset data.
        """
        if var not in self.variables:
            raise TypeError('Must choose from "tmax" or "tmin".')

        url = self._build_url(var)
        xray = open_dataset(url)

        start = self._dtime_to_dtime64(self.start)
        end = self._dtime_to_dtime64(self.end)

        if self.date:
            end = end + np.timedelta64(1, "D")

        # find index and value of bounds
        # 1/100 degree adds a small buffer for this 800 m res data
        north_ind = np.argmin(abs(xray.lat.values - (self.bbox.north + 1.0)))
        south_ind = np.argmin(abs(xray.lat.values - (self.bbox.south - 1.0)))
        west_ind = np.argmin(abs(xray.lon.values - (self.bbox.west - 1.0)))
        east_ind = np.argmin(abs(xray.lon.values - (self.bbox.east + 1.0)))

        north_val = xray.lat.values[north_ind]
        south_val = xray.lat.values[south_ind]
        west_val = xray.lon.values[west_ind]
        east_val = xray.lon.values[east_ind]

        setattr(self, "src_bounds_wsen", (west_val, south_val, east_val, north_val))

        subset = xray.loc[dict(time=slice(start, end), lat=slice(north_val, south_val), lon=slice(west_val, east_val))]

        date_ind = self._date_index()
        subset["time"] = date_ind

        if not grid_conform:
            setattr(self, var, subset)

        else:
            if var == "tmin":
                arr = subset.tmin.values
            elif var == "tmax":
                arr = subset.tmax.values
            else:
                arr = None

            if temp_units_out == "K":
                arr += 273.15

            conformed_array = self.conform(arr, out_file=out_file)

            return conformed_array

    def _build_url(self, var):

        # ParseResult('scheme', 'netloc', 'path', 'params', 'query', 'fragment')
        url = urlunparse(
            [
                self.scheme,
                self.service,
                "/thredds/dodsC/topowx?crs,lat[0:1:3249],lon[0:1:6999],{}," "time".format(var),
                "",
                "",
                "",
            ]
        )

        return url


class GridMet(Thredds):
    # Addition - Documentation reformatted to Google style and edited docstring
    """ U of I Gridmet

    Return as numpy array per met variable in daily stack unless modified.

    Available variables: ['bi', 'elev', 'erc', 'fm100', fm1000', 'pdsi', 'pet', 'pr', 'rmax', 'rmin', 'sph', 'srad',
                          'th', 'tmmn', 'tmmx', 'vs']
        ----------
        Observation elements to access. Currently available elements:
        - 'bi' : burning index [-]
        - 'elev' : elevation above sea level [m]
        - 'erc' : energy release component [-]
        - 'fm100' : 100-hour dead fuel moisture [%]
        - 'fm1000' : 1000-hour dead fuel moisture [%]
        - 'pdsi' : Palmer Drough Severity Index [-]
        - 'pet' : daily reference potential evapotranspiration [mm]
        - 'pr' : daily accumulated precipitation [mm]
        - 'rmax' : daily maximum relative humidity [%]
        - 'rmin' : daily minimum relative humidity [%]
        - 'sph' : daily mean specific humidity [kg/kg]
        - 'prcp' : daily total precipitation [mm]
        - 'srad' : daily mean downward shortwave radiation at surface [W m-2]
        - 'th' : daily mean wind direction clockwise from North [degrees]
        - 'tmmn' : daily minimum air temperature [K]
        - 'tmmx' : daily maximum air temperature [K]
        - 'vs' : daily mean wind speed [m -s]

    variable:
        String of which GridMET variable to request
    service:
        A string used to construct data download URL = 'thredds.northwestknowledge.net:8080'
    scheme:
        A string used to construct data download URL = 'http'
    kwords:
        A dictionary mapping GridMET variable strings to full variable names
    """
    # Addition - Added type hints
    def __init__(self,
                 variable: Optional[str],
                 date: Optional[Union[str, datetime]] = None,
                 start: Optional[Union[str, datetime]] = None,
                 end: Optional[Union[str, datetime]] = None,
                 bbox: Optional[BBox] = None,
                 target_profile: Optional[dict] = None,
                 clip_feature: Optional[dict] = None,
                 lat: Optional[float] = None,
                 lon: Optional[float] = None):
        # Addition - added docstring
        """Initializes the instance based on user input.

        Args:
            variable:
                The string for which GridMET variable to request.
            date:
                **See Parent Class, Thredds**
            start:
                **See Parent Class, Thredds**
            end:
                **See Parent Class, Thredds**
            bbox:
                **See Parent Class, Thredds**
            target_profile:
                **See Parent Class, Thredds**
            clip_feature:
                **See Parent Class, Thredds**
            lat:
                **See Parent Class, Thredds**
            lon:
                **See Parent Class, Thredds**
        """
        Thredds.__init__(self)

        self.date = date
        self.start = start
        self.end = end

        if isinstance(start, str):
            self.start = datetime.strptime(start, "%Y-%m-%d")
        if isinstance(end, str):
            self.end = datetime.strptime(end, "%Y-%m-%d")
        if isinstance(date, str):
            self.date = datetime.strptime(date, "%Y-%m-%d")

        self.variable = variable

        if variable != "elev":
            if self.start and self.end is None:
                raise AttributeError("Must set both start and end date")

        self.bbox = bbox
        self.target_profile = target_profile
        self.clip_feature = clip_feature
        self.lat = lat
        self.lon = lon

        self.service = "thredds.northwestknowledge.net:8080"
        self.scheme = "http"

        self.temp_dir = mkdtemp()

        self.available = [
            "elev",
            "pr",
            "rmax",
            "rmin",
            "sph",
            "srad",
            "th",
            "tmmn",
            "tmmx",
            "pet",
            "vs",
            "erc",
            "bi",
            "fm100",
            "pdsi",
        ]

        if self.variable not in self.available:
            Warning("Variable {} is not available".format(self.variable))

        self.kwords = {
            "bi": "daily_mean_burning_index_g",
            "elev": "",
            "erc": "energy_release_component-g",
            "fm100": "dead_fuel_moisture_100hr",
            "fm1000": "dead_fuel_moisture_1000hr",
            "pdsi": "daily_mean_palmer_drought_severity_index",
            "etr": "daily_mean_reference_evapotranspiration_alfalfa",
            "pet": "daily_mean_reference_evapotranspiration_grass",
            "pr": "precipitation_amount",
            "rmax": "daily_maximum_relative_humidity",
            "rmin": "daily_minimum_relative_humidity",
            "sph": "daily_mean_specific_humidity",
            "srad": "daily_mean_shortwave_radiation_at_surface",
            "th": "daily_mean_wind_direction",
            "tmmn": "daily_minimum_temperature",
            "tmmx": "daily_maximum_temperature",
            "vs": "daily_mean_wind_speed",
            "vpd": "daily_mean_vapor_pressure_deficit",
        }

        self.units = {
            "bi": "-",  # related to 10 times the flame length
            "elev": "m",
            "erc": "-",  # related to available energy (BTU) per unit area (ft^2)
            "fm100": "%",  # water in fuel available to fire, percent of dry weight. 1-3 in diameter veg
            "fm1000": "%",  # water in fuel available to fire, percent of dry weight. 3-8 in diameter veg
            "pdsi": "-",  # based on temp and precip data. Generally -10 to +10, outside -4 to +4 is extreme
            "etr": "mm",
            "pet": "mm",
            "pr": "mm",
            "rmax": "%",
            "rmin": "%",
            "sph": "kg/kg",
            "srad": "w/m^2",
            "th": "degrees",
            "tmmn": "k",
            "tmmx": "k",
            "vs": "m/s",
            "vpd": "kpa",
        }

        if variable != "elev":
            if self.date:
                self.start = self.date
                self.end = self.date

            if self.start.year < self.end.year:
                self.single_year = False

            if self.start > self.end:
                raise ValueError("start date is after end date")

        if not self.bbox and not self.lat:
            raise AttributeError("No bbox or coordinates given")

    def subset_daily_tif(self, out_filename: Optional[str]) -> np.ndarray:
        # Addition - added docstring
        """Acquires and manipulates GridMET data as a raster data array.

        Subsets GridMET variable of choice for spatial and time constraints. Also conforms subset data based on
        target raster and clip feature if specified in the instance.

        Args:
            out_filename:
                File path string where conformed dataset is saved to.

        Returns:
            An array of the requested data subset, with any transformations/clipping applied.
        """
        url = self._build_url()
        url = url + "#fillmismatch"
        xray = open_dataset(url, decode_times=True)

        north_ind = np.argmin(abs(xray.lat.values - (self.bbox.north + 1.0)))
        south_ind = np.argmin(abs(xray.lat.values - (self.bbox.south - 1.0)))
        west_ind = np.argmin(abs(xray.lon.values - (self.bbox.west - 1.0)))
        east_ind = np.argmin(abs(xray.lon.values - (self.bbox.east + 1.0)))

        north_val = xray.lat.values[north_ind]
        south_val = xray.lat.values[south_ind]
        west_val = xray.lon.values[west_ind]
        east_val = xray.lon.values[east_ind]

        setattr(self, "src_bounds_wsen", (west_val, south_val, east_val, north_val))

        if self.variable == "elev":
            subset = xray.loc[
                dict(
                    lat=slice((self.bbox.north + 1), (self.bbox.south - 1)),
                    lon=slice((self.bbox.west - 1), (self.bbox.east + 1)),
                )
            ]
            setattr(self, "width", subset.dims["lon"])
            setattr(self, "height", subset.dims["lat"])
            arr = subset.elevation.values
            arr = self.conform(arr, out_file=out_filename)
            return arr

        else:
            xray = xray.rename({"day": "time"})
            subset = xray.loc[
                dict(time=slice(self.start, self.end), lat=slice(north_val, south_val), lon=slice(west_val, east_val))
            ]

            setattr(self, "width", subset.dims["lon"])
            setattr(self, "height", subset.dims["lat"])
            arr = subset[self.kwords[self.variable]].values
            arr = self.conform(arr, out_file=out_filename)
            rmtree(self.temp_dir)
            return arr

    def subset_nc(self, out_filename=None, return_array=False) -> Union[None, Dataset]:
        # Addition - added docstring
        """Acquires and manipulates GridMET data as xarray or netcdf.

        Args:
            out_filename:
                String of file path to save the resulting data as a netcdf. If specified, data is saved to this path,
                default = None.
            return_array:
                Boolean to determine if the xarray dataset requested is returned.

        Returns:
            If return_array = True, returns an xarray Dataset with the requested data.
        """
        url = self._build_url()
        url = url + "#fillmismatch"
        xray = open_dataset(url)

        north_ind = np.argmin(abs(xray.lat.values - self.bbox.north))
        south_ind = np.argmin(abs(xray.lat.values - self.bbox.south))
        west_ind = np.argmin(abs(xray.lon.values - self.bbox.west))
        east_ind = np.argmin(abs(xray.lon.values - self.bbox.east))

        north_val = xray.lat.values[north_ind]
        south_val = xray.lat.values[south_ind]
        west_val = xray.lon.values[west_ind]
        east_val = xray.lon.values[east_ind]

        setattr(self, "src_bounds_wsen", (west_val, south_val, east_val, north_val))

        if self.variable != "elev":
            xray = xray.rename({"day": "time"})
            subset = xray.loc[
                dict(time=slice(self.start, self.end), lat=slice(north_val, south_val), lon=slice(west_val, east_val))
            ]

            geotrans = subset.crs.GeoTransform.split(" ")
            new_geotran = " ".join([geotrans[0], geotrans[1], geotrans[2], geotrans[4], "0.0", geotrans[5]])
            subset["crs"] = subset.crs.assign_attrs(crs_wkt=subset.crs.attrs["spatial_ref"], GeoTransform=new_geotran)

            date_ind = self._date_index()
            subset["time"] = date_ind
            if out_filename:
                subset.to_netcdf(out_filename)
            if return_array:
                return subset

        else:
            subset = xray.loc[dict(lat=slice((self.bbox.north + 1),
                                             (self.bbox.south - 1)),
                                   lon=slice((self.bbox.west - 1),
                                             (self.bbox.east + 1)))]
            # Addition - to make the CRS information recognizable by GDAL
            geotrans = subset.crs.GeoTransform.split(' ')
            new_geotran = ' '.join([geotrans[0], geotrans[1], geotrans[2], geotrans[4], '0.0', geotrans[5]])
            subset['crs'] = subset.crs.assign_attrs(crs_wkt=subset.crs.attrs['spatial_ref'], GeoTransform=new_geotran)

            if out_filename:
                subset.to_netcdf(out_filename)
            if return_array:
                return subset

    def get_point_timeseries(self):
        # Addition - added docstring
        """Returns pandas DataFrame of requested GridMET time-series data for a single point."""

        url = self._build_url()
        url = url + "#fillmismatch"
        xray = open_dataset(url)
        subset = xray.sel(lon=self.lon, lat=self.lat, method="nearest")
        subset = subset.loc[dict(day=slice(self.start, self.end))]
        # Updating coordinates to be actual gridmet centroid
        self.lon = round(float(subset.coords["lon"].values), 3)
        self.lat = round(float(subset.coords["lat"].values), 3)
        subset = subset.rename({"day": "time"})
        date_ind = self._date_index()
        subset["time"] = date_ind
        time = subset["time"].values
        series = subset[self.kwords[self.variable]].values
        df = pd.DataFrame(data=series, index=time)
        df.columns = [self.variable]
        return df

    def get_point_elevation(self):
        # Addition - added docstring
        """Returns elevation at a single point."""

        url = self._build_url()
        url = url + "#fillmismatch"
        xray = open_dataset(url)
        subset = xray.sel(lon=self.lon, lat=self.lat, method="nearest")
        # Updating coordinates to be actual gridmet centroid
        self.lon = round(float(subset.coords["lon"].values), 3)
        self.lat = round(float(subset.coords["lat"].values), 3)
        elev = subset.get("elevation").values[0]
        return elev

    def _build_url(self):

        # ParseResult('scheme', 'netloc', 'path', 'params', 'query', 'fragment')
        if self.variable == "elev":
            url = urlunparse(
                [
                    self.scheme,
                    self.service,
                    "/thredds/dodsC/MET/{0}/metdata_elevationdata.nc".format(self.variable),
                    "",
                    "",
                    "",
                ]
            )
        else:
            url = urlunparse(
                [
                    self.scheme,
                    self.service,
                    "/thredds/dodsC/agg_met_{}_1979_CurrentYear_CONUS.nc".format(self.variable),
                    "",
                    "",
                    "",
                ]
            )

        return url

    def write_netcdf(self, outputroot):
        # Addition - added docstring
        """Saves GridMET data as netcdf

        Args:
            outputroot:
                File path to save netcdf to.
        """
        url = self._build_url()
        xray = open_dataset(url)
        if self.variable != "elev":
            subset = xray.loc[dict(day=slice(self.start, self.end))]
            subset.rename({"day": "time"}, inplace=True)
        else:
            subset = xray
        subset.to_netcdf(path=outputroot, engine="netcdf4")

# ========================= EOF ====================================================================
