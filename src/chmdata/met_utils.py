""" This module was adapted from https://github.com/phydrus/pyet """
# Only great_circle_distance is used right now.
# I don't think any of these functions are used in the other modules yet, but they seemed useful.

from typing import Tuple
import urllib

import numpy as np
from numpy import tan, cos, pi, sin, arccos, mod, exp, log, broadcast_to, sqrt, clip, radians
import pandas as pd
import requests

# Specific heat of air [MJ kg-1 °C-1]
CP = 1.013 * 10**-3
# Stefan Boltzmann constant - hourly [MJm-2K-4h-1]
STEFAN_BOLTZMANN_HOUR = 2.042 * 10**-10
# Stefan Boltzmann constant - daily [MJm-2K-4d-1]
STEFAN_BOLTZMANN_DAY = 4.903 * 10**-9


# Spatial Utilities
# -------------------
def elevation_from_coordinate(lat: float, lon: float) -> float:
    """Returns elevation in meters from USGS National Map services given decimal degree coordinates."""
    params = {"output": "json", "x": lon, "y": lat, "units": "Meters"}

    url = r"https://epqs.nationalmap.gov/v1/json?"
    result = requests.get(url + urllib.parse.urlencode(params), timeout=20)
    elev = float(result.json()["value"])
    return elev


def great_circle_distance(here: Tuple[float, float], there: Tuple[float, float]) -> float:
    """Calculate great circle distance between 2 points in km.

    reference: https://en.wikipedia.org/wiki/Great-circle_distance

    Args:
        here: 1st point, (lat, lon), decimal degrees
        there: 2nd point, (lat, lon), decimal degrees

    Returns:
        d: distance in km
    """
    r = 6371.0  # km
    lat1 = radians(here[0])
    lon1 = radians(here[1])
    lat2 = radians(there[0])
    lon2 = radians(there[1])
    central_angle = arccos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(abs(lon1 - lon2)))
    d = r * central_angle
    return d


# Combination PET methods
# -----------------------
def get_index_shape(df):
    """Method to return the index and shape of the input data."""
    try:
        index = pd.DatetimeIndex(df.index)
    except AttributeError:
        index = pd.DatetimeIndex(df.time)
    return index, df.shape


def penman(
    tmean,
    wind,
    rs=None,
    rn=None,
    g=0,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    pressure=None,
    elevation=None,
    lat=None,
    n=None,
    nn=None,
    rso=None,
    aw=2.6,
    bw=0.536,
    a=1.35,
    b=-0.35,
    albedo=0.23,
):
    """Evaporation calculated according to [penman_1948]_.

    Following [penman_1948]_ and [valiantzas_2006]_.

    .. math:: PE = \\frac{\\Delta (R_n-G) + \\gamma 2.6 (1 + 0.536 u_2)
        (e_s-e_a)}{\\lambda (\\Delta +\\gamma)}

    Args:
        tmean: pandas.Series
            average day temperature [°C]
        wind: pandas.Series
            mean day wind speed [m/s]
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        rn: pandas.Series, optional
            net radiation [MJ m-2 d-1]
        g: pandas.Series/int, optional
            soil heat flux [MJ m-2 d-1]
        tmax: pandas.Series, optional
            maximum day temperature [°C]
        tmin: pandas.Series, optional
            minimum day temperature [°C]
        rhmax: pandas.Series, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series, optional
            mean daily relative humidity [%]
        pressure: float, optional
            atmospheric pressure [kPa]
        elevation: float, optional
            the site elevation [m]
        lat: float, optional
            the site latitude [rad]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        aw: float, optional
            wind coefficient [-]
        bw: float, optional
            wind coefficient [-]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        albedo: float, optional
            surface albedo [-]

    Returns:
        pandas.Series containing the calculated evaporation

    Examples:
    >>> et_penman = penman(tmean, wind, rn=rn, rh=rh)

    References:
    .. [penman_1948] Penman, H. L. (1948). Natural evaporation from open water,
       bare soil and grass. Proceedings of the Royal Society of London. Series
       A. Mathematical and Physical Sciences, 193, 120-145.
    .. [valiantzas_2006] Valiantzas, J. D. (2006). Simplified versions for the
       Penman evaporation equation using routine weather data. Journal of
       Hydrology, 331(3-4), 690-702.
    """
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    lambd = calc_lambda(tmean)

    ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = calc_es(tmean=tmean, tmax=tmax, tmin=tmin)

    if rn is None:
        rn = get_rn(tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh, elevation, rso, a, b, ea, albedo)

    fu = aw * (1 + bw * wind)

    den = lambd * (dlt + gamma)
    num1 = dlt * (rn - g) / den
    num2 = gamma * (es - ea) * fu / den
    return num1 + num2


def pm_asce(
    tmean,
    wind,
    rs=None,
    rn=None,
    g=0,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    pressure=None,
    elevation=None,
    lat=None,
    n=None,
    nn=None,
    rso=None,
    a=1.35,
    b=-0.35,
    cn=900,
    cd=0.34,
    ea=None,
    albedo=0.23,
    kab=None,
    as1=0.25,
    bs1=0.5,
    etype="os",
):
    """Evaporation calculated according to [monteith_1965]_.

    Following [monteith_1965]_ and [walter_2000]_
    .. math:: PE = \\frac{\\Delta (R_{n}-G)+ \\rho_a c_p K_{min}
        \\frac{e_s-e_a}{r_a}}{\\lambda(\\Delta +\\gamma(1+\\frac{r_s}{r_a}))}

    Args:
        tmean: pandas.Series
            average day temperature [°C]
        wind: pandas.Series
            mean day wind speed [m/s]
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        rn: pandas.Series, optional
            net radiation [MJ m-2 d-1]
        g: pandas.Series/int, optional
            soil heat flux [MJ m-2 d-1]
        tmax: pandas.Series, optional
            maximum day temperature [°C]
        tmin: pandas.Series, optional
            minimum day temperature [°C]
        rhmax: pandas.Series, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series, optional
            mean daily relative humidity [%]
        pressure: float, optional
            atmospheric pressure [kPa]
        elevation: float, optional
            the site elevation [m]
        lat: float, optional
            the site latitude [rad]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        cn: float, optional
            numerator constant [K mm s3 Mg-1 d-1]
        cd: float, optional
            denominator constant [s m-1]
        ea: pandas.Series/float, optional
            actual vapor pressure [kPa]
        albedo: float, optional
            surface albedo [-]
        kab: float, optional
            coefficient derived from as1, bs1 for estimating clear-sky radiation [degrees].
        as1: float, optional
            regression constant,  expressing the fraction of extraterrestrial
            reaching the earth on overcast days (n = 0) [-]
        bs1: float, optional
            empirical coefficient for extraterrestrial radiation [-]
        etype: str, optional
            "os" => ASCE-PM method is applied for a reference surfaces
            representing clipped grass (a short, smooth crop)
            "rs" => ASCE-PM method is applied for a reference surfaces
            representing alfalfa (a taller, rougher agricultural crop),)

    Returns:
        pandas.Series containing the calculated evaporation

    Examples:
        >>> et_pm = pm_asce(tmean, wind, rn=rn, rh=rh)

    References:
    .. [monteith_1965] Monteith, J. L. (1965). Evaporation and environment.
       In Symposia of the society for experimental biology (Vol. 19, pp.
       205-234). Cambridge University Press (CUP) Cambridge.
    .. [walter_2000] Walter, I. A., Allen, R. G., Elliott, R., Jensen, M. E.,
       Itenfisu, D., Mecham, B., ... & Martin, D. (2000). ASCE's standardized
       reference evapotranspiration equation. In Watershed management and
       operations management 2000 (pp. 1-11).
    """
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    if ea is None:
        ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = calc_es(tmean=tmean, tmax=tmax, tmin=tmin)
    if rn is None:
        rn = get_rn(
            tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh, elevation, rso, a, b, ea, albedo, as1, bs1, kab
        )

    if etype == "rs":
        cn = 1600
        cd = 0.38

    den = dlt + gamma * (1 + cd * wind)
    num1 = (0.408 * dlt * (rn - g)) / den
    num2 = gamma * cn / (tmean + 273) * wind * (es - ea) / den
    return num1 + num2


def pm(
    tmean,
    wind,
    rs=None,
    rn=None,
    g=0,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    pressure=None,
    elevation=None,
    lat=None,
    n=None,
    nn=None,
    rso=None,
    a=1.35,
    b=-0.35,
    lai=None,
    croph=None,
    r_l=100,
    r_s=70,
    ra_method=1,
    a_sh=1,
    a_s=1,
    lai_eff=1,
    srs=0.0009,
    co2=300,
    albedo=0.23,
):
    """Evaporation calculated according to [monteith_1965]_.

    Following [monteith_1965]_, [allen_1998]_, [zhang_2008]_,
    [schymanski_2016]_ and [yang_2019]_.

    .. math:: PE = \\frac{\\Delta (R_{n}-G)+ \\rho_a c_p K_{min}
        \\frac{e_s-e_a}{r_a}}{\\lambda(\\Delta +\\gamma(1+\\frac{r_s}{r_a}))}

    Args:
        tmean: pandas.Series
            average day temperature [°C]
        wind: pandas.Series
            mean day wind speed [m/s]
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        rn: pandas.Series, optional
            net radiation [MJ m-2 d-1]
        g: pandas.Series/int, optional
            soil heat flux [MJ m-2 d-1]
        tmax: pandas.Series, optional
            maximum day temperature [°C]
        tmin: pandas.Series, optional
            minimum day temperature [°C]
        rhmax: pandas.Series, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series, optional
            mean daily relative humidity [%]
        pressure: float, optional
            atmospheric pressure [kPa]
        elevation: float, optional
            the site elevation [m]
        lat: float, optional
            the site latitude [rad]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        lai: pandas.Series/float, optional
            leaf area index [-]
        croph: pandas.series/float, optional
            crop height [m]
        r_l: pandas.series/float, optional
            bulk stomatal resistance [s m-1]
        r_s: pandas.series/float, optional
            bulk surface resistance [s m-1]
        ra_method: float, optional
            1 => ra = 208/wind
            2 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.
        a_s: float, optional
            Fraction of one-sided leaf area covered by stomata (1 if stomata are 1
            on one side only, 2 if they are on both sides)
        a_sh: float, optional
            Fraction of projected area exchanging sensible heat with the air (2)
        lai_eff: float, optional
            0 => LAI_eff = 0.5 * LAI
            1 => LAI_eff = lai / (0.3 * lai + 1.2)
            2 => LAI_eff = 0.5 * LAI; (LAI>4=4)
            3 => see [zhang_2008]_.
        srs: float, optional
            Relative sensitivity of rl to Δ[CO2]
        co2: float
            CO2 concentration [ppm]
        albedo: float, optional
            surface albedo [-]

    Returns:
        pandas.Series containing the calculated evaporation

    Examples:
        >>> et_pm = pm(tmean, wind, rn=rn, rh=rh)

    References:
    .. [monteith_1965] Monteith, J. L. (1965). Evaporation and environment.
       In Symposia of the society for experimental biology (Vol. 19, pp.
       205-234). Cambridge University Press (CUP) Cambridge.
    .. [schymanski_2016] Schymanski, S. J., & Or, D. (2017). Leaf-scale
       experiments reveal an important omission in the Penman–Monteith
       equation. Hydrology and Earth System Sciences, 21(2), 685-706.
    """
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    lambd = calc_lambda(tmean)

    ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = calc_es(tmean=tmean, tmax=tmax, tmin=tmin)

    res_a = calc_res_aero(wind, ra_method=ra_method, croph=croph)
    res_s = calc_res_surf(lai=lai, r_s=r_s, r_l=r_l, lai_eff=lai_eff, srs=srs, co2=co2)
    gamma1 = gamma * a_sh / a_s * (1 + res_s / res_a)

    if rn is None:
        rn = get_rn(tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh, elevation, rso, a, b, ea, albedo)

    kmin = 86400  # unit conversion s d-1
    rho_a = calc_rho(pressure, tmean, ea)

    den = lambd * (dlt + gamma1)
    num1 = dlt * (rn - g) / den
    num2 = rho_a * CP * kmin * (es - ea) * a_sh / res_a / den
    return num1 + num2


def pm_fao56(
    tmean,
    wind,
    rs=None,
    rn=None,
    g=0,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    pressure=None,
    elevation=None,
    lat=None,
    n=None,
    nn=None,
    rso=None,
    a=1.35,
    b=-0.35,
    albedo=0.23,
):
    """Evaporation calculated according to [allen_1998]_.

    .. math:: PE = \\frac{0.408 \\Delta (R_{n}-G)+\\gamma \\frac{900}{T+273}
    (e_s-e_a) u_2}{\\Delta+\\gamma(1+0.34 u_2)}

    Args:
        tmean: pandas.Series
            average day temperature [°C]
        wind: pandas.Series
            mean day wind speed [m/s]
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        rn: pandas.Series, optional
            net radiation [MJ m-2 d-1]
        g: pandas.Series/int, optional
            soil heat flux [MJ m-2 d-1]
        tmax: pandas.Series, optional
            maximum day temperature [°C]
        tmin: pandas.Series, optional
            minimum day temperature [°C]
        rhmax: pandas.Series, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series, optional
            mean daily relative humidity [%]
        pressure: float, optional
            atmospheric pressure [kPa]
        elevation: float, optional
            the site elevation [m]
        lat: float, optional
            the site latitude [rad]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        albedo: float, optional
            surface albedo [-]

    Returns:
        pandas.Series containing the calculated evaporation

    Examples:
        >>> et_fao56 = pm_fao56(tmean, wind, rn=rn, rh=rh)
    """
    if tmean is None:
        tmean = (tmax + tmin) / 2
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)

    gamma1 = gamma * (1 + 0.34 * wind)

    ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = calc_es(tmean=tmean, tmax=tmax, tmin=tmin)

    if rn is None:
        rn = get_rn(tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh, elevation, rso, a, b, ea, albedo)

    den = dlt + gamma1
    num1 = (0.408 * dlt * (rn - g)) / den
    num2 = (gamma * (es - ea) * 900 * wind / (tmean + 273)) / den
    return num1 + num2


def priestley_taylor(
    tmean,
    wind,
    rs=None,
    rn=None,
    g=0,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    pressure=None,
    elevation=None,
    lat=None,
    n=None,
    nn=None,
    rso=None,
    a=1.35,
    b=-0.35,
    alpha=1.26,
    albedo=0.23,
):
    """Evaporation calculated according to [priestley_and_taylor_1965]_.

    .. math:: PE = \\frac{\\alpha_{PT} \\Delta (R_n-G)}
    {\\lambda(\\Delta +\\gamma)}

    Args:
        tmean: pandas.Series
            average day temperature [°C]
        wind: pandas.Series
            mean day wind speed [m/s]
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        rn: pandas.Series, optional
            net radiation [MJ m-2 d-1]
        g: pandas.Series/int, optional
            soil heat flux [MJ m-2 d-1]
        tmax: pandas.Series, optional
            maximum day temperature [°C]
        tmin: pandas.Series, optional
            minimum day temperature [°C]
        rhmax: pandas.Series, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series, optional
            mean daily relative humidity [%]
        pressure: float, optional
            atmospheric pressure [kPa]
        elevation: float, optional
            the site elevation [m]
        lat: float, optional
            the site latitude [rad]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        alpha: float, optional
            calibration coeffiecient [-]
        albedo: float, optional
            surface albedo [-]

    Returns:
        pandas.Series containing the calculated evaporation

    Examples:
        >>> pt = priestley_taylor(tmean, wind, rn=rn, rh=rh)

    References:
        .. [priestley_and_taylor_1965] Priestley, C. H. B., & TAYLOR, R. J. (1972).
           On the assessment of surface heat flux and evaporation using large-scale
           parameters. Monthly weather review, 100(2), 81-92.
    """
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    lambd = calc_lambda(tmean)

    if rn is None:
        rn = get_rn(tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh, elevation, rso, a, b, albedo=albedo)

    return (alpha * dlt * (rn - g)) / (lambd * (dlt + gamma))


def kimberly_penman(
    tmean,
    wind,
    rs=None,
    rn=None,
    g=0,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    pressure=None,
    elevation=None,
    lat=None,
    n=None,
    nn=None,
    rso=None,
    a=1.35,
    b=-0.35,
    albedo=0.23,
):
    """Evaporation calculated according to [wright_1982]_.

    Following [oudin_2005]_.
    .. math:: PE = \\frac{\\Delta (R_n-G)+ \\gamma (e_s-e_a) w}
        {\\lambda(\\Delta +\\gamma)}
    .. math:: w =  u_2 * (0.4 + 0.14 * exp(-(\\frac{J_D-173}{58})^2)) +
            (0.605 + 0.345 * exp(-(\\frac{J_D-243}{80})^2))

    Args:
        tmean: pandas.Series/xarray.DataArray
            average day temperature [°C]
        wind: pandas.Series/xarray.DataArray
            mean day wind speed [m/s]
        rs: pandas.Series/xarray.DataArray, optional
            incoming solar radiation [MJ m-2 d-1]
        rn: pandas.Series/xarray.DataArray, optional
            net radiation [MJ m-2 d-1]
        g: pandas.Series/int/xarray.DataArray, optional
            soil heat flux [MJ m-2 d-1]
        tmax: pandas.Series/xarray.DataArray, optional
            maximum day temperature [°C]
        tmin: pandas.Series/xarray.DataArray, optional
            minimum day temperature [°C]
        rhmax: pandas.Series/xarray.DataArray, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series/xarray.DataArray, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series/xarray.DataArray, optional
            mean daily relative humidity [%]
        pressure: float/xarray.DataArray, optional
            atmospheric pressure [kPa]
        elevation: float/xarray.DataArray, optional
            the site elevation [m]
        lat: float/xarray.DataArray, optional
            the site latitude [rad]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        albedo: float, optional
            surface albedo [-]

    Returns:
        pandas.Series containing the calculated evaporation

    References:
        .. [wright_1982] Wright, J. L. (1982). New evapotranspiration crop
           coefficients. Proceedings of the American Society of Civil Engineers,
           Journal of the Irrigation and Drainage Division, 108(IR2), 57-74.
    """
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    lambd = calc_lambda(tmean)

    ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = calc_es(tmean=tmean, tmax=tmax, tmin=tmin)

    if rn is None:
        rn = get_rn(tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh, elevation, rso, a, b, ea, albedo)

    j = day_of_year(tmean.index)
    w = wind * (0.4 + 0.14 * exp(-(((j - 173) / 58) ** 2)) + (0.605 + 0.345 * exp((j - 243) / 80) ** 2))

    den = lambd * (dlt + gamma)
    num1 = dlt * (rn - g) / den
    num2 = gamma * (es - ea) * w / den
    return num1 + num2


def thom_oliver(
    tmean,
    wind,
    rs=None,
    rn=None,
    g=0,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    pressure=None,
    elevation=None,
    lat=None,
    n=None,
    nn=None,
    rso=None,
    aw=2.6,
    bw=0.536,
    a=1.35,
    b=-0.35,
    lai=None,
    croph=None,
    r_l=100,
    r_s=70,
    ra_method=1,
    lai_eff=1,
    srs=0.0009,
    co2=300,
    albedo=0.23,
):
    """Evaporation calculated according to [thom_1977]_.

    Following [oudin_2005]_.
    .. math:: PE = \\frac{\\Delta (R_{n}-G)+ 2.5 \\gamma (e_s-e_a) w}
        {\\lambda(\\Delta +\\gamma(1+\\frac{r_s}{r_a}))}$
        $w=2.6(1+0.53u_2)

    Args:
        tmean: pandas.Series
            average day temperature [°C]
        wind: pandas.Series
            mean day wind speed [m/s]
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        rn: pandas.Series, optional
            net radiation [MJ m-2 d-1]
        g: pandas.Series/int, optional
            soil heat flux [MJ m-2 d-1]
        tmax: pandas.Series, optional
            maximum day temperature [°C]
        tmin: pandas.Series, optional
            minimum day temperature [°C]
        rhmax: pandas.Series, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series, optional
            mean daily relative humidity [%]
        pressure: float, optional
            atmospheric pressure [kPa]
        elevation: float, optional
            the site elevation [m]
        lat: float, optional
            the site latitude [rad]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        aw: float, optional
            wind coefficient [-]
        bw: float, optional
            wind coefficient [-]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        lai: pandas.Series/float, optional
            leaf area index [-]
        croph: pandas.series/float, optional
            crop height [m]
        r_l: pandas.series/float, optional
            bulk stomatal resistance [s m-1]
        r_s: pandas.series/float, optional
            bulk surface resistance [s m-1]
        ra_method: float, optional
            1 => ra = 208/wind
            2 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.
        lai_eff: float, optional
            0 => LAI_eff = 0.5 * LAI
            1 => LAI_eff = lai / (0.3 * lai + 1.2)
            2 => LAI_eff = 0.5 * LAI; (LAI>4=4)
            3 => see [zhang_2008]_.
        srs: float, optional
            Relative sensitivity of rl to Δ[CO2]
        co2: float
            CO2 concentration [ppm]
        albedo: float, optional
            surface albedo [-]

    Returns:
        pandas.Series containing the calculated evaporation

    References:
        .. [thom_1977] Thom, A. S., & Oliver, H. R. (1977). On Penman's equation
           for estimating regional evaporation. Quarterly Journal of the Royal
           Meteorological Society, 103(436), 345-357.
    """
    if pressure is None:
        pressure = calc_press(elevation)
    gamma = calc_psy(pressure)
    dlt = calc_vpc(tmean)
    lambd = calc_lambda(tmean)

    ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)
    es = calc_es(tmean=tmean, tmax=tmax, tmin=tmin)

    res_a = calc_res_aero(wind, ra_method=ra_method, croph=croph)
    res_s = calc_res_surf(lai=lai, r_s=r_s, r_l=r_l, lai_eff=lai_eff, srs=srs, co2=co2)
    gamma1 = gamma * (1 + res_s / res_a)

    if rn is None:
        rn = get_rn(tmean, rs, lat, n, nn, tmax, tmin, rhmax, rhmin, rh, elevation, rso, a, b, ea, albedo)

    w = aw * (1 + bw * wind)

    den = lambd * (dlt + gamma1)
    num1 = dlt * (rn - g) / den
    num2 = 2.5 * gamma * (es - ea) * w / den
    return num1 + num2


def get_rn(
    tmean,
    rs=None,
    lat=None,
    n=None,
    nn=None,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    elevation=None,
    rso=None,
    a=1.35,
    b=-0.35,
    ea=None,
    albedo=0.23,
    as1=0.25,
    bs1=0.5,
    kab=None,
):
    """Calculate net radiation [MJ/m2/d]."""
    tindex, shape = get_index_shape(tmean)
    rns = calc_rad_short(
        rs=rs, tindex=tindex, lat=lat, n=n, nn=nn, shape=shape, albedo=albedo, as1=as1, bs1=bs1
    )  # [MJ/m2/d]
    rnl = calc_rad_long(
        rs=rs,
        tmean=tmean,
        tmax=tmax,
        tmin=tmin,
        rhmax=rhmax,
        rhmin=rhmin,
        rh=rh,
        elevation=elevation,
        lat=lat,
        rso=rso,
        a=a,
        b=b,
        ea=ea,
        kab=kab,
        tindex=tindex,
        shape=shape,
    )  # [MJ/m2/d]
    rn = rns - rnl
    # rn = rns
    # # print("rns", rns)
    # # print("rnl", rnl)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(rs, label='rs')
    # plt.plot(rso, label='rso')
    # plt.plot(rns, label='rns')
    # plt.plot(rnl, label='rnl')
    # plt.plot(rn, label='rn')
    # plt.legend()
    return rn


# Other meteorology utility functions
# -----------------------------------
def calc_psy(pressure, tmean=None):
    """Psychrometric constant [kPa °C-1].

    if tmean is none:
        Based on equation 8 in [allen_1998]_.
    elif rh is None:
        From FAO (1990), ANNEX V, eq. 4.

    Args:
        pressure: float; atmospheric pressure [kPa].
        tmean: float, optional; average day temperature [°C].

    Returns:
        pandas.Series containing the Psychrometric constant [kPa °C-1].

    Examples:
        >>> psy = calc_psy(pressure, tmean)

    References:
        .. [allen_1998] Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).
           Crop evapotranspiration-Guidelines for computing crop water
           requirements-FAO Irrigation and drainage paper 56. Fao, Rome, 300.
           (http://www.fao.org/3/x0490e/x0490e06.htm#TopOfPage).
    """
    if tmean is None:
        return 0.000665 * pressure
    else:
        lambd = calc_lambda(tmean)  # MJ kg-1
        return CP * pressure / (0.622 * lambd)


def calc_vpc(tmean):
    """Slope of saturation vapour pressure curve at air Temperature [kPa °C-1].

    Based on equation 13. in [allen_1998]_.

    Args:
        tmean: float/pandas.Series/ndarray, optional; average day temperature [°C]

    Returns:
        pandas.Series containing the calculated Saturation vapour pressure
        [kPa °C-1].

    Examples:
        # >>> vpc = calc_vpc(tmean)
    """
    es = calc_e0(tmean)
    return 4098 * es / (tmean + 237.3) ** 2


def calc_lambda(tmean):
    """Latent Heat of Vaporization [MJ kg-1].

    Based on equation (3-1) in [allen_1998]_.

    Parameters:
        tmean: pandas.Series/float, optional; average day temperature [°C]

    Returns:
        pandas.Series containing the calculated Latent Heat of Vaporization [MJ kg-1].

    Examples:
    >>> lambd = calc_lambda(tmean)
    """
    return 2.501 - 0.002361 * tmean


def calc_press(elevation):
    """Atmospheric pressure [kPa].

    Based on equation 7 in [allen_1998]_.

    Args:
        elevation: float, optional; the site elevation [m]

    Returns:
        pandas.Series containing the calculated atmospheric pressure [kPa].

    Examples:
    >>> pressure = calc_press(elevation)
    """
    return 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26


def calc_rho(pressure, tmean, ea):
    """atmospheric air density calculated according to [allen_1998]_..

    Based on equation (3-5) in [allen_1998]_.
    .. math:: rho = 3.486 \\frac{P}{T_{KV}}

    Args:
        pressure: pandas.Series/float; atmospheric pressure [kPa]
        tmean: pandas.Series/float, optional; average day temperature [°C]
        ea: pandas.Series/float, optional; actual vapour pressure [kPa]

    Returns:
        pandas.Series containing the calculated mean air density

    Examples:
        >>> rho = calc_rho(pressure, tmean, ea)
    """
    tkv = (273.16 + tmean) * (1 - 0.378 * ea / pressure) ** -1
    return 3.486 * pressure / tkv


def calc_e0(tmean):
    """Saturation vapor pressure at the air temperature T [kPa].

    Based on equation 11 in [allen_1998]_.

    Args:
        tmean: float/pandas.Series/ndarray, optional; average day temperature [°C]

    Returns:
        pandas.Series containing the calculated saturation vapor pressure at the air temperature tmean [kPa].

    Examples:
        >>> e0 = calc_e0(tmean)
    """
    return 0.6108 * exp(17.27 * tmean / (tmean + 237.3))


def calc_es(tmean=None, tmax=None, tmin=None):
    """Saturation vapor pressure [kPa].

    Based on equation 11, 12 in [allen_1998]_.

    Args:
        tmean: float/pandas.Series/ndarray, optional
            average day temperature [°C]
        tmax: float/pandas.Series/ndarray, optional
            maximum day temperature [°C]
        tmin: float/pandas.Series/ndarray, optional
            minimum day temperature [°C]

    Returns:
        pandas.Series containing the calculated saturation vapor pressure [kPa].

    Examples:
        # >>> es = calc_es(tmean)
    """
    if tmax is not None:
        eamax = calc_e0(tmax)
        eamin = calc_e0(tmin)
        return (eamax + eamin) / 2
    else:
        return calc_e0(tmean)


def calc_ea(tmean=None, tmax=None, tmin=None, rhmax=None, rhmin=None, rh=None):
    """Actual vapor pressure [kPa].

    Based on equation 17, 19 in [allen_1998]_.

    Args:
        tmean: float/pandas.Series/ndarray, optional
            average day temperature [°C]
        tmax: float/pandas.Series/ndarray, optional
            maximum day temperature [°C]
        tmin: float/pandas.Series/ndarray, optional
            minimum day temperature [°C]
        rhmax: float/pandas.Series/ndarray, optional
            maximum daily relative humidity [%]
        rhmin: float/pandas.Series/ndarray, optional
            mainimum daily relative humidity [%]
        rh: float/pandas.Series/ndarray, optional
            mean daily relative humidity [%]

    Returns:
        pandas.Series containing the calculated actual vapor pressure [kPa].

    Examples
        >>> ea = calc_ea(tmean, rh)
    """
    if rhmax is not None:  # eq. 11
        esmax = calc_e0(tmax)
        esmin = calc_e0(tmin)
        return (esmin * rhmax / 200) + (esmax * rhmin / 200)
    else:  # eq. 14
        if tmax is not None:
            es = calc_es(tmax=tmax, tmin=tmin)
        else:
            es = calc_e0(tmean)
        return rh / 100 * es


def day_of_year(tindex):
    """Day of the year (1-365) based on pandas.Index

    Parameters:
        tindex: pandas.Index

    Returns:
        array of with ints specifying day of year.
    """
    return pd.to_numeric(tindex.strftime("%j"))


def daylight_hours(tindex, lat):
    """Daylight hours [hour].

    Based on equation 34 in [allen_1998]_.

    Args:
        tindex: pandas.Index
        lat: float; the site latitude [rad]

    Returns:
        pandas.Series containing the calculated daylight hours [hour]
    """
    j = day_of_year(tindex)
    sol_dec = solar_declination(j)
    sangle = sunset_angle(sol_dec, lat)
    return 24 / pi * sangle


def sunset_angle(sol_dec: float | pd.Series | np.ndarray, lat: float) -> pd.Series:
    """Sunset hour angle from latitude and solar declination - daily [rad].

    Based on equations 25 in [allen_1998]_.

    Args:
        sol_dec: solar declination [rad]
        lat: the site latitude [rad]

    Returns:
        Calculated sunset hour angle - daily [rad]
    """
    return arccos(-tan(sol_dec) * tan(lat))


def _wrap(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Wrap floating point values into range - github.com/WSWUP/RefET

    Args:
        x: Values to wrap.
        x_min: Minimum value in output range.
        x_max: Maximum value in output range.

    Returns:
        processed ndarray.
    """
    return mod((x - x_min), (x_max - x_min)) + x_min


def solar_declination(j):
    """Solar declination from day of year [rad].

    Based on equations 24 in [allen_1998]_.

    Args:
        j: array.py; day of the year (1-365)

    Returns:
        array.py of solar declination [rad].
    """
    return 0.409 * sin(2.0 * pi / 365.0 * j - 1.39)


def relative_distance(j):
    """Inverse relative distance between earth and sun from day of the year.

    Based on equations 23 in [allen_1998]_.

    Args:
        j: array.py; day of the year (1-365)

    Returns:
        array.py specifyng relative distance between earth and sun.
    """
    return 1 + 0.033 * cos(2.0 * pi / 365.0 * j)


def extraterrestrial_r(tindex, lat, shape):
    """Extraterrestrial daily radiation [MJ m-2 d-1].

    Based on equation 21 in [allen_1998]_.

    Args:
        tindex: pandas.Index
        lat: float; the site latitude [rad]
        shape: list; shape of the input file

    Returns:
        Pandas Series of calculated extraterrestrial radiation [MJ m-2 d-1].
    """
    j = broadcast_to(day_of_year(tindex), shape)
    dr = relative_distance(j)
    sol_dec = solar_declination(j)

    omega = sunset_angle(sol_dec, lat)
    xx = sin(sol_dec) * sin(lat)
    yy = cos(sol_dec) * cos(lat)

    return broadcast_to(118.08 / 3.141592654 * dr * (omega * xx + yy * sin(omega)), shape)


def calc_res_surf(lai=None, r_s=70, r_l=100, lai_eff=0, srs=None, co2=None):
    """Surface resistance [s m-1].

    Args:
        lai: pandas.Series/float, optional
            leaf area index [-]
        r_s: pandas.series/float, optional
            surface resistance [s m-1]
        r_l: float, optional
            bulk stomatal resistance [s m-1]
        lai_eff: float, optional
            1 => LAI_eff = 0.5 * LAI
            2 => LAI_eff = lai / (0.3 * lai + 1.2)
            3 => LAI_eff = 0.5 * LAI; (LAI>4=4)
            4 => see [zhang_2008]_.
        srs: float, optional
            Relative sensitivity of rl to Δ[CO2] [yang_2019]_
        co2: float
            CO2 concentration [ppm]

    Returns:
        pandas.Series containing the calculated surface resistance

    References:
        .. [zhang_2008] Zhang, B., Kang, S., Li, F., & Zhang, L. (2008). Comparison
           of three evapotranspiration models to Bowen ratio-energy balance method
           for a vineyard in an arid desert region of northwest China. Agricultural
           and Forest Meteorology, 148(10), 1629-1640.
        .. [yang_2019] Yang, Y., Roderick, M. L., Zhang, S., McVicar, T. R., &
           Donohue, R. J. (2019). Hydrologic implications of vegetation response to
           elevated CO 2 in climate projections. Nature Climate Change, 9, 44-48.
    """
    if lai is None:
        return r_s
    else:
        fco2 = 1 + srs * (co2 - 300)
        return fco2 * r_l / calc_laieff(lai=lai, lai_eff=lai_eff)


def calc_laieff(lai=None, lai_eff=0):
    """Effective leaf area index [-].

    Args:
        lai: pandas.Series/float, optional
            leaf area index [-]
        lai_eff: float, optional
            0 => LAI_eff = 0.5 * LAI
            1 => LAI_eff = lai / (0.3 * lai + 1.2)
            2 => LAI_eff = 0.5 * LAI; (LAI>4=4)
            3 => see [zhang_2008]_.

    Returns:
        pandas.Series containing the calculated effective leaf area index
    """
    if lai_eff == 0:
        return 0.5 * lai
    if lai_eff == 1:
        return lai / (0.3 * lai + 1.2)
    if lai_eff == 2:
        laie = lai.copy()
        laie[(lai > 2) & (lai < 4)] = 2
        laie[lai > 4] = 0.5 * lai
        return laie
    if lai_eff == 3:
        laie = lai.copy()
        laie[lai > 4] = 4
        return laie * 0.5


def calc_res_aero(wind, croph=None, zw=2, zh=2, ra_method=1):
    """Aerodynamic resistance [s m-1].

    Args:
        wind: pandas.Series
            mean day wind speed [m/s]
        croph: pandas.series/float, optional
            crop height [m]
        zw: float, optional
            height of wind measurement [m]
        zh: float, optional
             height of humidity and or air temperature measurement [m]
        ra_method: float, optional
            1 => ra = 208/wind
            2 => ra is calculated based on equation 36 in FAO (1990), ANNEX V.

    Returns:
        pandas.Series containing the calculated aerodynamic resistance
    """
    if ra_method == 1:
        return 208 / wind
    else:
        d = 0.667 * croph
        zom = 0.123 * croph
        zoh = 0.0123 * croph
        return (log((zw - d) / zom)) * (log((zh - d) / zoh) / (0.41**2) / wind)


# Radiation utility functions (originally from rad_utils.py)
# ----------------------------------------------------------
def calc_rad_long(
    rs,
    tindex=None,
    shape=None,
    tmean=None,
    tmax=None,
    tmin=None,
    rhmax=None,
    rhmin=None,
    rh=None,
    elevation=None,
    lat=None,
    rso=None,
    a=1.35,
    b=-0.35,
    ea=None,
    kab=None,
):
    """Net longwave radiation [MJ m-2 d-1].

    Based on equation 39 in [allen_1998]_.

    Args:
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        tindex: pandas.DatetimeIndex
        shape: tuple
            shape of the underlying data.
        tmean: pandas.Series, optional
            average day temperature [°C]
        tmax: pandas.Series, optional
            maximum day temperature [°C]
        tmin: pandas.Series, optional
            minimum day temperature [°C]
        rhmax: pandas.Series, optional
            maximum daily relative humidity [%]
        rhmin: pandas.Series, optional
            mainimum daily relative humidity [%]
        rh: pandas.Series, optional
            mean daily relative humidity [%]
        elevation: float, optional
            the site elevation [m]
        lat: float, optional
            the site latitude [rad]
        rso: pandas.Series/float, optional
            clear-sky solar radiation [MJ m-2 day-1]
        a: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        b: float, optional
            empirical coefficient for Net Long-Wave radiation [-]
        ea: pandas.Series, optional
            actual vapor pressure [kPa]
        kab: float, optional
            coefficient that can be derived from the as and bs coefficients of the Angstrom formula,
            where Kab = as + bs, and where Kab represents the fraction of extraterrestrial radiation
            reaching the earth on clear-sky days [-]

    Returns:
        pandas.Series containing the calculated net longwave radiation.
    """
    if ea is None:
        ea = calc_ea(tmean=tmean, tmax=tmax, tmin=tmin, rhmax=rhmax, rhmin=rhmin, rh=rh)

    if rso is None:
        ra = extraterrestrial_r(tindex, lat, shape)
        rso = calc_rso(ra=ra, elevation=elevation, kab=kab)
    solar_rat = clip(rs / rso, 0.3, 1)
    if tmax is not None:
        tmp1 = STEFAN_BOLTZMANN_DAY * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2
    else:
        tmp1 = STEFAN_BOLTZMANN_DAY * (tmean + 273.16) ** 4

    tmp2 = 0.34 - 0.14 * sqrt(ea)  # OK
    tmp3 = a * solar_rat + b  # OK
    tmp3 = clip(tmp3, 0.05, 1)
    return tmp1 * tmp2 * tmp3


def calc_rad_short(rs=None, tindex=None, lat=None, shape=None, albedo=0.23, n=None, nn=None, as1=0.25, bs1=0.5):
    """Net shortwave radiation [MJ m-2 d-1].

    Based on equation 38 in [allen_1998]_.

    Args:
        rs: pandas.Series, optional
            incoming solar radiation [MJ m-2 d-1]
        tindex: pandas..DatetimeIndex
        lat: float, optional
            the site latitude [rad]
        shape: tuple
            shape of the underlying data.
        albedo: float, optional
            surface albedo [-]
        n: pandas.Series/float, optional
            actual duration of sunshine [hour]
        as1: float, optional
            regression constant,  expressing the fraction of extraterrestrial
            reaching the earth on overcast days (n = 0) [-]
        bs1: float, optional
            empirical coefficient for extraterrestrial radiation [-]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]

    Returns:
        pandas.Series containing the calculated net shortwave radiation
    """
    if rs is not None:
        return (1 - albedo) * rs
    else:
        return (1 - albedo) * calc_rad_sol_in(tindex, lat, shape, n, as1=as1, bs1=bs1, nn=nn)


def calc_rad_sol_in(tindex, lat, shape, n, as1=0.25, bs1=0.5, nn=None):
    """Incoming solar radiation [MJ m-2 d-1].

    Based on equation 35 in [allen_1998]_.

    Args:
        tindex: pandas.DatetimeIndex
        lat: float, optional
            the site latitude [rad]
        shape: tuple
            shape of the underlying data.
        n: pandas.Series/float
            actual duration of sunshine [hour]
        as1: float, optional
            regression constant,  expressing the fraction of extraterrestrial
            reaching the earth on overcast days (n = 0) [-]
        bs1: float, optional
            empirical coefficient for extraterrestrial radiation [-]
        nn: pandas.Series/float, optional
            maximum possible duration of sunshine or daylight hours [hour]

    Returns:
        pandas.Series containing the calculated net shortwave radiation
    """
    ra = extraterrestrial_r(tindex, lat, shape)
    if nn is None:
        nn = daylight_hours(tindex, lat)
    return (as1 + bs1 * n / nn) * ra


def calc_rso(ra, elevation, kab=None):
    """Clear-sky solar radiation [MJ m-2 day-1].

    Based on equation 37 in [allen_1998]_.

    Args:
        ra: pandas.Series, optional
            Extraterrestrial daily radiation [MJ m-2 d-1]
        elevation: float, optional
            the site elevation [m]
        kab: float, optional
            coefficient that can be derived from the as and bs coefficients of the Angstrom formula,
            where Kab = as + bs, and where Kab represents the fraction of extraterrestrial radiation
            reaching the earth on clear-sky days [-]

    Returns:
        pandas.Series containing the calculated Clear-sky solar radiation
    """
    if kab is None:
        return (0.75 + (2 * 10**-5) * elevation) * ra
    else:
        return kab * ra
