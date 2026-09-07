# ERA5-Land seasonal covariates

Obtain ERA5-Land data from the Copernicus Climate Data Store: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land>. Dataset DOI: <https://doi.org/10.24381/cds.e2161bac>.

The active configuration and feature list determine the required variables, years, seasons and summary statistics. Existing filename patterns follow forms such as:

```text
2m_temperature_2019_summer_mean.asc
2m_temperature_2019_winter_min.asc
<variable>_<year>_<season>_<statistic>.asc
```

Preparation steps:

1. Download the required hourly or monthly ERA5-Land variables for the full seasonal windows.
2. Apply the study's Summer/Winter definition consistently, including the year assigned to cross-calendar winter months.
3. Convert units before aggregation where required.
4. Calculate the configured seasonal statistics or anomalies.
5. crop/reproject/align to the `EPSG:2154`, 1-km study grid and write `-9999` as nodata.
6. Record the CDS request, retrieval date, variable identifiers, unit conversions and checksums.

Do not commit NetCDF/GRIB downloads or derived ASCII rasters.