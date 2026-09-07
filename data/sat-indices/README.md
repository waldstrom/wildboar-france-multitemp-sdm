# Satellite-index composites — derived

This directory contains annual/monthly vegetation and surface-index composites derived from Sentinel-2 imagery or the explicitly documented source scenes. The configured path pattern is:

```text
data/sat-indices/{index}/{year}_{month}_{index_lc}.tif
```

Active index families include BSI, EVI, EVI2, NBR, NDBI, NDMI, NDSI, NDVI, NDWI and VSSI. Obtain Sentinel-2 Level-2A data through the Copernicus Data Space Ecosystem: <https://dataspace.copernicus.eu/>.

For exact reproduction, document the collection/version, atmospheric product level, cloud and shadow mask, temporal compositing window, spectral-band resampling, index formula, invalid-value handling and spatial aggregation. The `{index_lc}` suffix and any land-cover stratification must match the selected configuration and preprocessing code.

Each index subdirectory contains a placeholder README so the expected layout exists without any imagery.