# Terrain variables

This directory contains locally prepared elevation and terrain derivatives. Configurations reference names including:

```text
DEM_Altitude.asc
DEM_Slope.asc
DEM_Aspect.asc
DEM_Elevation_Range.asc
DEM_Terrain_Ruggedness.asc
```

A suitable open source is the Copernicus DEM GLO-30 collection: <https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM>.

Download tiles covering metropolitan France, mosaic them, reproject to `EPSG:2154`, derive slope/aspect/range/ruggedness with a documented GIS implementation, and aggregate to the common 1-km grid. Terrain derivatives depend on algorithm, neighbourhood and edge handling, so record all parameters and software versions. Verify that this source and processing match the original study before claiming numerical reproduction.