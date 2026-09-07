# Distances to land-cover classes — derived

These layers are derived from the annual land-cover products in `data/land-cover/`; they are not separate downloads.

Expected filenames follow:

```text
{year}_class{class}_distance.asc
```

For each active class and year, create a binary source mask using the documented threshold/rule and calculate Euclidean distance in metres in a projected CRS. Align the resulting raster exactly to the common `EPSG:2154`, 1-km grid. Record whether distance is measured to any source pixel, to polygons, to pixel centres, or after a minimum-fraction threshold; these choices affect model values materially.

Do not commit these derived rasters.