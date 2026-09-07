# Fixed THEIA-derived cover layers

This directory is reserved for selected cover fractions derived from THEIA OSO land-cover products. Configurations reference names such as:

```text
LC01_Dense_Urbanization.asc
LC03_Industrial_Zones.asc
LC14_Fruit_Orchards.asc
LC15_Vineyards.asc
LC19_Natural_Heath.asc
LC20_Rocky_Surfaces.asc
LC21_Beaches_Dunes.asc
LC22_Snowy_Glaciers.asc
```

Obtain the appropriate OSO release from the official THEIA catalogue: <https://catalogue.theia.data-terra.org/collection/THEIA_OSO_RASTER_L3B>. Select the release/year used by the study, apply its published nomenclature, and aggregate each class to a fractional cover layer on the common 1-km grid.

Historical configurations note that these layers may initially be percentages (`0–100`) and must be marked as continuous. Record the source release, class code, scaling and whether the layer is temporally fixed or year-specific. Do not commit the rasters.