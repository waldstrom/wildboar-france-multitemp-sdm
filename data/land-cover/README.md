# Annual land-cover fractions

This directory is reserved for annual 1-km fractional land-cover layers. The configurations use filenames such as:

```text
2019_class2_CORNMAIZ.asc
2019_class5_AGRILAND.asc
2019_class6_GRASSLND.asc
2019_class7_OAKGROVE.asc
2019_class8_CONIFERS.asc
2019_class9_MIXFORST.asc
2019_class10_BEECHGRV.asc
2019_class12_WATER.asc
```

The full project crosswalk also contains `BGROUND`, wheat, barley, rapeseed, urban and other classes. Do not assume that one upstream product provides every class.

Potential official sources are:

- THEIA OSO annual land-cover maps: <https://catalogue.theia.data-terra.org/collection/THEIA_OSO_RASTER_L3B>
- French Registre Parcellaire Graphique (RPG) for annual crop parcels: <https://geoservices.ign.fr/rpg>

Reconstruction requires the exact project class crosswalk. Harmonise changing upstream nomenclatures across years, rasterise/reclassify at source resolution, calculate the fraction of each 1-km cell covered by each class, and align all outputs to the common grid. Fractions expected by the principal annual configurations are on a `0–1` scale; `BGROUND` should not be included alongside a complete set of component fractions because it is linearly dependent.

The precise source releases and crosswalk used for the published analysis must be added to the provenance record before release. A plausible replacement product is not sufficient for exact numerical reproduction.