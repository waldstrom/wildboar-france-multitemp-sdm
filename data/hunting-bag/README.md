# Wild-boar hunting-bag covariate

The winter monotemporal variant can include annual departmental wild-boar hunting bags using filenames such as `hunting_bag_2017.asc` through `hunting_bag_2023.asc`.

A public starting point is the French wild-ungulate monitoring network and its departmental hunting-table series:

- OFB network: <https://ofb.gouv.fr/reseau-ongules-sauvages>
- Open-data catalogue: <https://www.data.gouv.fr/fr/datasets/evolution-des-tableaux-de-chasse-departementaux-du-grand-gibier-en-france-donnees-depuis-1973/>

Before rasterisation, verify how hunting seasons are assigned to model years, whether the predictor represents totals, area-standardised density or another transformation, and how missing departments are handled. Join the selected annual values to the same departmental boundaries used by the study, rasterise to the common 1-km grid, and document the boundary vintage and all transformations.

Hunting bags are an ecological proxy, not occurrence records. Keep their interpretation and temporal alignment explicit.