# Configuration correction policy

The correction workflow does **not** edit the repository's original YAML files.
For each timestamped run it:

1. selects the embedded winter scenario semantically;
2. copies the source scenario into the run's `01_INPUT` and `02_MODELS` tree;
3. applies `multitemporal.include_hunting: true`;
4. forces both `elapid` and `gbm`;
5. assigns an isolated experiment and stack directory;
6. writes a unified `config.diff`;
7. rebuilds the stack from scratch.

This avoids an accidental, unreviewed change to the historical configuration while
providing exact corrected YAML files that can later be committed deliberately.

`review/config/configwinter_monotemporal_hunting.overlay.yaml` is the minimal logical
patch. `00_RUN/disabled_hunting_config_inventory.csv` inventories every structurally
detectable winter-monotemporal occurrence with hunting disabled, including aggregate
scenario YAMLs.
