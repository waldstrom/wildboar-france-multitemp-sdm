# Predictor stacks — generated

This directory is for generated Zarr predictor stacks, for example:

```text
Winter_2017.zarr/
Summer_2023.zarr/
```

The pipeline creates these from the prepared inputs. They can be large, contain redundant copies of source values and are never versioned. Delete and rebuild them whenever the input grid, variable selection or preprocessing configuration changes.