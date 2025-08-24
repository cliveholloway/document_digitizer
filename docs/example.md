# Example

This repo comes with 4 sample scans (from [the project](https://HerbertHolloway.org) that inspired this).

Before you start, make sure the `config.toml` has `log_target` to `stdout`.

You can stitch them together to see the output, and play with the config
values to see what happens by running:

```
document-deskew data/sample_scans output/deskewed -v

# then review the output before stitching

document-stitch output/deskewed output/stitched -v 
```

You can then review the final, stitched output.
