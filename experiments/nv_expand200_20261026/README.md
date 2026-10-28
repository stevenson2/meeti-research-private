# Fixed new-patient N/V experiment: code snapshot

This is an exact, selected Python source snapshot of the completed NV_EXPAND200_20261026 experiment. It is not a portable training package and must not be launched automatically from this upload repository.

The original local research tree owns the data, configuration/freeze files, preprocessing library, initializations, checkpoints and prediction artifacts. Those dependencies are intentionally absent here. Acquisition/vault/key utilities and machine-local settings are not copied. Read SOURCE_MANIFEST.json for file identities and the explicit postfreeze audit-helper correction.

The model definitions contain 62,834 parameters for S/S_LP and 828,514 for L/L_LP. These counts do not prove board memory, operator support or latency. Historical targets in source comments are not claims of achieved accuracy.

Results, aggregate tables and protocol summary: [research stage](../../docs/reports/nv_expand200_20261028/README.md).
