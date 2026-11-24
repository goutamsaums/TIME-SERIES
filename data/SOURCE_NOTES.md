# ONSA data sources

The two input files are user-supplied processed daily station products attributed to the Nevada Geodetic Laboratory (NGL), University of Nevada, Reno. They are not raw satellite observation (RINEX) files. The precise reference-frame realization is not independently confirmed in this project.

| File | Use in final experiment |
|---|---|
| `ONSA1.CSV` | Daily position and formal uncertainty inputs. Original whitespace-delimited file; preserve capitalization. |
| `ONSA.qa.gz` | Original compressed quality assurance records. Preserve as supplied. |

The original input files have not been changed. The observation archive spans 2 January 1993 to 24 October 2026, with 11,835 daily solutions. The final notebook downloads both input files listed above directly from the `data/` directory of the project GitHub repository.

Source: https://geodesy.unr.edu/ — see also Blewitt, Hammond and Kreemer (2018), https://doi.org/10.1029/2018EO104623. Check the original provider's redistribution conditions before making the data public; this package does not assign a new license to the source data.
