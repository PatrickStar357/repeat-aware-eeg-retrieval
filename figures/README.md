# Editable scientific figures

Figures 1, 2 and 6 use documented THINGSplus-CC0 concept photographs.
They are illustrations, not viewed stimuli or model outputs. Figure 6 shows
all ten actual candidate record IDs for each of six fixed queries, with
24 verified model choices and 44 records selected by none of the scorers.
No original stimulus photograph is used or required for reconstruction.

    python draw_figures.py --inputs source_data --output regenerated --languages en

Add --languages en zh on Windows with SimHei installed. Figures 3--5 retain
paired accuracy slopes, fixed-coverage bars with all participant points, and
signed covariance heatmaps. Source CSVs and all original participant values
are included. Delivered alignment/font/collision reports match the final
exports; fresh alignment checking requires the optional local QA helper.
Style-only reference: AimTS, ICDE 2025, DOI 10.1109/ICDE65448.2025.00149.
