# Corrected shared-encoder Wishart comparison

Every architecture now has exactly one encoder.

`output_split` expands only the final intensity head from C=5 to K*C=15.
The pre-head history representation is exactly identical for all three
mixture components; component k uses output rows `5k:5(k+1)`.

`lal_fixed` keeps a shared five-output head and introduces only the
LaL-style cluster state: CT-LSTM initial state for NHP and BOS embedding for
THP/COTIC.  It is initialized by two LaL splits to K=3 and then trained at
fixed K.  There are no split/merge/delete walks in either parameterization.

Both branches are evaluated with and without the same latent law
`W ~ Wishart_15(20, M/20)`.  The likelihood is the classical continuous-time
TPP likelihood with a terminal compensator; W is integrated by Monte Carlo.

## Likelihood

| architecture | parameterization | model | parameter_count | best_epoch | suffix_nll_per_exposure | delta_suffix_vs_no_w | full_marginal_nll_per_exposure |
|---|---|---|---|---|---|---|---|
| nhp | output_split | no_wishart | 4113 | 295 | 3.551045 | 0.000000 | 3.902924 |
| nhp | output_split | latent_wishart_attention | 4338 | 270 | 3.460866 | -0.090180 | 3.842290 |
| nhp | lal_fixed | no_wishart | 4059 | 290 | 3.577608 | 0.000000 | 3.916487 |
| nhp | lal_fixed | latent_wishart_attention | 4284 | 190 | 3.471475 | -0.106133 | 3.874753 |
| thp | output_split | no_wishart | 11295 | 120 | 3.628967 | 0.000000 | 3.950262 |
| thp | output_split | latent_wishart_attention | 11520 | 210 | 3.465284 | -0.163682 | 3.818307 |
| thp | lal_fixed | no_wishart | 11031 | 140 | 3.535851 | 0.000000 | 3.878920 |
| thp | lal_fixed | latent_wishart_attention | 11256 | 155 | 3.485538 | -0.050313 | 3.833652 |
| cotic | output_split | no_wishart | 58345 | 110 | 3.899844 | 0.000000 | 4.207411 |
| cotic | output_split | latent_wishart_attention | 58570 | 95 | 3.639093 | -0.260751 | 3.929130 |
| cotic | lal_fixed | no_wishart | 57781 | 80 | 3.872938 | 0.000000 | 4.102616 |
| cotic | lal_fixed | latent_wishart_attention | 58006 | 135 | 3.661768 | -0.211170 | 3.951349 |

## Clustering

| architecture | parameterization | model | representation | purity | ari | active_k | cluster_sizes | mean_entropy |
|---|---|---|---|---|---|---|---|---|
| nhp | output_split | no_wishart | prefix_half | 0.8479 | 0.6043 | 3 | [305, 354, 406] | 0.2882 |
| nhp | output_split | no_wishart | full_sequence | 0.9418 | 0.8351 | 3 | [350, 347, 368] | 0.0766 |
| nhp | output_split | latent_wishart_attention | prefix_half | 0.8160 | 0.5489 | 3 | [215, 395, 455] | 0.4100 |
| nhp | output_split | latent_wishart_attention | full_sequence | 0.9183 | 0.7804 | 3 | [296, 357, 412] | 0.1426 |
| nhp | lal_fixed | no_wishart | prefix_half | 0.4685 | 0.0702 | 2 | [255, 0, 810] | 0.8772 |
| nhp | lal_fixed | no_wishart | full_sequence | 0.4977 | 0.1038 | 3 | [316, 9, 740] | 0.8338 |
| nhp | lal_fixed | latent_wishart_attention | prefix_half | 0.8207 | 0.5564 | 3 | [441, 209, 415] | 0.4719 |
| nhp | lal_fixed | latent_wishart_attention | full_sequence | 0.9239 | 0.7928 | 3 | [407, 289, 369] | 0.1646 |
| thp | output_split | no_wishart | prefix_half | 0.8028 | 0.5119 | 3 | [304, 368, 393] | 0.2913 |
| thp | output_split | no_wishart | full_sequence | 0.8789 | 0.6826 | 3 | [320, 351, 394] | 0.0916 |
| thp | output_split | latent_wishart_attention | prefix_half | 0.8610 | 0.6359 | 3 | [371, 332, 362] | 0.2897 |
| thp | output_split | latent_wishart_attention | full_sequence | 0.9521 | 0.8636 | 3 | [360, 350, 355] | 0.0970 |
| thp | lal_fixed | no_wishart | prefix_half | 0.5136 | 0.1132 | 3 | [328, 379, 358] | 0.6374 |
| thp | lal_fixed | no_wishart | full_sequence | 0.5136 | 0.1062 | 3 | [311, 462, 292] | 0.3241 |
| thp | lal_fixed | latent_wishart_attention | prefix_half | 0.8516 | 0.6108 | 3 | [323, 356, 386] | 0.3250 |
| thp | lal_fixed | latent_wishart_attention | full_sequence | 0.9390 | 0.8273 | 3 | [343, 353, 369] | 0.1171 |
| cotic | output_split | no_wishart | prefix_half | 0.3333 | 0.0000 | 1 | [0, 1065, 0] | 0.0000 |
| cotic | output_split | no_wishart | full_sequence | 0.3333 | 0.0000 | 1 | [0, 1065, 0] | 0.0000 |
| cotic | output_split | latent_wishart_attention | prefix_half | 0.8516 | 0.6160 | 3 | [323, 366, 376] | 0.3492 |
| cotic | output_split | latent_wishart_attention | full_sequence | 0.9446 | 0.8442 | 3 | [365, 345, 355] | 0.1305 |
| cotic | lal_fixed | no_wishart | prefix_half | 0.4075 | 0.0149 | 3 | [528, 109, 428] | 1.0965 |
| cotic | lal_fixed | no_wishart | full_sequence | 0.4000 | 0.0126 | 3 | [536, 108, 421] | 1.0959 |
| cotic | lal_fixed | latent_wishart_attention | prefix_half | 0.7624 | 0.4877 | 3 | [159, 376, 530] | 0.4086 |
| cotic | lal_fixed | latent_wishart_attention | full_sequence | 0.9023 | 0.7471 | 3 | [263, 353, 449] | 0.1898 |

## Stable 256-sample Wishart audit

Every saved Wishart checkpoint was evaluated three more times with 256 fresh
W samples. Suffix-NLL standard deviations are below 0.001 and every repeat
uses all three MAP clusters.

| architecture | parameterization | no-W suffix NLL | W suffix NLL | no-W full purity/ARI | W full purity/ARI |
|---|---|---:|---:|---:|---:|
| NHP | output-split | 3.551045 | 3.460095 | 0.9418 / 0.8351 | 0.9243 / 0.7947 |
| NHP | LaL-fixed | 3.577608 | 3.471363 | 0.4977 / 0.1038 | 0.9243 / 0.7936 |
| THP | output-split | 3.628967 | 3.463962 | 0.8789 / 0.6826 | **0.9509 / 0.8600** |
| THP | LaL-fixed | 3.535851 | 3.482394 | 0.5136 / 0.1062 | 0.9352 / 0.8177 |
| COTIC | output-split | 3.899844 | 3.636450 | 0.3333 / 0.0000 | **0.9427 / 0.8392** |
| COTIC | LaL-fixed | 3.872938 | 3.659475 | 0.4000 / 0.0126 | 0.8923 / 0.7268 |

## Conclusions

The corrected `output_split` model is the preferred Wishart architecture.
It has one encoder and one K*C output head; the Wishart branch adds 225 raw
matrix parameters and no additional encoder. Its pre-head state is exactly
the same for all k by construction and by the saved architecture audit.

Wishart improves suffix NLL for all six paired comparisons. For THP and COTIC
output-split it also improves clustering. COTIC is the sharpest negative
control: no-W collapses every test sequence into one cluster, while Wishart
recovers balanced sizes and full purity/ARI 0.9427/0.8392.

The fixed-K LaL-style alternative has no random walk, but its no-W versions
separate poorly on all three architectures. Wishart rescues them strongly.
Output-split Wishart is nevertheless better than LaL-fixed Wishart in
predictive NLL on NHP, THP, and COTIC, and in clustering on THP/COTIC.
