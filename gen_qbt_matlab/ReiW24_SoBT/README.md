Code, Data and Results for Numerical Experiments in "Data-driven 
balanced truncation for second-order systems with generalized proportional 
damping"
===========================================================================

This archive contains the companion codes, data and computed results for 
the paper:

S. Reiter, S. W. R. Werner; "Data-driven 
 balanced truncation for second-order systems with generalized proportional 
 damping"

which implement numerical experiments using intrusive and data-driven 
(non-intrusive) methods for the structured and unstructured surrogate 
modeling of second-order linear dynamical systems. 


## Dependencies and Installation

The code was tested on a on a MacBook Air with 8GB of RAM and an Apple M2 
processor running macOS Ventura version 13.4 with MATLAB 23.2.0.2515942 
(R2023b) Update 7.
The MORLAB—The Model Order Reduction LABoratory version 6.0 is used to 
compute the intrusive reduced-order models via first-order and second-order
position-velocity balanced truncation.


## Getting Started

The `runme*.m` files can be used to reproduce experiments from Section 6 of 
the companion paper. The scripts correspond to the following experiments:

* `runme_sobutterfly`: experiments involving the butterfly gyroscope 
  benchmark
* `runme_soplate`: experiments involving the plate with tuned vibration 
  absorbers
* `runme_somsd`: experiments involving the mass-spring-damper network

The results computed by these scripts will be saved to the `results`
folder. Existing results will be overwritten.


## Author

Sean Reiter
* affiliation: Virginia Tech (USA)
* email: seanr7@vt.edu
* orcid: [0000-0002-7510-1530](https://orcid.org/0000-0002-7510-1530)

Steffen W. R. Werner
* affiliation: Virginia Tech (USA)
* email: steffen.werner@vt.edu
* orcid: [0000-0003-1667-4862](https://orcid.org/0000-0003-1667-4862)


## License

Copyright (C) 2025 Sean Reiter, Steffen W. R. Werner

In general, this software is licensed under the BSD-2 License.
See [COPYING](COPYING) for a copy of the license.

The files in `results` are licensed under the CC-BY 4.0 License.
See [COPYING_DATA](COPYING_DATA) for a copy of the license.

## Citation


### DOI

The DOI for version 1.0 is
[10.5281/zenodo.1157038](https://doi.org/10.5281/zenodo.1157038).


### Cite as

S. Reiter and S. W. R. Werner. Code, data and results for numerical 
experiments in "Data-driven balanced truncation for second-order systems 
with generalized proportional damping" (version 1.0), June 2025. 
doi:10.5281/zenodo.1157038


### BibTeX

```BibTeX
@MISC{supReiW25,
  author =       {Reiter, S. and Werner, S.~W.~R.},
  title  =       {Code, Data and Results for Numerical Experiments in
                  ``{D}ata-driven balanced truncation for second-order systems
                  with generalized proportional damping'' (version 1.0)},
  month  =       jun,
  year   =       {2025},
  doi    =       {10.5281/zenodo.1157038}
}
````
