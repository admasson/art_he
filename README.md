# ART_He - A Radiative Transfert code for Helium escape

This 1D NLTE radiative transfer code offers to calculate the metastable He triplet lines profile during a planet's transit, given vertical abundance and velocity profiles of metastable Helium from your favorite hydrodynamic simulation. The code takes into account the Kepler orbit of the planet, allowing to accurately take into account the geometry of the transit and limb-darkening effect.

The model computation is done using a single class: "He_model()", entirely defined in the "art_he.py" module (from art_he import He_model). 
We advise to start by downloading the full repository and launching the "tutorial.ipynb" notebook in jupyter lab for a complete example of how to use the model.

The "Example_profiles/" folder contains an example of an abundance (Abund_HD209_ratio90_T_10000_MLR_2p37e11.txt) and a velocity profile (Velocity_HD209_ratio90_T_10000_MLR_2p37e11.txt) of the metastable He escapes, in the format used by the module. They have been computed for HD 209458 b, using a 1D spherically symmetric hydrodynamic code based on the Parker wind equations (Lampon et al. 2020), fixing the H/He ratio to a solar (90/10) value.

Acknowledgments: this code has been developed in collaboration with M. López-Puertas and M. Lampón from Instituto de Astrofísica de Andalucía (IAA-CSIC), Glorieta de la Astronomía s/n, 18008 Granada, Spain, who developed the original IDL code this module is based on.
