# ART_He - A Radiative Transfert code for Helium escape

This 1D NLTE radiative transfer code offers to calculate the metastable He triplet lines profile during a planet's transit, given vertical abundance and velocity profiles of metastable Helium from your favorite hydrodynamic simulation. The code takes into account the Kepler orbit of the planet, allowing to accurately take into account the geometry of the transit and limb-darkening effect.

The model computation is done using a single class: "He_model()", entirely defined in the "art_he.py" module (from art_he import He_model). 
We advise to start by downloading the full repository and launching the "tutorial.ipynb" notebook in jupyter lab for a complete example on how to use the model.
