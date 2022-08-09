# GNSS_Refractometry_SWE

Snow water equivalent estimation (SWE) based on GNSS (Global Navigation Satellite System) refractometry using the biased up-component and post processing in RTKLIB.


The biased up-component of a short GNSS baseline between a base antenna (mounted on a pole) and a rover antenna (mounted underneath the snowpack) is used in this approach. High-end receivers are used in a field setup, connected to high-end multi-frequency and multi-GNSS antennas. The receivers logged multi-GNSS RINEX data with 30s sampling rate, which are used for post processing using the open-source GNSS processing software RTKLIB. 


The python script contains a workflow from post processing of daily GNSS RINEX files to filtered and plotted SWE timeseries. A RTKLIB configuration file is attached, which is used in Steiner et al. (2022).


The method follows Steiner et al. (2022, 2020): 

Steiner, L.; Studemann, G.; Grimm, D.; Marty, C.; Leinss, S. (Near) Real-Time Snow Water Equivalent Observation Using GNSS Refractometry and RTKLib. 2022, submitted to Sensors.

L. Steiner, M. Meindl, C. Marty and A. Geiger, "Impact of GPS Processing on the Estimation of Snow Water Equivalent Using Refracted GPS Signals," in IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 1, pp. 123-135, Jan. 2020, doi: 10.1109/TGRS.2019.2934016.


Example data is publicly available on:

Steiner, L. GNSS refractometry data from Davos Weissfluhjoch, Switzerland in 2016/17. Zenodo, 2022, embargoed until September 2022, doi:10.5281/zenodo.6514932
