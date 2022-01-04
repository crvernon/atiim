Getting started
===============

About
-----

The hydrodynamics of floodplain and/or tidal-fluvial systems directly impact habitat quality and opportunity. Inundation patterns can be influenced by tidal cycles, hydro system operations, meteorology and climate, and physical barriers such as dikes and tide gates. These patterns are controlling factors in the development of physical and biological structure, including fluxes through the terrestrial and aquatic systems. Ongoing wetland/riparian restoration efforts are intended to increase available habitat opportunity through hydrologic reconnection between main stem river channels and diked areas of the historical river floodplain. The habitat opportunity can be evaluated by quantifying wetted area, frequency, and duration of inundation.

To address the challenge of rapidly characterizing spatiotemporally-complex inundation patterns in dynamic systems, such as estuarine tidal-fluvial environments, the Pacific Northwest National Laboratory (PNNL) developed the Area-Time Inundation Index Model (ATIIM). Originally developed as part of the U.S. Army Corps of Engineers (Portland District) sponsored “Cumulative Effects” project (EST-P-02-04), the ATIIM integrates in-situ, modeled, or scenario-based hourly water-surface elevation (WSE) data and advanced terrain processing of high-resolution terrestrial and bathymetric elevation data (Figure 1). The model establishes and describes patterns in the spatial and temporal relationships of the land and water including non-dimensional area-time and volume-time inundation indices. The ATIIM uses a spatially-based region-growing wetted-area algorithm that enforces hydrologic connectivity and determines numerous metrics such as site average bankfull elevation, two- and three-dimensional inundation extent/volume, and habitat opportunity. Importantly, it is well suited to represent numerous types of channel network morphologies that represent multiple inlets and outlets, flow-through, and multi-directional flow, while preserving the landscape microtopography, a critical factor in low-lying estuarine environments. The development of microtopographic terrain model is particularly useful in low elevation range estuarine environments and can be used to extract a fine level of detail in the inundation patterns and also reveal high resolution and complex channel hydrography. Hydrological process metrics such as inundation frequency, duration, maximum area, and maximum frequency area are also calculated and can inform evaluation of proposed restoration sites; e.g., determine relationships between WSE and habitat opportunity, contrast alternative restoration designs, terrain modifications, predict impacts of altered flow regimes including sea-level rise and/or extreme event scenarios, and estimate nutrient and biomass fluxes. In an adaptive management framework, this model can be used to provide standardized site comparisons and effectiveness monitoring indicators of changes in the developmental trajectories of restoration sites. In an adaptive management framework, this model can be used to provide standardized site comparisons and effectiveness monitoring of changes in the developmental trajectories of restoration sites.

The original version of ATIIM was developed with ESRI’s Arc Macro Language (AML) based on Arc/INFO v.8.x Under this recent effort, the ATIIM code was ported to Python and currently operates via the common and familiar ArcGIS software suite through ArcMap. The majority of ported model functions are now self-reliant within ATIIM or call outside open-source codes and don’t rely heavily on embedded functions within ArcGIS. As such, there are several other software packages required for ATIIM to function and these are bundled within the install. A development goal for ATIIM was to allow use of the model using a basic license version of ArcGIS and therefore not require users to have expensive extensions to license. This goal also enables an eventual goal to allow ATIIM to be supported within other GIS softwares, including open-source QGIS. This goal is not fully realized at this version, as there are still functions that rely on the Spatial Analyst license, but continued development will achieve the intended goal. One of the main objectives in the development of ATIIM has been to provide a cost-effective, easy-to-use, rapid assessment tool suitable for the desktop planning environment that represents an advance over methods that estimate inundation but do not enforce hydrological connectivity.

The ATIIM outputs a wide suite of metrics over a spatial and temporal continuum, taking the form of spatial data, tables, metrics, and plots. The modeling captures continuous spatial and temporal effects at user-defined time-steps (typically hourly increments) over the study period, thus snapshots of inundation conditions are available for any particular time of interest. A series of metrics are produced and can be used for a variety of situations to best characterize inundation events as they affect habitat quality and availability for local species. The development of ATIIM has relied on, and has been guided by, feedback from practitioners and their respective needs. New metrics, data processing steps, and data sets can be added based on your feedback.
There are three classes of data output produced by ATIIM: 1) spatial data including raster and vector representations of the site under different flow states and restoration designs (Figure 2); 2) tabular data providing site characteristics and metrics; and 3) graph data derived from the analysis and post-processing of the spatial data (Figure 3).


Python version support
----------------------

Officially Python 3.7, 3.8, and 3.9


Installation
------------

**atiim** can be installed via pip by running the following from a terminal window::

    pip install atiim

Conda/Miniconda users can utilize the ``environment.yml`` stored in the root of this repository by executing the following from a terminal window::

    conda env create --file environment.yml

It may be favorable to the user to create a virtual environment for the **atiim** package to minimize package version conflicts.  See `creating virtual environments <https://docs.python.org/3/library/venv.html>`_ to learn how these function and can be setup.

Installing package data
-----------------------

**atiim** requires package data to be installed from Zenodo to keep the package lightweight.  After **atiim** has been installed, run the following from a Python prompt:

**NOTE**:  The package data will require approximately 195 MB of storage.

.. code-block:: python

    import atiim

    atiim.install_package_data()

This will automatically download and install the package data necessary to run the examples in accordance with the version of **atiim** you are running.  You can pass an alternative directory to install the data into (default is to install it in the package data directory) using ``data_dir``.  When doing so, you must modify the configuration file to point to your custom paths.


Dependencies
------------

=============   ================
Dependency      Minimum Version
=============   ================
numpy           1.19.4
pandas          1.1.4
rasterio        1.2.3
requests        2.25.1
joblib          1.0.1
matplotlib      3.3.3
seaborn         0.11.1
whitebox        1.5.1
fiona           1.8.19
pyproj          3.0.1
rtree           0.9.7
shapely         1.7.1
geopandas       0.9.0
=============   ================


Optional dependencies
---------------------

==================    ================
Dependency            Minimum Version
==================    ================
build                 0.5.1
nbsphinx              0.8.6
setuptools            57.0.0
sphinx                4.0.2
sphinx-panels         0.6.0
sphinx-rtd-theme      0.5.2
twine                 3.4.1
pytest                6.2.4
pytest-cov            2.12.1
==================    ================
