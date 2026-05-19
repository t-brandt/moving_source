# moving_source
Model and fit the track of a moving point source in up-the-ramp data.

This package is not yet `pip` installable, but may be cloned and used locally after running
`python setup.py build_ext --inplace`
in a terminal window.  The demonstration notebook shows the basic functionality.  A sample effective point-spread function (ePSF) from the Roman Space Telescope's Wide Field Instrument is provided as an example, and is used by the demonstration notebook.

Requirements: `astropy`, `Cython`, `numpy`, `scipy`, `matplotlib`

In the future this software may be incorporated into other packages for modeling and fitting the tracks of moving objects.

If you make use of this software in your research, please cite https://arxiv.org/abs/placeholder.
