# tle_conjunctions

Kerry N. Wood (kerry.n.wood@gmail.com)

# A Numpy-vectorized version of the Healy Algorithm

Healy's ephemerides-based, first-order conjunction detection algorithm using SGP4 (thanks Brandon Rhodes) and Numpy.

The algorithm does a first-order approximation of miss-distance at each ephemeris step.  So, runtime is sensitive to step-size. For LEO orbits ~5 minutes is sufficient.  For GEO ~15 is usually sufficient.  Use the minimum of the two.

This implementation takes two catalogs, a start time, end time, and step.  Note that it does no conjunction "refining" when a miss is identified.

This implementation also naively generates ephemeris sets: it will generate time-aligned ephemeris for all objects over the specified window.  **IOW** split runs into shorter time periods so that you don't have huge ephemeris sets.

Both ephemeris generation and conjunction detection is multi-core


### To-do

- allow for externally generated ephemerides
- refine conjunctions; when a miss-distance threshold is identified, refine by propagating within the window


### Original paper:

```
@article{healy1995close,
	title={Close conjunction detection on parallel computer},
	author={Healy, Liam M},
	journal={Journal of Guidance, Control, and Dynamics},
	volume={18},
	number={4},
	pages={824--829},
	year={1995}
}
```
