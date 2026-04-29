# DAS_sensitivity
Jupyter Notebook to forward model the amplitude from a point source recorded by a DAS array based on 3D sensitivity equations (dip and azimuth) in a ray-based approach.<br>
This code was developed for the paper : Understanding fiber-optic sensitivity to a wavefield: A framework to separate site amplification from orientation effects (preprint: https://doi.org/10.31223/X56J3X)

## Content
In this package you can create <u>traveltime grids</u>, model the recorded <u>amplitude</u> in the fiber and compare it to a measured amplitude to create a <u>rescaling factor</u> <br>
Additionnaly, you can also look at the senitivity of a single channel to the space around it and the impact of the velocity structure. <br>

## workflow sensitivity of the full fiber to a source 
1) Create travel time grid for all sensor using Pykonal<br>
2) Get sensitivity to a source<br>
3) Compare the modeled to the measured amplitude to isolate a correction factor<br>

## workflow sensitivity around a single channel
1) Create travel time grid for all sensor using Pykonal<br>
2) Measure the sensitivity to sources scattered in the space <br>
   
## known difficulties for windows user 
If you work on a windows machine you will have an error when running the solver.solve(): <br>
'Buffer dtype mismatch, expected 'Py_ssize_t' but got 'long''<br>
A solution exists on: https://github.com/malcolmw/pykonal/pull/46<br>
<u>To apply it you have to: </u><br>
1) downlad Pykonal from source : https://github.com/malcolmw/pykonal<br>
    and unzip in.<br>
2) in it you will find: pykonal-master\pykonal-master\pykonal\heapq.pyx <br>
    Change line 36  from :  self.cy_heap_index = np.full(values.shape, fill_value=-1)<br>
                    to   :  self.cy_heap_index = np.full(values.shape, fill_value=-1, dtype=np.intp)<br>
3) reinstall Pykonal from the main folder containing the setup.py: <br>
    In anaconda move to the folder containing the new pykonal (cd path)<br>
    run pip install .<br>
4) Restart your kernel and it should work<br>

# DAS_sensitivity

Jupyter Notebook package to forward-model the amplitude recorded by a DAS (Distributed Acoustic Sensing) array from a point source using **3D sensitivity equations** (dip and azimuth) in a ray-based approach.

This code was developed for the paper:  
**Understanding fiber-optic sensitivity to a wavefield: A framework to separate site amplification from orientation effects**  
Preprint: https://doi.org/10.31223/X56J3X

---

## Overview

This package allows you to:

- create **travel-time grids**
- model the recorded **amplitude** along the fiber
- compare modeled and measured amplitudes
- estimate a **rescaling factor** to isolate site amplification effects

Additionally, you can:

- analyze the **sensitivity of a single channel** to the surrounding space
- evaluate the impact of the **velocity structure** on DAS sensitivity

---

## Workflow — Full Fiber Sensitivity to a Source

### 1. Create travel-time grids

Generate travel-time grids for all sensors using **Pykonal**

### 2. Compute sensitivity to a source

Model the DAS response using the 3D sensitivity equations

### 3. Compare modeled and measured amplitudes

Use the comparison to isolate a correction (rescaling) factor

---

## Workflow — Sensitivity Around a Single Channel

### 1. Create travel-time grids

Generate travel-time grids for all sensors using **Pykonal**

### 2. Measure sensitivity in the surrounding space

Evaluate the sensitivity to sources distributed around the channel

---

## Known Issue for Windows Users

When running on a Windows machine, you may encounter the following error when executing:

```python
solver.solve()
Error Message
Buffer dtype mismatch, expected 'Py_ssize_t' but got 'long'
Cause

This is a known issue with Pykonal on Windows systems.

A fix has been proposed here:
https://github.com/malcolmw/pykonal/pull/46

Solution

To apply the fix manually, follow these steps:

1. Download Pykonal from Source

Download the source code from:

https://github.com/malcolmw/pykonal

Then extract (unzip) the downloaded folder.

2. Modify the heapq.pyx File

Locate the following file:

pykonal-master/pykonal-master/pykonal/heapq.pyx

Find line 36 and change:

self.cy_heap_index = np.full(values.shape, fill_value=-1)

to:

self.cy_heap_index = np.full(
    values.shape,
    fill_value=-1,
    dtype=np.intp
)
3. Reinstall Pykonal

From the main folder containing setup.py:

Open Anaconda Prompt
Navigate to the folder:
cd path/to/pykonal
Run:
pip install .
4. Restart Your Kernel

After installation is complete, restart your Python kernel.

The issue should now be resolved.

