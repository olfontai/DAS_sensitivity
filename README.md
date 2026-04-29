# DAS_sensitivity
Jupyter Notebook to forward model the amplitude from a point source recorded by a DAS array based on 3D sensitivity equations (dip and azimuth) in a ray-based approach.
Code used in paper : https://doi.org/10.31223/X56J3X

## Content
In this package you can create <u>traveltime grids</u>, model the recorded <u>amplitude</u> in the fiber and compare it to a measured amplitude to create a <u>rescaling factor</u> <br>
Additionnaly, you can also look at the senitivity of a single channel to the space around it. <br>

## workflow sensitivity to a source 
1) Create travel time grid for all sensor using Pykonal<br>
2) Get sensitivity to a source<br>
3) Compare the modeled to the measured amplitude to isolate a correction factor<br>

## workflow sensitivity of  single channel
1) Create travel time grid for all sensor using Pykonal<br>
2) Measure the snsitivity to sources scattered in the space <br>
   
## known difficulties for windows user 
If you work on a windows machine you will have an error when running the solver.solve(): <br>
'Buffer dtype mismatch, expected 'Py_ssize_t' but got 'long''<br>
A solution exists on: https://github.com/malcolmw/pykonal/pull/46<br>
<u>To apply it you have to: </u><br>
1 downlad Pykonal from source : https://github.com/malcolmw/pykonal<br>
    and unzip in.<br>
2 in it you will find: pykonal-master\pykonal-master\pykonal\heapq.pyx <br>
    Change line 36  from :  self.cy_heap_index = np.full(values.shape, fill_value=-1)<br>
                    to   :  self.cy_heap_index = np.full(values.shape, fill_value=-1, dtype=np.intp)<br>
3 reinstall Pykonal from the main folder containing the setup.py: <br>
    In anaconda move to the folder containing the new pykonal (cd path)<br>
    run pip install .<br>
4 Restart your kernel and it should work<br>
