# DAS_sensitivity
Jupyter Notebook to assess DAS sensitivity to seismic waves, based on 3D sensitivity equations (dip and azimuth) in a ray-based approach.
Code used in paper : 

## STEPS
1) Create travel time grid for all sensor using Pykonal
2) Get sensitivity to a source
   
## known difficulties 
If you work on a windows machine you will have an error when running the solver.solve(): 
- Buffer dtype mismatch, expected 'Py_ssize_t' but got 'long'
A solution exists on: https://github.com/malcolmw/pykonal/pull/46
To apply it you have to: 
1 downlad Pykonal from source : https://github.com/malcolmw/pykonal
    and unzip in.
2 in it you will find: pykonal-master\pykonal-master\pykonal\heapq.pyx 
    Change line 36  from :  self.cy_heap_index = np.full(values.shape, fill_value=-1)
                    to   :  self.cy_heap_index = np.full(values.shape, fill_value=-1, dtype=np.intp)
3 reinstall Pykonal from the main folder containing the setup.py: 
    In anaconda move to the folder containing the new pykonal (cd path)
    run pip install .
4 Restart your kernel and it should work
