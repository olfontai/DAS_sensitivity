# DAS_sensitivity
Jupyter Notebook to forward model the amplitude of a source recorded in a DAS array based on 3D sensitivity equations (dip and azimuth) in a ray-based approach.
Code used in paper : 

## workflow
1) Create travel time grid for all sensor using Pykonal
2) Get sensitivity to a source
   
## known difficulties 
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
