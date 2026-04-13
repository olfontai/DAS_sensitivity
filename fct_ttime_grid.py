import numpy as np 
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import xarray as xr

def fill_nans_with_nearest(data):
    """
    Fill NaNs in a NumPy array with the nearest non-NaN value.
    Works with 2D or 3D data.
    """
    mask = np.isnan(data)
    if not np.any(mask):
        return data  # No NaNs to fill

    # Get indices of nearest non-NaN neighbors
    distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)

    # Use the indices to index into original data
    filled = data[tuple(indices)]
    return filled

def spatial_lowpass (velocity_grid, sigma):
    # Apply to your xarray.DataArray
    velocity_grid = xr.DataArray(
        fill_nans_with_nearest(velocity_grid.values),
        dims=velocity_grid.dims,
        coords=velocity_grid.coords,
        attrs=velocity_grid.attrs
    )
    

    # Example: velocity is an xarray.DataArray with dims ('z', 'y', 'x')
    # Step 1: Get grid spacing
    dz = np.abs(np.diff(velocity_grid.coords['depth'].values)).mean()*1000 # Km to m
    dy = np.abs(np.diff(velocity_grid.coords['latitude'].values).mean())*111320 # ° lat to m
    dx = np.abs(np.diff(velocity_grid.coords['longitude'].values).mean())*111320*np.cos(np.radians(velocity_grid.coords['latitude'].values[0])) # ° lon to m
    print('Grid Z spacing: '+str(dz))
    print('Grid latitude spacing: '+str(dy))
    print('Grid longitude spacing: '+str(dx))

    # Step 2: Define Fresnel radius and convert to standard deviation in grid points
    sigma_z = sigma / dz
    sigma_y = sigma / dy
    sigma_x = sigma / dx

    # Step 3: Apply Gaussian filter
    return  xr.DataArray(
        gaussian_filter(velocity_grid.values, sigma=(sigma_z, sigma_y, sigma_x), mode='nearest'),
        dims=velocity_grid.dims,
        coords=velocity_grid.coords,
        attrs=velocity_grid.attrs
    )
    
def resampel_grid (ttime_grid, new_lat, new_lon, new_depth): 
    """_summary_

    Args:
        ttime_grid (xarray): _description_
        new_lat (np.array): np.linspace(depth min, depth_max, depth resolution)
        new_lon (np.array): np.linspace(lon min, lon_max, lon resolution)
        new_depth (np.array): np.linspace(lat min, lat max, lat resolution)
    """
    # Build the target grid
    target_grid = {
        "latitude": new_lat,
        "longitude": new_lon,
        "depth": new_depth
    }

    # Interpolate onto the target grid
    return ttime_grid.interp(target_grid)