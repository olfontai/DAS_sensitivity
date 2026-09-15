import numpy as np 
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import xarray as xr
import pandas as pd 
import matplotlib.pyplot as plt 
import pykonal 
from pykonal.transformations import geo2sph, sph2geo

def load_velocity_model(
    filepath_velocity,
    depths,
    phase="P",
    smoothing_window=1,
    plot=True,
):
    """
    Load, interpolate, optionally smooth, and plot a 1D velocity model.

    Parameters
    ----------
    filepath_velocity : str or Path
        Path to the CSV velocity model.
    depths : array-like
        Depth values onto which the velocity model is interpolated.
    phase : str
        Velocity phase/column to use (e.g. "P" or "S").
    smoothing_window : int
        Moving-average smoothing window.
        Use 1 for no smoothing.
    plot : bool
        Whether to display the velocity model.

    Returns
    -------
    velocity_layers_interp : pandas.DataFrame
        Interpolated (and optionally smoothed) velocity model.
    """

    # Load velocity model
    velocity_layers = pd.read_csv(
        filepath_velocity,
        index_col=0
    )
    velocity_layers = velocity_layers[[phase]]
    velocity_layers.index.name = "depth"

    # Make sure depths are sorted
    velocity_layers = velocity_layers.sort_index()

    # Interpolate/reindex velocity model onto requested depths
    velocity_layers_interp = velocity_layers.reindex(
        depths,
        method="ffill"
    )

    velocity_layers_interp.index.name = "depth"

    # Optional smoothing
    if smoothing_window > 1:

        # Pad the edges to avoid losing values
        padded_data = np.pad(
            velocity_layers_interp[phase].values,
            pad_width=smoothing_window // 2,
            mode="edge"
        )

        kernel = np.ones(smoothing_window) / smoothing_window

        smoothed_data = np.convolve(
            padded_data,
            kernel,
            mode="valid"
        )

        # Ensure the output has exactly the same length
        smoothed_data = smoothed_data[:len(velocity_layers_interp)]

        velocity_layers_interp[phase] = smoothed_data

    # Plot
    if plot:

        fig, ax = plt.subplots(figsize=(7.5, 6))

        # Original model
        velocity_layers[phase].plot(
            drawstyle="steps-post",
            ax=ax,
            label="Original model"
        )

        # Interpolated model
        velocity_layers_interp.sort_index()[phase].plot(
            drawstyle="steps-post",
            xlabel="Depth (km)",
            ylabel="Speed (km/s)",
            title=f"1D {phase}-wave velocity model",
            ax=ax,
            grid=True,
            marker="s",
            ls="",
            label="Interpolated model"
        )

        # Highlight interpolation range
        ax.axvspan(
            np.min(depths),
            np.max(depths),
            alpha=0.2
        )

        ax.legend()
        plt.show()

    return velocity_layers_interp

def create_ttitme(source, velocities_df, latitudes, longitudes): 
    
    velocities = velocities_df.stack().to_xarray()
    #velocities = velocities.rename({"level_1": "phase"})
    velocities = velocities.sel(level_1 =velocities_df.columns[0])
    velocity_model = velocities.expand_dims(latitude=latitudes, longitude=longitudes)
    # velocity_model = spatial_lowpass (velocity_model, sigma = 1000)

    ################ Put model back in good orientation ##################

    ref_lon = 'min'
    ref_lat = 'max'
    ref_depth = 'max'

    # Ensure coordinates are in descending order
    if velocity_model.latitude.values[0] > velocity_model.latitude.values[-1] and ref_lat == 'min' :
        velocity_model = velocity_model.sortby("latitude", ascending=True)
    if velocity_model.latitude.values[0] < velocity_model.latitude.values[-1] and ref_lat == 'max' :
        velocity_model = velocity_model.sortby("latitude", ascending=False)

    if velocity_model.longitude.values[0] > velocity_model.longitude.values[-1] and ref_lon == 'min' :
        velocity_model = velocity_model.sortby("longitude", ascending=True)
    if velocity_model.longitude.values[0] < velocity_model.longitude.values[-1] and ref_lon == 'max' :
        velocity_model = velocity_model.sortby("longitude", ascending=False)

    if velocity_model.depth.values[0] > velocity_model.depth.values[-1] and ref_depth == 'min' :
        velocity_model = velocity_model.sortby("depth", ascending=True)
    if velocity_model.depth.values[0] < velocity_model.depth.values[-1] and ref_depth == 'max' :
        velocity_model = velocity_model.sortby("depth", ascending=False)

    ############### CREATE MODEL REFERENCE ###################
    
    latitudes = velocity_model.coords['latitude'].values
    longitudes = velocity_model.coords['longitude'].values
    depths = velocity_model.coords['depth'].values
    reference_point = geo2sph((latitudes.max(), longitudes.min(), depths.max()))

    node_intervals = (
        np.abs(depths[1] - depths[0]),
        np.deg2rad(np.abs(latitudes[1] - latitudes[0])),
        np.deg2rad(np.abs(longitudes[1] - longitudes[0])))

    # transpose velocity model to go from lat lon depth to depth lat lon
    velocities = velocity_model.transpose( 'depth','latitude', 'longitude').copy()

    ### Create and run model ###
    solver = pykonal.solver.PointSourceSolver(coord_sys="spherical")
    solver.velocity.min_coords = reference_point
    solver.velocity.node_intervals = node_intervals
    solver.velocity.npts = velocities.values.shape
    solver.velocity.values = velocities.values
    # Initialize the source location with a random location within the
    # computational grid.
    
    ############# Verify source is in the space ####################

    source_coord = np.array(geo2sph(source.location.values).squeeze())
    max_coord = reference_point + (np.array(node_intervals)*velocities.values.shape)
    is_inside = np.all((source_coord >= reference_point) & (source_coord <= max_coord))
    if is_inside:
        pass
    else:
        print('error for channel: str(Station_Name)')
        dimensions = np.array(['depth', 'latitude', 'longitude'])
        for i, (s, mn, mx) in enumerate(zip(source_coord, reference_point, max_coord)):
            if not (mn <= s <= mx):
                print(f"Dimension {dimensions[i]}: {s:.6f} not in [{mn:.6f}, {mx:.6f}]")

    solver.src_loc = np.array(geo2sph(source.location.values).squeeze())

    ################## Compute traveltimes #####################

    solver.solve()

    tt = solver.tt.values
    tt[np.isinf(tt)] = np.nan
    travel_times = velocities.copy()
    travel_times.values  = tt

    travel_times.attrs['node_intervals'] = node_intervals
    travel_times.attrs['reference_point'] = reference_point


    return travel_times

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