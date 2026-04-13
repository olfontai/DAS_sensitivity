import numpy as np
import pandas as pd

def get_ray_length (ray):
    """_summary_

    Args:
        ray (_type_): _description_
    """
    new_ray = ray.T
    differences = np.diff(new_ray, axis=1)
    segment_lengths = np.linalg.norm(differences, axis=0)
    ray_length = np.sum(segment_lengths)
    return ray_length 


def moment_tensor_radiation_FF (wave_type, moment_tensor, azimuth, dip): 
    """Give the amplitude (radiation patern) for a specific d

    Args:
        moment_tensor (_type_):3x3 matrix with the moment tensor from the source
        azimuth (_type_): departing ray's azimuth (in degree), North = 0 turning in trigonometric direction
        colatitude (_type_): departing ray's dip from the Z axis (co_latitude) in degrees
    """
    azimuth = np.radians(azimuth)
    dip = np.radians(dip)
    nx = np.cos(dip) * np.cos(azimuth)
    ny = np.cos(dip) * np.sin(azimuth)
    nz = np.sin(dip)
    # unit vector in the ray's direction
    n = np.stack([nx, ny, nz], axis=-1)
    # FF_A is the Far Field Amplitude (= radiation patern)
    if wave_type == 'P':
        FF_A = np.abs(np.einsum('ij,...i,...j->...', moment_tensor, n, n))
    if wave_type == 'SV':
        # unit vector in the plan with Z and n but orthogonal to n (SV polarisation vector or vertical transverse)
        t_sv = np.stack([-np.sin(dip) * np.cos(azimuth), -np.sin(dip) * np.sin(azimuth), np.cos(dip)], axis=-1)
        FF_A = np.abs(np.einsum('ij,...i,...j->...', moment_tensor, t_sv, n))
    if wave_type == 'SH':
        # unit vector orthogonal to n and t_sv (SH polarisation vector)
        t_sh = np.stack([np.sin(azimuth), -np.cos(azimuth), np.zeros_like(azimuth)], axis=-1)
        FF_A = np.abs(np.einsum('ij,...i,...j->...', moment_tensor, t_sh, n))
    return FF_A
def single_force_radiation_FF (wave_type, force, azimuth, dip): 
    """_summary_

    Args:
        force (_type_): 3x1 vector with x, y and Z component of the force vector
        azimuth (_type_): departing ray's azimuth (in degree), North = 0 turning in trigonometric direction
        colatitude (_type_): departing ray's dip from the Z axis (co_latitude) in degrees
    """
    azimuth = np.radians(azimuth)
    dip = np.radians(dip)
    # FF_A is the Far Field Amplitude (= radiation patern)
    if wave_type == 'P':
        # Calculate unit vector n in spherical coordinates
        nx = np.cos(dip) * np.cos(azimuth)
        ny = np.cos(dip) * np.sin(azimuth)
        nz = np.sin(dip)
        # unit vector in the ray's direction
        n = np.stack([nx, ny, nz], axis=-1)
        FF_A = np.einsum('i,...i->...', force, n)
    if wave_type == 'SV':
        # unit vector in the plan with Z and n but orthogonal to n (SV polarisation vector or vertical transverse)
        t_sv = np.stack([-np.sin(dip) * np.cos(azimuth), -np.sin(dip) * np.sin(azimuth), np.cos(dip)], axis=-1)
        FF_A = np.einsum('i,...i->...', force, t_sv)
    if wave_type == 'SH':
        # unit vector orthogonal to n and t_sv (SH polarisation vector)
        t_sh = np.stack([np.sin(azimuth), -np.cos(azimuth), np.zeros_like(azimuth)], axis=-1)
        FF_A = np.einsum('i,...i->...', force, t_sh) 
    return FF_A

def get_fiber_sensitivty (wave_type, Azimuth_DAS,azimuth_ray, dip_ray, dip_DAS , GaugeLength =10 , velocity = 400, wavelength = 500):
    """_summary_

    Args:
        wave_type (_type_): _description_
        GaugeLength (_type_): _description_
        velocity (_type_): _description_
        wavelength (_type_): _description_
        Azimuth_DAS (_type_): _description_
        dip_DAS (_type_): _description_
        equation_set (_type_): _description_

    Returns:s
        _type_: _description_
    """

    # test dot_product 
    DAS_az_rad = np.radians(Azimuth_DAS)
    dip_DAS_rad = np.radians(dip_DAS)
    ray_az_rad = np.radians(azimuth_ray)
    ray_dip_rad = np.radians(dip_ray)

    cos_dependency = (np.cos(dip_DAS_rad)*np.cos(ray_dip_rad)*np.cos(DAS_az_rad - ray_az_rad))+(np.sin(dip_DAS_rad)*np.sin(ray_dip_rad))

    if wave_type == 'P' or wave_type == 'p':
      DAS_sensitivity = cos_dependency*np.sin(cos_dependency/2)

    if wave_type =='SH' :
      sin_factor = np.sin(ray_az_rad-DAS_az_rad)*np.cos(dip_DAS_rad)
      DAS_sensitivity = sin_factor*np.sin(cos_dependency/2)

    if wave_type == 'SV' :
      sin_factor = -(np.cos(dip_DAS_rad)*np.sin(ray_dip_rad)*np.cos(DAS_az_rad-ray_az_rad))+(np.sin(dip_DAS_rad)*np.cos(ray_dip_rad))
      DAS_sensitivity = sin_factor*np.sin(cos_dependency/2)

    if wave_type == 's' or wave_type == 'S' : 
      #SV
      sin_factor = -(np.cos(dip_DAS_rad)*np.sin(ray_dip_rad)*np.cos(DAS_az_rad-ray_az_rad))+(np.sin(dip_DAS_rad)*np.cos(ray_dip_rad))
      SV = sin_factor*np.sin(cos_dependency/2)
      #SH
      sin_factor = np.sin(ray_az_rad-DAS_az_rad)*np.cos(dip_DAS_rad)
      SH = sin_factor*np.sin(cos_dependency/2)
      # combined sensitivity
      DAS_sensitivity = np.sqrt(((SV**2)+(SH**2))/2)

    if wave_type == 'Rayleigh': 
            DAS_sensitivity = np.cos(ray_az_rad-DAS_az_rad)*np.sin((GaugeLength*np.cos(ray_az_rad-DAS_az_rad))/2)

    if wave_type == 'Love' : 
            DAS_sensitivity = np.sin(ray_az_rad-DAS_az_rad)*np.sin((GaugeLength*np.cos(ray_az_rad-DAS_az_rad))/2)
            
    return DAS_sensitivity



#########################################
######### DAS channel metadata ##########
#########################################

R_EARTH = 6371.0  # km

def geodetic_to_ecef(lat, lon, depth):
    lat = np.radians(lat)
    lon = np.radians(lon)

    r = R_EARTH - depth

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.array([x, y, z])

def ecef_to_enu(vec, lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)

    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
    ])

    return R @ vec

def compute_azimuth_dip(df):
    az = []
    dip = []

    for i in range(len(df) - 1):
        p1 = geodetic_to_ecef(df.iloc[i].lat, df.iloc[i].lon, df.iloc[i].depth)
        p2 = geodetic_to_ecef(df.iloc[i+1].lat, df.iloc[i+1].lon, df.iloc[i+1].depth)

        vec = p2 - p1

        enu = ecef_to_enu(vec, df.iloc[i].lat, df.iloc[i].lon)
        E, N, U = enu

        azimuth = np.abs(360 - np.degrees(np.arctan2(E, N)) % 360)
        dip_angle = -np.degrees(np.arctan2(-U, np.sqrt(E**2 + N**2)))

        az.append(azimuth)
        dip.append(dip_angle)

    az.append(az[-1])
    dip.append(dip[-1])

    df["azimuth"] = az
    df["dip"] = dip

    return df

def get_ray_length(ray):
    r = ray[:, 0] * 1e3  # km → m
    theta = ray[:, 1]
    phi = ray[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    xyz = np.stack([x, y, z], axis=1)
    return np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)) / 1000  # km

import numpy as np

R_EARTH = 6371000.0  # m, mean Earth radius (change if you want WGS84 ellipsoid)

def latlon_depth_to_ecef(lat_deg, lon_deg, depth_m, R=R_EARTH):
    """Convert lat, lon (deg) and depth (m, positive downward) to ECEF (m)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = R - depth_m  # radius from Earth's center to the point
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])

def ecef_to_enu_vector(d_ecef, lat0_deg, lon0_deg):
    """Rotate ECEF vector difference into local ENU at (lat0, lon0)."""
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)
    slat = np.sin(lat0); clat = np.cos(lat0)
    slon = np.sin(lon0); clon = np.cos(lon0)

    # Rotation matrix (3x3) from ECEF->ENU
    R = np.array([
        [-slon,             clon,              0.0],
        [-slat*clon,       -slat*slon,         clat],
        [ clat*clon,        clat*slon,         slat]
    ])
    enu = R.dot(d_ecef)
    return enu  # [E, N, U]

def azimuth_and_dip(lat1, lon1, depth1, lat2, lon2, depth2, R=R_EARTH):
    p1 = latlon_depth_to_ecef(lat1, lon1, depth1, R=R)
    p2 = latlon_depth_to_ecef(lat2, lon2, depth2, R=R)
    d = p2 - p1
    E, N, U = ecef_to_enu_vector(d, lat1, lon1)
    # Azimuth: clockwise from North
    az_rad = -np.arctan2(E, N)
    az_deg = (np.degrees(az_rad) + 360.0) % 360.0
    # Dip: positive downward. Up is positive upward, so use -U.
    horiz = np.hypot(E, N)
    dip_rad = np.arctan2(-U, horiz)   # -U so positive dip is downward
    dip_deg = -np.degrees(dip_rad)
    return az_deg, dip_deg



