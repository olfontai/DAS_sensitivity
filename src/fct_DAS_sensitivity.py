import numpy as np
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from pykonal.transformations import geo2sph, sph2geo


class Location:
    def __init__(self, latitude, longitude, depth=0):
        if not (-90 <= latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180")

        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth

    def __repr__(self):
        return (f"Location(lat={self.latitude}, lon={self.longitude}, "
                f"elev={self.elevation}, depth={self.depth})")
    @property
    def values(self):
        return np.array([self.latitude, self.longitude, self.depth])



class SourceMechanism(ABC):
    """
    Abstract base class for seismic source mechanisms.

    A source mechanism defines how energy is released at the source,
    e.g., as a force, a moment tensor, or an isotropic explosion.
    """

    @abstractmethod
    def describe(self):
        """Return a human-readable description of the mechanism."""
        pass

@dataclass


class SingleForce(SourceMechanism):
    """
    Represents a single force source.

    This models a force acting in a specific direction (fx, fy, fz),
    often used for controlled sources (e.g., vibroseis trucks).
    """

    fx: float
    fy: float
    fz: float

    def as_vector(self):
        """Return the force as a 3D NumPy vector."""
        return np.array([self.fx, self.fy, self.fz])

    def describe(self):
        return f"Single force: ({self.fx}, {self.fy}, {self.fz})"
    
@dataclass
class MomentTensor(SourceMechanism):
    """
    Represents a seismic moment tensor source.

    The moment tensor describes general seismic sources, including:
    - double-couple (earthquakes)
    - CLVD
    - isotropic components

    Stored in symmetric form:
        [mxx mxy mxz]
        [mxy myy myz]
        [mxz myz mzz]
    """

    mxx: float
    myy: float
    mzz: float
    mxy: float
    mxz: float
    myz: float

    def as_matrix(self):
        """Return the 3x3 moment tensor matrix."""
        return np.array([
            [self.mxx, self.mxy, self.mxz],
            [self.mxy, self.myy, self.myz],
            [self.mxz, self.myz, self.mzz]
        ])

    def describe(self):
        return "Moment tensor source"
@dataclass
class IsotropicSource(SourceMechanism):
    """
    Represents an isotropic source (explosion or implosion).

    This produces:
    - P-waves equally in all directions (spherical radiation)
    - No S-waves
    """

    magnitude: float  # scalar strength

    def describe(self):
        return f"Isotropic source (explosion/implosion), magnitude={self.magnitude}"
class NoSource(SourceMechanism):
    """
    Represents the absence of a source mechanism.
    Useful as a placeholder or default.
    """

    def describe(self):
        return "No source mechanism defined"
class SeismicSource:
    """
    Represents a seismic source with:
    - a name
    - a spatial location
    - a source mechanism

    Provides methods to:
    - describe the source
    - compute radiation amplitudes
    - visualize radiation patterns
    """

    def __init__(self, name, location, mechanism=None):
        self.name = name
        self.location = location
        self.mechanism = mechanism
    
    def describe(self):
        """Return a readable description of the source."""
        mech_desc = self.mechanism.describe() if self.mechanism else "No mechanism"
        return f"{self.name} at {self.location} → {mech_desc}"
    
    def radiation_amplitude(self, azimuth, dip, wavetype, degrees=True):
        """
        Compute radiation amplitude for a single wave type.

        Parameters
        ----------
        azimuth : float or array
        dip : float or array
        wavetype : str
            "P", "SV", "SH", or "S"
        degrees : bool
            If True, inputs are in degrees

        Returns
        -------
        array or float
            منتخب amplitude
        """

        if self.mechanism is None:
            raise ValueError("No mechanism defined")

        azimuth = np.asarray(azimuth)
        dip = np.asarray(dip)

        if degrees:
            azimuth = np.radians(azimuth)
            dip = np.radians(dip)

        # --- Direction vectors ---
        nx = np.sin(dip) * np.cos(azimuth)
        ny = np.sin(dip) * np.sin(azimuth)
        nz = np.cos(dip)

        n = np.stack([nx, ny, nz], axis=-1)

        t_sv = np.stack([
            np.cos(dip) * np.cos(azimuth),
            np.cos(dip) * np.sin(azimuth),
            -np.sin(dip)
        ], axis=-1)

        t_sh = np.stack([
            -np.sin(azimuth),
            np.cos(azimuth),
            np.zeros_like(azimuth)
        ], axis=-1)

        # --- Mechanism handling ---
        if isinstance(self.mechanism, MomentTensor):
            M = self.mechanism.as_matrix()

            A_P  = np.einsum('...i,ij,...j->...', n, M, n)
            A_SV = np.einsum('...i,ij,...j->...', t_sv, M, n)
            A_SH = np.einsum('...i,ij,...j->...', t_sh, M, n)

        elif isinstance(self.mechanism, SingleForce):
            F = self.mechanism.as_vector()

            A_P  = np.einsum('...i,i->...', n, F)
            A_SV = np.einsum('...i,i->...', t_sv, F)
            A_SH = np.einsum('...i,i->...', t_sh, F)

        elif isinstance(self.mechanism, IsotropicSource):
            A_P  = np.ones_like(nx) * self.mechanism.magnitude
            A_SV = np.ones_like(nx) * self.mechanism.magnitude
            A_SH = np.ones_like(nx) * self.mechanism.magnitude

        else:
            raise TypeError("Unknown mechanism type")

        # --- Return only requested wavetype ---
        if wavetype == "P":
            return A_P
        elif wavetype == "SV":
            return A_SV
        elif wavetype == "SH":
            return A_SH
        elif wavetype == "S":
            return np.sqrt(A_SV**2 + A_SH**2)
        else:
            raise ValueError("wavetype must be 'P', 'SV', 'SH', or 'S'")
    
    def plot_radiation_pattern(self, resolution=100):
        """
        Plot 3D radiation patterns for P, SV, SH, and total S waves.

        Parameters
        ----------
        resolution : int
            Number of angular samples (higher = smoother plot)

        Notes
        -----
        - Uses spherical sampling of directions
        - Colors represent signed amplitude
        - Geometry shows radiation lobes
        """
        if self.mechanism is None:
            raise ValueError("No mechanism defined")

        theta = np.linspace(0, np.pi, resolution)
        phi = np.linspace(0, 2 * np.pi, resolution)
        theta, phi = np.meshgrid(theta, phi)

        # Unit vectors
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)
        n = np.stack([nx, ny, nz], axis=-1)

        t_sv = np.stack([
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta)
        ], axis=-1)

        t_sh = np.stack([
            -np.sin(phi),
            np.cos(phi),
            np.zeros_like(phi)
        ], axis=-1)

        # ---- CASE 1: Moment Tensor ----
        if isinstance(self.mechanism, MomentTensor):
            M = np.array(self.mechanism.as_matrix())

            A_P = np.einsum('ij,...i,...j->...', M, n, n)
            A_SV = np.einsum('ij,...i,...j->...', M, t_sv, n)
            A_SH = np.einsum('ij,...i,...j->...', M, t_sh, n)

        # ---- CASE 2: Single Force ----
        elif isinstance(self.mechanism, SingleForce):
            F = np.array(self.mechanism.as_vector())

            A_P = np.einsum('i,...i->...', F, n)
            A_SV = np.einsum('i,...i->...', F, t_sv)
            A_SH = np.einsum('i,...i->...', F, t_sh)

        # ---- CASE 3: Isotropic ----
        elif isinstance(self.mechanism, IsotropicSource):
            A_P = np.ones_like(nx) * self.mechanism.magnitude
            A_SV = A_P
            A_SH = A_P

        else:
            raise TypeError("Unknown mechanism type")

        A_S_total = np.sqrt(A_SV**2 + A_SH**2)

        # Coordinates
        x_P, y_P, z_P = np.abs(A_P) * nx, np.abs(A_P) * ny, np.abs(A_P) * nz
        x_SV, y_SV, z_SV = np.abs(A_SV) * nx, np.abs(A_SV) * ny, np.abs(A_SV) * nz
        x_SH, y_SH, z_SH = np.abs(A_SH) * nx, np.abs(A_SH) * ny, np.abs(A_SH) * nz
        x_S, y_S, z_S = A_S_total * nx, A_S_total * ny, A_S_total * nz

        vmax = np.max([
            np.abs(A_P).max(),
            np.abs(A_SV).max(),
            np.abs(A_SH).max()
        ])

        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = cm.coolwarm

        all_x = np.concatenate([x_P.flatten(), x_SV.flatten(), x_SH.flatten(), x_S.flatten()])
        all_y = np.concatenate([y_P.flatten(), y_SV.flatten(), y_SH.flatten(), y_S.flatten()])
        all_z = np.concatenate([z_P.flatten(), z_SV.flatten(), z_SH.flatten(), z_S.flatten()])

        max_range = np.max([np.abs(all_x).max(), np.abs(all_y).max(), np.abs(all_z).max()])
        limits = [-max_range, max_range]

        fig = plt.figure(figsize=(16, 10))

        def plot(ax, x, y, z, values, title):
            step = 3
            ax.plot_surface(
                x[::step, ::step],
                y[::step, ::step],
                z[::step, ::step],
                facecolors=cmap(norm(values[::step, ::step])),
                edgecolor='k',
                linewidth=0.3,
                alpha=0.9,
                shade=False
            )
            ax.set_title(title)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_zlim(limits)
            ax.set_box_aspect([1, 1, 1])

        ax1 = fig.add_subplot(141, projection='3d')
        plot(ax1, x_P, y_P, z_P, A_P, "P")

        ax2 = fig.add_subplot(142, projection='3d')
        plot(ax2, x_SV, y_SV, z_SV, A_SV, "SV")

        ax3 = fig.add_subplot(143, projection='3d')
        plot(ax3, x_SH, y_SH, z_SH, A_SH, "SH")

        ax4 = fig.add_subplot(144, projection='3d')
        plot(ax4, x_S, y_S, z_S, A_S_total, "S")

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=[ax1, ax2, ax3, ax4], shrink=0.3)

        fig.suptitle(f"{self.name} Radiation Pattern", fontsize=16, y=0.655)
        plt.show()
    def plot_beachball(self, resolution=200):
        """
        Plot a beachball diagram (lower hemisphere projection).

        Only works for MomentTensor sources.

        Parameters
        ----------
        resolution : int
            Number of angular samples
        """

        if not isinstance(self.mechanism, MomentTensor):
            raise TypeError("Beachball plot only valid for MomentTensor sources")

        M = self.mechanism.as_matrix()

        # Grid in spherical coordinates (lower hemisphere only)
        theta = np.linspace(0, np.pi/2, resolution)  # 0 → 90° (downward)
        phi = np.linspace(0, 2*np.pi, resolution)
        theta, phi = np.meshgrid(theta, phi)

        # Direction vectors
        nx = np.sin(theta) * np.cos(phi)
        ny = np.sin(theta) * np.sin(phi)
        nz = np.cos(theta)

        n = np.stack([nx, ny, nz], axis=-1)

        # P-wave radiation
        A_P = np.einsum('...i,ij,...j->...', n, M, n)

        # --- Projection to 2D (equal-area projection) ---
        r = np.sqrt(2) * np.sin(theta / 2)
        x = r * np.sin(phi)
        y = r * np.cos(phi)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(6, 6))

        # Fill compressional / dilatational
        ax.contourf(
            x, y, A_P,
            levels=[-np.inf, 0, np.inf],
            colors=['white', 'black'],
            alpha=0.9
        )

        # Draw nodal lines
        ax.contour(x, y, A_P, levels=[0], colors='k', linewidths=1)

        # Outer circle
        circle = plt.Circle((0, 0), 1, edgecolor='k', facecolor='none', linewidth=1.5)
        ax.add_patch(circle)

        ax.set_aspect('equal')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')

        ax.set_title(f"{self.name} Beachball")

        plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class DASSensor:
    """
    Models Distributed Acoustic Sensing (DAS) fiber response
    and provides plotting utilities.
    """

    def __init__(self, azimuth, dip, gauge_length=10, wavelength=None, velocity=None, latitude = None, longitude = None, depth = None):
        self.azimuth = azimuth
        self.dip = dip
        self.gauge_length = gauge_length
        self.wavelength = wavelength
        self.velocity = velocity
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
    def values(self):
        return np.array([self.latitude, self.longitude, self.depth])
    
    def metadata(self):
        """
        Export all instance attributes as a dictionary.
        """
        return {
            "azimuth": self.azimuth,
            "dip": self.dip,
            "gauge_length": self.gauge_length,
            "wavelength": self.wavelength,
            "velocity": self.velocity,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth": self.depth,
        }


    def sensitivity(self, wave_type, azimuth_ray, dip_ray):
        DAS_az = np.radians(self.azimuth)
        DAS_dip = np.radians(self.dip)
        ray_az = np.radians(azimuth_ray)
        ray_dip = np.radians(dip_ray)

        # Common geometric term
        cos_dep = (
            np.cos(DAS_dip)*np.cos(ray_dip)*np.cos(DAS_az - ray_az)
            + np.sin(DAS_dip)*np.sin(ray_dip)
        )

        # -------------------------------
        # CASE 1: Long wavelength approx
        # -------------------------------
        if self.wavelength is None:
            if wave_type in ['P', 'p']:
                return cos_dep**2 # * np.sin(cos_dep / 2)

            if wave_type == 'SH':
                sin_factor = np.sin(ray_az - DAS_az) * np.cos(DAS_dip)
                return sin_factor * cos_dep #* np.sin(cos_dep / 2)

            if wave_type == 'SV':
                sin_factor = (
                    -np.cos(DAS_dip)*np.sin(ray_dip)*np.cos(DAS_az - ray_az)
                    + np.sin(DAS_dip)*np.cos(ray_dip)
                )
                return sin_factor *cos_dep # np.sin(cos_dep / 2)

            if wave_type in ['S', 's']:
                sv = self.sensitivity('SV', azimuth_ray, dip_ray)
                sh = self.sensitivity('SH', azimuth_ray, dip_ray)
                return np.sqrt((sv**2 + sh**2) / 2)

        # -------------------------------
        # CASE 2: Full wavelength physics
        # -------------------------------
        else:
            k = 2 * np.pi / self.wavelength
            g = self.gauge_length
            c = self.velocity if self.velocity is not None else 1.0
            rescaling = 2/(k*g) # allow to match the values from the point strain sensing

            common_sin = np.sin((k * g / 2) * cos_dep)
            prefactor = (2 * c * k) / g

            if wave_type in ['P', 'p']:
                return  rescaling * cos_dep * common_sin # prefactor

            if wave_type == 'SV':
                sv_term = (
                    -np.cos(DAS_dip)*np.sin(ray_dip)*np.cos(DAS_az - ray_az)
                    + np.sin(DAS_dip)*np.cos(ray_dip)
                )
                return  rescaling * sv_term * common_sin # prefactor

            if wave_type == 'SH':
                sh_term = np.cos(DAS_dip) * np.sin(ray_az - DAS_az)
                return  rescaling * sh_term * common_sin # prefactor

            if wave_type in ['S', 's']:
                sv = self.sensitivity('SV', azimuth_ray, dip_ray)
                sh = self.sensitivity('SH', azimuth_ray, dip_ray)
                return np.sqrt((sv**2 + sh**2) / 2)

    # --------------------- PLOTTING METHODS ---------------------

    def plot_polar_pattern(self, wave_type, cmap=None, title="DAS Polar Pattern"):
        """
        Polar plot of the DAS amplitude pattern.
        
        Parameters
        ----------
        azimuth_grid : 2D array
            Grid of azimuths in degrees, shape matches data
        dip_grid : 2D array
            Grid of dips in degrees, shape matches data
        data : 2D array
            DAS sensitivity or amplitude for each azimuth/dip
        """
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        import numpy as np

        az = np.linspace(0, 360, 100)      # 1D
        dip = np.linspace(-90, 90, 50)       # 1D

        ones = np.ones((len(dip), len(az)))
        az_grid = az * ones
        dip_grid = dip[:, np.newaxis] * ones

        data = self.sensitivity( wave_type, az_grid, dip_grid)
        #data = np.where(np.abs(data)<1e-14, np.nan, data)

        # Normalize data
        data_norm = data 

        figure, ax = plt.subplots(
            1, 1, figsize=(8, 8),
            subplot_kw=dict(projection='polar'), tight_layout=True
        )

        if cmap is None:
            if wave_type in ['P', 'S'] and data.min()>0:
                cmap = 'Reds'

            elif wave_type in ['SV', 'SH'] or wave_type in ['P', 'S'] and data.min()<0:
                cmap = 'seismic'
                
            else:
                cmap = 'cool'

        # Convert angles to radians for plotting
        theta = np.radians(az_grid)
        r = dip_grid  # dip in degrees for radial coordinate

        # Plot
        cs = ax.contourf(theta, r, data_norm, levels=100, cmap=cmap)

        # Ticks and labels
        ax.tick_params(axis='both', labelsize=20)
        ax.set_rticks([])
        angles = np.arange(0, 360, 45)
        ax.set_xticks(np.deg2rad(angles))
        ax.set_xticklabels([f"{a}°" for a in angles], fontsize=20)
        ax.tick_params(axis='x', pad=15)
        ax.grid(linewidth=1)

        # Central line at r=0
        line, = ax.plot(np.linspace(-np.pi, np.pi, 300), np.zeros(300), color='black', lw=2)
        line.set_path_effects([pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])

        # Colorbar
        cbar = figure.colorbar(cs, ax=ax, fraction=0.05, pad=0.11)
        cbar.set_label('Normalized amplitude', fontsize=18)
        #cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=18)

        plt.title(title, fontsize=16)
        plt.show()

    def plot_3d_fiber_response(self, wave_type, cmap=None, title="DAS 3D Response"):
        """
        3D plot of the DAS fiber response with directional line.
        """
        az = np.linspace(0, 360, 500)      # 1D
        dip = np.linspace(-90, 90, 500)       # 1D

        ones = np.ones((len(dip), len(az)))
        az_grid = az * ones
        dip_grid = dip[:, np.newaxis] * ones

        data = self.sensitivity( wave_type, az_grid, dip_grid)
        #data = np.where(np.abs(data)<1e-14, np.nan, data)
        data = data.T
        data_signed = data
        data_radius = np.abs(data)
        # Spherical grid
        theta = np.linspace(0, np.pi, data_radius.shape[0])
        phi = np.linspace(0, 2*np.pi, data_radius.shape[1])
        Theta, Phi = np.meshgrid(theta, phi)

        # Radius for shape
        rho = data_radius / np.max(data_radius)

        # Convert to Cartesian
        X = rho * np.sin(Theta) * np.cos(Phi)
        Y = rho * np.sin(Theta) * np.sin(Phi)
        Z = rho * np.cos(Theta)

        # Normalize signed values for color mapping
        vmax = np.max(np.abs(data_signed))
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        colors = plt.cm.seismic(norm(data_signed))  # Diverging color map

        # Plotting
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        if cmap is None:
            if wave_type in ['P', 'S'] and data.min()>0:
                cmap = 'Reds'

            elif wave_type in ['SV', 'SH'] or wave_type in ['P', 'S'] and data.min()<0:
                cmap = 'seismic'
                
            else:
                cmap = 'cool'

        # Mesh step size for cleaner grid
        step = 1
        surf = ax.plot_surface(
            X[::step, ::step], Y[::step, ::step], Z[::step, ::step],
            facecolors=colors[::step, ::step],
            edgecolor='k', linewidth=0.3, alpha=0.6 , shade=False
        )

        # Equal scaling
        max_range = 1.05  # a bit over 1 to give margin
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_box_aspect([1, 1, 1])

        # Colorbar using same normalization
        mappable = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
        mappable.set_array(data_signed)
        #fig.colorbar(mappable, shrink=0.4, aspect=15, pad=0.005, label="Amplitude")

        # Optional directional line
        azimuth = np.radians(self.azimuth)
        dip = np.radians(-self.dip+90)
        dx, dy, dz = np.sin(dip) * np.cos(azimuth), np.sin(dip) * np.sin(azimuth), -np.cos(dip)
        t = np.linspace(-1.5, 1.5, 100)
        ax.plot3D(t*dx, t*dy, t*dz, color='black', linewidth=2.5, label="DAS Fiber")

        ticks = [-1, -0.5, 0.5, 1]

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)

        # Axes & view
        ax.tick_params(axis='x', labelsize=20, pad = 4,labelrotation=15)
        ax.tick_params(axis='y', labelsize=20, pad = 0,labelrotation=-15 )
        ax.tick_params(axis='z', labelsize=20, pad = 10)


        # Increase label font size
        ax.set_xlabel("X", fontsize=20, labelpad=0)
        ax.set_ylabel("Y", fontsize=20, labelpad=0)
        ax.set_zlabel("Z", fontsize=20, labelpad=0)
        ax.view_init(45, 45)
        # ax.legend()

        plt.tight_layout()
        plt.show()

class GeoUtils:
    """
    Utility functions for coordinate transformations.
    """

    R_EARTH = 6371000.0  # meters

    @staticmethod
    def latlon_to_ecef(lat, lon, depth):
        lat = np.radians(lat)
        lon = np.radians(lon)

        r = GeoUtils.R_EARTH - depth

        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)

        return np.array([x, y, z])

    @staticmethod
    def ecef_to_enu(vec, lat, lon):
        lat = np.radians(lat)
        lon = np.radians(lon)

        R = np.array([
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
            [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]
        ])

        return R @ vec

    @staticmethod
    def azimuth_dip(lat1, lon1, depth1, lat2, lon2, depth2):
        p1 = GeoUtils.latlon_to_ecef(lat1, lon1, depth1)
        p2 = GeoUtils.latlon_to_ecef(lat2, lon2, depth2)

        d = p2 - p1
        E, N, U = GeoUtils.ecef_to_enu(d, lat1, lon1)

        az = (np.degrees(-np.arctan2(E, N)) + 360) % 360
        dip = -np.degrees(np.arctan2(-U, np.hypot(E, N)))

        return az, dip

class Ray:
    """
    Represents a seismic ray path.

    The ray is expected as an array:
    [r, theta, phi] or Cartesian depending on usage.
    """

    def __init__(self, ray_array):
        self.ray = np.asarray(ray_array)

    def plot_views(self):
        """
        Plot the ray in 4 views:
        - 3D
        - Top view (lon vs lat)
        - Lateral (lon vs depth)
        - Longitudinal (lat vs depth)
        """

        # Convert to geographic coordinates (lat, lon, depth)
        geo = np.array(sph2geo(self.ray))

        lat, lon, depth = geo.T

        fig = plt.figure(figsize=(10, 10))

        # --- 3D plot ---
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(lon, lat, depth)
        ax1.scatter(lon[0], lat[0], depth[0], c = 'r', label = 'DAS')
        ax1.scatter(lon[-1], lat[-1], depth[-1], c = 'green', label = 'source')
        ax1.set_title("3D View")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_zlabel("Depth")
        ax1.view_init(90, 90)

        ax1.invert_zaxis()

        # --- Top view ---
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(lon, lat)
        ax2.set_title("Top View (Map)")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.axis('equal')

        # --- Lateral view (lon vs depth) ---
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(lon, depth)
        ax3.set_title("Lateral View")
        ax3.set_xlabel("Longitude")
        ax3.set_ylabel("Depth")
        ax3.invert_yaxis()

        # --- Longitudinal view (lat vs depth) ---
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(lat, depth)
        ax4.set_title("Longitudinal View")
        ax4.set_xlabel("Latitude")
        ax4.set_ylabel("Depth")
        ax4.invert_yaxis()

        plt.tight_layout()
        plt.show()

    def azimuth_and_dip(self, side="DAS", n_nodes=10, average=True):
        """
        Compute azimuth and dip of the ray near one end.

        Parameters
        ----------
        side : str
            'DAS' (start of ray) or 'source' (end of ray)
        n_nodes : int
            Number of nodes to consider
        average : bool
            If True, average segment directions (better for curved rays)

        Returns
        -------
        azimuth : float (degrees, from North, clockwise)
        dip : float (degrees, positive downward)
        """

        # Select segment depending on side
        if side == "DAS":
            sl = slice(0, n_nodes)
        elif side == "source":
            sl = slice(-n_nodes, None)
        else:
            raise ValueError("side must be 'DAS' or 'source'")
        
                # Convert to geographic coordinates (lat, lon, depth)
        path_node = np.array(sph2geo(self.ray[sl,:]))

        # --- Compute direction ---
        if average:
            # Mean of segment vectors (better for curved rays)
            vectors = np.diff(path_node, axis=0)
            direction = np.median(vectors, axis=0)
        else:
            # Simple start → end
            direction = path_node[-1] - path_node[0]
        
        if side == 'DAS':
            direction = -direction

        dlat, dlon, ddepth = direction.T

        # Mean latitude for longitude scaling
        mean_lat = np.radians(path_node[0].mean())

        # Convert degrees to km
        dy = dlat * 111.32
        dx = dlon * 111.32 * np.cos(mean_lat)

        # Azimuth
        azimuth = -np.degrees(np.arctan2(dx, dy)) % 360

        # Horizontal distance in km
        horizontal = np.sqrt(dx**2 + dy**2)

        # Dip (positive downward)
        dip = -np.degrees(np.arctan2(ddepth, horizontal))

        return azimuth, dip 

    def length_cartesian(self):
        """
        Compute ray length assuming Cartesian coordinates.
        """
        ray_t = self.ray.T
        differences = np.diff(ray_t, axis=1)
        segment_lengths = np.linalg.norm(differences, axis=0)
        return np.sum(segment_lengths)

    def length_spherical(self):
        """
        Compute ray length from spherical coordinates [r, theta, phi].
        Returns length in km.
        """
        r = self.ray[:, 0] * 1e3
        theta = self.ray[:, 1]
        phi = self.ray[:, 2]

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        xyz = np.stack([x, y, z], axis=1)

        return np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)) / 1000
    

#########################################################
###################### SOLO FCT #########################
#########################################################

#########################################
######### DAS channel metadata ##########
#########################################

import math

def distance_3d_km(lat1, lon1, depth1_km, lat2, lon2, depth2_km):
    """
    Calculate 3D distance (km) between two points defined by:
    latitude, longitude, and depth (km).

    Depth is positive downward (e.g., underground or below sea level).

    Returns:
        distance in kilometers
    """

    # Earth radius in km
    R = 6371.0

    # Convert lat/lon from degrees to radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Haversine formula for surface distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    surface_distance = R * c

    # Depth difference
    depth_diff = depth2_km - depth1_km

    # Full 3D distance using Pythagoras
    distance = math.sqrt(surface_distance**2 + depth_diff**2)

    return distance



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

    