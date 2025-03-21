import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import datetime
from shapely.vectorized import contains
import cartopy.io.shapereader as shpreader
from util import lotlong2dist
import matplotlib.lines as mlines

def plot_interpolated_temperature(
    lat_lon_intp,
    dept_time_intp,
    train_points,  # Array of training points with temperatures
    ref_point,     # Reference point with temperature
    y,
    n_grid,
    depth_range,
    test_points=None,   # Array of test points without temperatures
    path_to_save = None,
    threshold_depth = 10,
    threshold_time = 0.05,
    threshold_geodetic = 2000,
    marker_size_range = (10, 100),

    is_uncertainty = False,

    no_edges = False,

    lat_lon_temp_min_max = None,

    depth_time_sample_ratio = None,

):
    '''
    is_uncertainty: bool
        If True, the last column of train_points and ref_point is not filled with temperature values.
    '''
    # Combine only train_points and test_points for filtering (exclude ref_point)
    if test_points is None:
        all_points = train_points
    else:
        all_points = np.vstack([train_points, test_points])
    n_train = len(train_points)  
    
    # Create train/test masks that we'll maintain through filtering
    is_train = np.zeros(len(all_points), dtype=bool)
    is_train[:n_train] = True  # First n_train points are training points

    # Calculate differences for filtering and marker sizes

    # Depth and time differences
    depth_diff = np.abs(all_points[:,0] - ref_point[0])
    time_diff = np.abs(all_points[:,3] - ref_point[3])
    dt_distance = np.sqrt((depth_diff / threshold_depth)**2 + (time_diff / threshold_time)**2)

    # Geodetic distances
    ref_latlon = np.array([[ref_point[1], ref_point[2]]])
    all_latlon = all_points[:,1:3]
    geodetic_distances = lotlong2dist(ref_latlon, all_latlon)[0]

    # Prepare masks based on filters

    # First subplot (Map): Filter by depth and time differences
    depth_mask = depth_diff < threshold_depth
    time_mask = time_diff < threshold_time
    mask_dt = depth_mask & time_mask
    observations_map = all_points[mask_dt]
    dt_distance_map = dt_distance[mask_dt]
    is_train_map = is_train[mask_dt]  # Keep track of which points are training points

    # Marker sizes represent depth and time distances (invisible dimensions)
    if dt_distance_map.max() == dt_distance_map.min():
        sizes_map = np.ones_like(dt_distance_map) * marker_size_range[1]
    else:
        sizes_map = (dt_distance_map - dt_distance_map.min()) / (dt_distance_map.max() - dt_distance_map.min()) * (marker_size_range[0] - marker_size_range[1]) + marker_size_range[1]

    # Second subplot (Depth-Time): Filter by geodetic distance
    geodetic_mask = geodetic_distances < threshold_geodetic
    observations_depth_time = all_points[geodetic_mask]
    geodetic_distances_depth_time = geodetic_distances[geodetic_mask]
    is_train_dt = is_train[geodetic_mask]  # Keep track of which points are training points

    # Marker sizes represent geodetic distances (invisible dimensions)
    if geodetic_distances_depth_time.max() == geodetic_distances_depth_time.min():
        sizes_depth_time = np.ones_like(geodetic_distances_depth_time) * marker_size_range[1]
    else:
        sizes_depth_time = (geodetic_distances_depth_time - geodetic_distances_depth_time.min()) / (geodetic_distances_depth_time.max() - geodetic_distances_depth_time.min()) * (marker_size_range[0] - marker_size_range[1]) + marker_size_range[1]

    # Prepare grids (existing code)
    lon_grid = np.linspace(-180, 180, n_grid)
    lat_grid = np.linspace(-90, 90, n_grid)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Read land polygons and create a mask (existing code)
    land_shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
    land_geoms = list(shpreader.Reader(land_shp).geometries())
    land_union = unary_union(land_geoms)
    land_union_prepared = prep(land_union)

    # Create a mask for the data points
    lon_flat = lon_mesh.ravel()
    lat_flat = lat_mesh.ravel()
    mask_flat = contains(land_union, lon_flat, lat_flat)
    mask = mask_flat.reshape(lon_mesh.shape)

    # Apply the mask to the interpolated data
    lat_lon_intp_masked = np.ma.array(lat_lon_intp, mask=mask)

    # Prepare depth and time grids
    dept_grid = np.linspace(depth_range[0], depth_range[1], n_grid)
    time_grid = np.linspace(0, 1, n_grid)
    dept_mesh, time_mesh = np.meshgrid(dept_grid, time_grid)

    # Plot the interpolated data over a real-world map
    fig = plt.figure(figsize=(12, 12))

    # First subplot: Map
    ax_map = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND)
    ax_map.add_feature(cfeature.OCEAN)
    ax_map.add_feature(cfeature.COASTLINE)
    ax_map.add_feature(cfeature.BORDERS, linestyle=':')

    if lat_lon_temp_min_max is None:
        lat_lon_temp_min_max = [np.nanmin(lat_lon_intp_masked), np.nanmax(lat_lon_intp_masked)]
        # counting all points with temperature
        if not is_uncertainty:
            lat_lon_temp_min_max[0] = np.nanmin([ lat_lon_temp_min_max[0], np.nanmin(observations_map[:,4]) ])
            lat_lon_temp_min_max[1] = np.nanmax([ lat_lon_temp_min_max[1], np.nanmax(observations_map[:,4]) ])
    # logging.info(f'lat_lon_temp_min_max: {lat_lon_temp_min_max}')

    # Plot interpolated temperature
    cf = ax_map.contourf(
        lon_mesh, lat_mesh, lat_lon_intp_masked, 60,
        transform=ccrs.PlateCarree(), cmap='viridis',
        vmin=lat_lon_temp_min_max[0], vmax=lat_lon_temp_min_max[1],
    )

    # Plot observations filtered by depth and time
    # Training points in red
    train_obs = observations_map[is_train_map]
    if train_obs.shape[0] > 0:
        sc_map_train = ax_map.scatter(
            train_obs[:,2], train_obs[:,1],  # lon, lat
            c='cyan' if is_uncertainty else train_obs[:,4], 
            cmap='viridis', norm=cf.norm,
            s=sizes_map[is_train_map], edgecolors='cyan', transform=ccrs.PlateCarree(),
            linewidths= 0 if no_edges else sizes_map[is_train_map] / 800,
        )

    # Test points in blue
    test_obs = observations_map[~is_train_map]
    if test_obs.shape[0] > 0:
        sc_map_test = ax_map.scatter(
            test_obs[:,2], test_obs[:,1],  # lon, lat
            # facecolors='none',
            c='fuchsia' if is_uncertainty else test_obs[:,4],
            cmap='viridis', norm=cf.norm, 
            edgecolors='fuchsia', transform=ccrs.PlateCarree(),
            s=sizes_map[~is_train_map], marker='o',
            linewidths= 0 if no_edges else sizes_map[~is_train_map] / 800,
        )

    # Plot reference point as a square
    if np.isnan(ref_point[4]):
        # plot solid square if no temperature is available
        ax_map.scatter(
            ref_point[2], ref_point[1],  # lon, lat
            c='red', marker='o', s=marker_size_range[1], edgecolors='red', transform=ccrs.PlateCarree(),
            linewidths= 0 if no_edges else marker_size_range[1] / 300,
        )
    elif np.isinf(ref_point[4]):
        pass
    else:
        ax_map.scatter(
            ref_point[2], ref_point[1],  # lon, lat
            # c=ref_point[4],
            c = 'red' if is_uncertainty else ref_point[4], 
            cmap='viridis', norm=cf.norm,
            marker='o', s=marker_size_range[1], edgecolors='red', transform=ccrs.PlateCarree(),
            linewidths= 0 if no_edges else marker_size_range[1] / 300,
        )

    # Update legend
    train_marker = mlines.Line2D([], [], color='cyan', marker='o', 
                                 fillstyle='full' if is_uncertainty else 'none', 
                                 linestyle='None',
                                markersize=marker_size_range[0]**0.5, markeredgewidth= marker_size_range[0] / 300,label='Train Points (Furthest)')
    test_marker = mlines.Line2D([], [], color='fuchsia', marker='o', 
                                fillstyle= 'full' if is_uncertainty else 'none',
                               linestyle='None', markersize=marker_size_range[0]**0.5, markeredgewidth= marker_size_range[0] / 300,label='Test Points (Furthest)')
    ref_marker = mlines.Line2D([], [], color='red', marker='o', 
                               fillstyle='full' if np.isnan(ref_point[4]) else 'none', 
                               linestyle='None',
                              markersize=marker_size_range[1]**0.5, markeredgewidth= marker_size_range[1] / 300,label='Central Point')
    # if test_points is not None:
    #     ax_map.legend(handles=[ref_marker, train_marker, test_marker], loc='upper right')
    # else:
    #     ax_map.legend(handles=[ref_marker, train_marker], loc='upper right')
    
    import matplotlib.ticker as mticker  # Add this import at the top if not already present

    gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))

    # Update map title with depth and time ranges and marker size information
    t = ref_point[3]
    t_datetime = datetime.datetime(y, 1, 1) + datetime.timedelta(
        days=t * (datetime.datetime(y + 1, 1, 1) - datetime.datetime(y, 1, 1)).days
    )
    depth_range_str = f"±{threshold_depth:.1f}m"
    time_range_str = f"±{threshold_time*365:.1f}days"
    marker_size_info_map = f"Marker size: {dt_distance_map.min()*threshold_depth:.2f} to {dt_distance_map.max()*threshold_depth:.2f} units"
    ax_map.set_title(f'{"Uncertainty of " if is_uncertainty else ""}Interpolated Temperature at {ref_point[0]:.0f}m ({depth_range_str}) \n'
                     f'at {t_datetime} ({time_range_str})')

    # Second subplot: Depth-Time
    ax_depth_time = fig.add_subplot(2, 1, 2)
    time_grid_dates = [
        datetime.datetime(y, 1, 1) + datetime.timedelta(
            days=tm * (datetime.datetime(y + 1, 1, 1) - datetime.datetime(y, 1, 1)).days
        )
        for tm in time_grid
    ]
    dept_mesh, time_mesh_dates = np.meshgrid(dept_grid, time_grid_dates)


    dept_time_temp_min_max = [np.nanmin(dept_time_intp), np.nanmax(dept_time_intp)]
    # counting all points with temperature
    if not is_uncertainty:
        dept_time_temp_min_max[0] = np.nanmin([ dept_time_temp_min_max[0], np.nanmin(observations_depth_time[:,4]) ])
        dept_time_temp_min_max[1] = np.nanmax([ dept_time_temp_min_max[1], np.nanmax(observations_depth_time[:,4]) ])

    # Plot interpolated temperature in depth-time space
    cf2 = ax_depth_time.contourf(
        time_mesh_dates, dept_mesh, dept_time_intp.T, 60,
        cmap='viridis',
        vmin=dept_time_temp_min_max[0], vmax=dept_time_temp_min_max[1],
    )

    # Convert times for observations
    observed_times = [
        datetime.datetime(y, 1, 1) + datetime.timedelta(
            days=t_ * (datetime.datetime(y + 1, 1, 1) - datetime.datetime(y, 1, 1)).days
        )
        for t_ in observations_depth_time[:,3]
    ]

    # Plot observations filtered by geodetic distance
    # Training points in red
    train_obs_dt = observations_depth_time[is_train_dt]
    train_times = [datetime.datetime(y, 1, 1) + datetime.timedelta(
        days=t_ * (datetime.datetime(y + 1, 1, 1) - datetime.datetime(y, 1, 1)).days
    ) for t_ in train_obs_dt[:,3]]
    
    if train_obs_dt.shape[0] > 0:
        s_ = sizes_depth_time.copy()
        if depth_time_sample_ratio is not None:
            # sort by depth
            # idx = np.argsort(train_obs_dt[:,0])
            idx = np.lexsort((train_obs_dt[:,0], train_obs_dt[:,3]))
            # sample every n-th point
            idx = idx[::depth_time_sample_ratio].astype(np.int64)
            # train_times_, train_obs_dt_ = train_times_[idx], train_obs_dt_[idx]
            # sizes_depth_time_ = sizes_depth_time_[idx]
        else:
            idx = slice(None)

        sc_depth_time_train = ax_depth_time.scatter(
            np.array(train_times)[idx].tolist(), train_obs_dt[idx,0],  # time, depth
            # c=train_obs_dt[:,4],
            c='cyan' if is_uncertainty else train_obs_dt[idx,4], 
            cmap='viridis', norm=cf2.norm,
            s=sizes_depth_time[is_train_dt][idx], edgecolors='cyan',
            linewidths= 0 if no_edges else sizes_depth_time[is_train_dt][idx] / 800, 
        )

    # Test points in blue
    test_obs_dt = observations_depth_time[~is_train_dt]
    if test_obs_dt.shape[0] > 0:
        test_times = [datetime.datetime(y, 1, 1) + datetime.timedelta(
            days=t_ * (datetime.datetime(y + 1, 1, 1) - datetime.datetime(y, 1, 1)).days
        ) for t_ in test_obs_dt[:,3]]
        
        sc_depth_time_test = ax_depth_time.scatter(
            test_times, test_obs_dt[:,0],  # time, depth
            # facecolors='none',
            c='fuchsia' if is_uncertainty else test_obs_dt[:,4],
            cmap='viridis', norm=cf2.norm, 
            edgecolors='fuchsia',
            s=sizes_depth_time[~is_train_dt], marker='o',
            linewidths= 0 if no_edges else sizes_depth_time[~is_train_dt] / 800,
        )

    # Plot reference point as a square in depth-time plot
    ref_time = datetime.datetime(y, 1, 1) + datetime.timedelta(
        days=ref_point[3] * (datetime.datetime(y +1, 1, 1) - datetime.datetime(y, 1, 1)).days
    )
    if np.isnan(ref_point[4]):
        # plot solid square if no temperature is available
        ax_depth_time.scatter(
            ref_time, ref_point[0],  # time, depth
            c='red', marker='o', s=marker_size_range[1], edgecolors='red',
            linewidths= 0 if no_edges else marker_size_range[1] / 300,
        )
    elif np.isinf(ref_point[4]):
        pass
    else:
        ax_depth_time.scatter(
            ref_time, ref_point[0],  # time, depth
            # c=ref_point[4],
            c = 'red' if is_uncertainty else ref_point[4],
            cmap='viridis', norm=cf2.norm,
            marker='o', s=marker_size_range[1], edgecolors='red',
            linewidths= 0 if no_edges else marker_size_range[1] / 300,
        )

    # Update legend for depth-time plot
    train_marker_dt = mlines.Line2D([], [], color='cyan', marker='o', 
                                    fillstyle='full' if is_uncertainty else 'none',
                                    linestyle='None',
                                   markersize=marker_size_range[0]**0.5, markeredgewidth= marker_size_range[0] / 300,label='Train Points (Furthest)')
    test_marker_dt = mlines.Line2D([], [], color='fuchsia', marker='o', 
                                fillstyle='full' if is_uncertainty else 'none',
                                  linestyle='None', markersize=marker_size_range[0]**0.5, markeredgewidth= marker_size_range[0] / 300,label='Test Points (Furthest)')
    ref_marker_dt = mlines.Line2D([], [], color='red', marker='o', fillstyle='full' if np.isnan(ref_point[4]) else 'none', linestyle='None',
                                 markersize=marker_size_range[1]**0.5, markeredgewidth= marker_size_range[1] / 300,label='Central Point')
    h = [ref_marker_dt]
    if train_obs_dt.shape[0] > 0 or train_obs.shape[0] > 0:
        h.append(train_marker_dt)
    if test_obs_dt.shape[0] > 0 or test_obs.shape[0] > 0:
        h.append(test_marker_dt)
    ax_depth_time.legend(handles=h, loc='upper right')

    # Update depth-time title with geodetic range and marker size information
    mean_lat = np.mean(observations_depth_time[:,1])
    mean_lon = np.mean(observations_depth_time[:,2])
    geodetic_range_str = f"within {threshold_geodetic:.0f}km"
    marker_size_info_dt = f"Marker size: {geodetic_distances_depth_time.min():.1f} to {geodetic_distances_depth_time.max():.1f} km"

    # Reverse the depth axis
    ax_depth_time.invert_yaxis()

    # Format the time axis
    from matplotlib.dates import DateFormatter, AutoDateLocator
    date_locator = AutoDateLocator()
    date_formatter = DateFormatter('%m-%d')
    ax_depth_time.xaxis.set_major_locator(date_locator)
    ax_depth_time.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate()

    # Update labels and title
    ax_depth_time.set_title(
        f'{"Uncertainty of " if is_uncertainty else ""}Interpolated Temperature in Depth-Time Space of {y} at lat={ref_point[1]:.1f}, lon={ref_point[2]:.1f} ({geodetic_range_str})'
    )
    ax_depth_time.set_xlabel('Time')
    ax_depth_time.set_ylabel('Depth (m)')

    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap='viridis', norm=cf.norm)
    sm.set_array([])  # Dummy array for the scalar mappable
    cbar_map = fig.colorbar(sm, ax=ax_map, orientation='vertical', fraction=0.046, pad=0.04)

    # Show colorbars
    # range is lat_lon_temp_min_max
    cbar_map.set_label('Temperature')
    cbar_map.mappable.set_clim(lat_lon_temp_min_max)

    sm2 = ScalarMappable(cmap='viridis', norm=cf2.norm)
    sm2.set_array([])  # Dummy array for the scalar mappable
    cbar_depth_time = fig.colorbar(sm2, ax=ax_depth_time, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_depth_time.set_label('Temperature')
    


    plt.tight_layout()
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()

