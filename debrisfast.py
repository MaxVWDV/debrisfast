import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from scipy.ndimage import median_filter, uniform_filter, distance_transform_edt, sobel, binary_closing
from skimage.morphology import closing, disk, remove_small_holes, binary_closing
from skimage.draw import polygon2mask
from pysheds.grid import Grid
import os
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (10, 10)

def debrisfast_main(dem_path,start_volume,start_point,
    num_iterations = 10, end_condition = 1e3,
    min_thickness = 1, cellsize=30, half_width = 25,
    laharz_alpha = 0.15, laharz_beta = 200,
    rho = 2000, erosion_coeff = 0.005, flow_time = 60, ero_max = 2, manning_n = 0.05):
    # Example parameter sets 
    # dem_path = 'waydem.tif'
    # start_volume = 1e6  # initial flow volume in m³
    # start_point = (296, 163)  #Starting point (row, col)
    # num_iterations = 10  # maximum number of iterations
    # end_condition = 1e3  # volume change threshold to stop
    # min_thickness = 1  # minimum flow thickness in meters
    # cellsize=30
    #laharz_alpha = 0.05
    #laharz_beta = 200,
    # rho = 2000  # kg/m³
    # erosion_coeff = 0.005  # unitless entrainment efficiency
    # flow_time = 60  # seconds duration of flow at given location
    # ero_max = 2  # Max erosion in meters; adjust as needed
    # manning_n = 0.1 #Manning's roughness coefficient
    
    #Preprocess DEM
    dem,hand,slope_deg,flow_path = debrisfast_dem_preprocess(dem_path, start_point, cellsize=30)
    
    #Extract cross-sections
    cross_section_values, distances, angles = debrisfast_xsections(dem, flow_path, half_width=25)
    
    #Compute flow outline, depth, and travel time
    flow_mask, flow_depth_raster, velocity_raster, travel_time_raster, erosion_total=debrisfast_mainflow(
        dem,hand,slope_deg,flow_path, dem_path,start_volume,start_point, cross_section_values, distances, angles,
        num_iterations = 10, end_condition = 1e3,
        min_thickness = 1, cellsize=30, half_width = 25,
        laharz_alpha = 0.15, laharz_beta = 200,
        rho = 2000, erosion_coeff = 0.005, flow_time = 60, ero_max = 2, manning_n = 0.05)
        
    #Plot figures
    debrisfast_figures(dem, flow_depth_raster, travel_time_raster)
    
    #Save outputs to directory
    output_dir='outputs/'
    debrisfast_save(
        output_dir,
        dem_path,
        flow_mask=flow_mask,
        flow_depth_raster=flow_depth_raster,
        velocity_raster=velocity_raster,
        travel_time_raster=travel_time_raster,
        erosion_total=erosion_total
    )
        
    return flow_mask, flow_depth_raster, velocity_raster, travel_time_raster, erosion_total

    
def debrisfast_dem_preprocess(dem_path, start_point, cellsize=30):
    # Load DEM 
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    # Hydrological conditioning 
    pit_filled = grid.fill_pits(dem)
    depressions_filled = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(depressions_filled)

    # Flow direction and accumulation 
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated, dirmap=dirmap)
    acc = grid.accumulation(fdir, dirmap=dirmap)

    #Stream mask and HAND - hight above nearest drainage
    hand = grid.compute_hand(fdir, dem, acc > 200)

    #Slope 
    dx = sobel(inflated, axis=1, mode='nearest')
    dy = sobel(inflated, axis=0, mode='nearest')
    dz_dx = dx / (8 * cellsize)
    dz_dy = dy / (8 * cellsize)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.rad2deg(slope_rad)

    #Manual flow path tracing from start_point 
    row, col = start_point
    rows, cols = [row], [col]

    # D8 flow direction lookup table
    delta = {
        1: (0, 1),    # East
        2: (1, 1),    # SE
        4: (1, 0),    # South
        8: (1, -1),   # SW
        16: (0, -1),  # West
        32: (-1, -1), # NW
        64: (-1, 0),  # North
        128: (-1, 1)  # NE
    }

    for _ in range(10000):  # max steps
        dir_val = fdir[row, col]
        if dir_val not in delta:
            break
        d_row, d_col = delta[dir_val]
        row, col = row + d_row, col + d_col
        if (row < 0 or row >= fdir.shape[0]) or (col < 0 or col >= fdir.shape[1]):
            break
        rows.append(row)
        cols.append(col)

    flow_path = list(zip(rows, cols))
    
    return(dem,hand,slope_deg,flow_path)


    
def debrisfast_xsections(dem, flow_path, half_width=25):
    #     Extract cross-sectional values perpendicular to a path in a DEM.
    #     Parameters:
    #     - dem: 2D numpy array of the DEM
    #     - x, y: arrays of column and row indices (same length)
    #     - angles: array of angles (in radians) at each point
    #     - half_width: number of pixels to sample on either side
    #     Returns:
    #     - cross_section_values: (N, 2*half_width+1) array of sampled elevations
    #     - distances: array of distances from center [-half_width, ..., +half_width]
    
    rows, cols = zip(*flow_path)
    rows = np.array(rows)
    cols = np.array(cols)

    # Approximate local angle along path using central differences
    dx = np.gradient(cols)
    dy = np.gradient(rows)
    angles = np.arctan2(dy, dx)
    
    num_points = len(cols)
    distances = np.arange(-half_width, half_width + 1)
    cross_section_values = np.full((num_points, len(distances)), np.nan)

    nrows, ncols = dem.shape

    for i in range(num_points):
        angle = angles[i] + np.pi / 2
        dx = np.cos(angle)
        dy = np.sin(angle)

        cross_x = cols[i] + distances * dx
        cross_y = rows[i] + distances * dy

        cross_x_int = np.clip(np.round(cross_x).astype(int), 0, ncols - 1)
        cross_y_int = np.clip(np.round(cross_y).astype(int), 0, nrows - 1)

        cross_section_values[i, :] = dem[cross_y_int, cross_x_int]

    return cross_section_values, distances, angles





def debrisfast_mainflow(
        dem,hand,slope_deg,flow_path, dem_path,start_volume,start_point, cross_section_values, distances, angles,
        num_iterations = 10, end_condition = 1e3,
        min_thickness = 1, cellsize=30, half_width = 25,
        laharz_alpha = 0.15, laharz_beta = 200,
        rho = 2000, erosion_coeff = 0.005, flow_time = 60, ero_max = 2, manning_n = 0.05):
    
    V = np.full(len(flow_path), start_volume)
    ErosionTotal_prev = np.zeros_like(dem)
    
    rows, cols = zip(*flow_path)

    for iteration in range(num_iterations):
        
        #Cross-sectional area and width calculation
        # Ensure cross_section_values and other variables (dd, angles, min_thickness, V, cellsize) are pre-defined.

        A = laharz_alpha * V ** (2 / 3)
        B = laharz_beta * np.max(V)**(2 / 3)

        dd = cross_section_values.shape[1] // 2


        max_width = np.floor(A / min_thickness / cellsize / 2).astype(int)
        width = np.minimum(max_width - 1, dd)

        depth = np.full(angles.shape, np.nan)

        area_minus = 0.5 * np.abs(
            (cellsize * 1) * (cross_section_values[:, dd + 2] - cross_section_values[:, dd]) +
            (cellsize * 2) * (cross_section_values[:, dd] - cross_section_values[:, dd + 1])
        )

        area_plus = 0.5 * np.abs(
            (cellsize * 1) * (cross_section_values[:, dd + np.minimum(max_width[0] - 1, dd)] - cross_section_values[:, dd - np.minimum(max_width[0] - 1, dd)]) +
            (cellsize * np.minimum(max_width[0] - 1, dd) * 2) * (cross_section_values[:, dd - np.minimum(max_width[0] - 1, dd)] - cross_section_values[:, dd])
        )

        depth_minus = np.ones_like(angles)
        depth_plus = np.ones_like(angles)

        for numpix in range(1, max(np.minimum(max_width[0] - 1, dd), 1) + 1):
            area = 0.5 * np.abs(
                (cellsize * numpix) * (cross_section_values[:, dd + numpix] - cross_section_values[:, dd - numpix]) +
                (cellsize * numpix * 2) * (cross_section_values[:, dd - numpix] - cross_section_values[:, dd])
            )

            alld = np.maximum(cross_section_values[:, dd - numpix], cross_section_values[:, dd + numpix]) - cross_section_values[:, dd]

            condition_width = (area > A) & (width == np.minimum(max_width - 1, dd))
            width[condition_width] = numpix * 2

            area_minus[area <= A] = area[area <= A]
            area_plus[area > A] = area[area > A]

            depth_minus[area <= A] = alld[area <= A]
            depth_plus[area > A] = alld[area > A]

        # Subpixel width and depth
        width_subpix = width.astype(float)
        normal = (area_minus < area_plus) & (area_minus < A) & (area_plus > A)

        width_subpix[normal] = width[normal] - 2 + 2 * ((A[normal] - area_minus[normal]) / (area_plus[normal] - area_minus[normal]))
        width_subpix[area_minus > A] = (A[area_minus > A] / area_minus[area_minus > A]) * 2
        width_subpix[area_plus < A] = area_minus[area_plus < A] / A[area_plus < A] * 2

        width_subpix[width_subpix < 0.25] = 0.25

        depth[normal] = depth_minus[normal] + (depth_plus[normal] - depth_minus[normal]) * ((A[normal] - area_minus[normal]) / (area_plus[normal] - area_minus[normal]))
        depth[area_minus > A] = depth_minus[area_minus > A] + (depth_plus[area_minus > A] - depth_minus[area_minus > A]) * (A[area_minus > A] / area_minus[area_minus > A])
        depth[area_plus < A] = depth_minus[area_plus < A] + (depth_plus[area_plus < A] - depth_minus[area_plus < A]) * (area_minus[area_plus < A] / A[area_plus < A])

        depth[depth < 1] = 1


        # Smooth widths and depths to prevent unphysical sudden changes

        width = uniform_filter(width.astype(float), size=10)
        depth = uniform_filter(depth.astype(float), size=10)


        ### Downstream cumulative area
        cum_area = np.cumsum(np.maximum(width_subpix, 1)) * cellsize**2
        cum_area[cum_area > B] = np.nan
        down_pix = np.nanargmax(cum_area)

        #  Begin: Mask out-of-bounds indices 
        r = np.array(rows[:down_pix])
        c = np.array(cols[:down_pix])
        
        valid = (
            (r >= 0) & (r < dem.shape[0]) &
            (c >= 0) & (c < dem.shape[1])
        )
        rv = r[valid]
        cv = c[valid]
        depthv = depth[:down_pix][valid]
        #  End mask 

        ### Generate flow mask
        poly_coords = []
        for i in range(down_pix):
            angle = angles[i] + np.pi/2
            dx, dy = np.cos(angle), np.sin(angle)
            hw = width[i] / 2

            lx, ly = cols[i] + hw*dx, rows[i] + hw*dy
            rx, ry = cols[i] - hw*dx, rows[i] - hw*dy
            poly_coords.append(((lx, ly), (rx, ry)))

        left_edges, right_edges = zip(*poly_coords)
        left_edges = np.array(left_edges)
        right_edges = np.array(right_edges)

        poly_y = np.concatenate([left_edges[:,1], right_edges[::-1,1]])  # y = rows
        poly_x = np.concatenate([left_edges[:,0], right_edges[::-1,0]])  # x = cols

        poly_y = np.clip(poly_y, 0, dem.shape[0]-1)
        poly_x = np.clip(poly_x, 0, dem.shape[1]-1)

        flow_mask = polygon2mask(dem.shape, np.column_stack([poly_y, poly_x]))
        flow_mask = binary_closing(flow_mask)

        ### Calculate flow depth raster
        # Fill centerline depthmap
        depthmap = np.full(dem.shape, np.nan)
        depthmap[rv, cv] = depthv

        # Interpolate only within flow_mask using nearest neighbor
        _, (row_idx, col_idx) = distance_transform_edt(np.isnan(depthmap), return_indices=True)

        # Clip indices to within valid array bounds
        row_idx = np.clip(row_idx, 0, depthmap.shape[0] - 1)
        col_idx = np.clip(col_idx, 0, depthmap.shape[1] - 1)

        nearest_depth = depthmap[row_idx, col_idx]

        # Restrict to flow polygon
        flow_depth_raster = np.full_like(depthmap, 0)

        flow_depth_raster[flow_mask] = np.maximum(nearest_depth[flow_mask] - hand[flow_mask], min_thickness)

        # plt.imshow(depthmap[:,:]);plt.show()
        flow_mask = flow_mask.astype(bool)

        flow_depth_raster = np.where(flow_mask, flow_depth_raster, 0)
        # plt.imshow(flow_depth_raster);plt.show() #plot every iteration

        slope_radians = np.deg2rad(slope_deg)
        slope_m_per_m = np.maximum(np.tan(slope_radians), 1e-3)  # avoid division by zero
        velocity_raster = (flow_depth_raster**(2/3)) * (slope_m_per_m**0.5) / manning_n

        travel_time_pix = cellsize / velocity_raster

        # Extract valid per-pixel travel time along centerline
        travelv = travel_time_pix[rv, cv]
        travelv[travelv > 1e2] = 1

        travelv = uniform_filter(travelv.astype(float), size=15)

        # Cumulative time downstream from each point
        cumulative_time = np.cumsum(travelv)

        # Create empty raster and insert centerline cumulative times
        travelmap = np.full_like(dem, np.nan)
        travelmap[rv, cv] = cumulative_time

        # Interpolate across the flow mask using nearest neighbor
        _, (row_idx, col_idx) = distance_transform_edt(np.isnan(travelmap), return_indices=True)

        # Clip for safety
        row_idx = np.clip(row_idx, 0, travelmap.shape[0] - 1)
        col_idx = np.clip(col_idx, 0, travelmap.shape[1] - 1)

        # Extrapolated cumulative travel time
        travel_time_raster = travelmap[row_idx, col_idx]
        travel_time_raster = np.where(flow_mask, travel_time_raster, np.nan)


        ### Erosion calculation
        slope_radians = np.deg2rad(slope_deg)
        erosion_total = erosion_coeff * flow_depth_raster * np.sin(slope_radians) * flow_time
        if erosion_total.ndim > 2 and erosion_total.shape[0] == 1:
            erosion_total = erosion_total.squeeze(axis=0)
        erosion_total = np.where(slope_deg < 2, 0, erosion_total)
        erosion_total = np.maximum(erosion_total, ErosionTotal_prev)
        ErosionTotal_prev = erosion_total.copy()
        erosion_total = np.where(flow_mask, erosion_total, 0)

        erosion_total = np.minimum(erosion_total, ero_max)


        ### Downstream sediment adjustment
        sed_change = erosion_total * cellsize**2
        sed_centerline = sed_change[rv, cv]
        cum_ero_change = np.cumsum(sed_centerline[::-1])[::-1]

        ### Adjust volumes
        V_prev = V.copy()
        V[:len(rv)] = start_volume + cum_ero_change

        # Convergence check
        v_diff = np.abs(np.max(V) - np.max(V_prev))
        print(f"Iteration {iteration+1}, volume diff: {v_diff:.2f}")

        if v_diff < end_condition:
            print('Convergence reached.')
            break
            
    return flow_mask, flow_depth_raster, velocity_raster, travel_time_raster, erosion_total




def debrisfast_figures(dem, flow_depth_raster, travel_time_raster):
    # Normalize DEM for display
    dem_clean = np.where(np.isnan(dem), np.nanmin(dem), dem)
    dem_norm = (dem_clean - np.nanmin(dem_clean)) / (np.nanmax(dem_clean) - np.nanmin(dem_clean))

    # Generate faint topographic RGB image using terrain colormap
    topo_rgb = plt.cm.terrain(dem_norm)[..., :3]  # RGB only

    # Hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem_clean, vert_exag=1, dx=30, dy=30)

    # Blend faint hillshade into topo colors (10% intensity)
    blended = topo_rgb * (1 - 0.1) + hillshade[..., np.newaxis] * 0.25

    # Mask flow depth
    flow_masked = np.ma.masked_where(flow_depth_raster <= 0, flow_depth_raster)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(blended, extent=[0, dem.shape[1], dem.shape[0], 0])
    im = ax.imshow(flow_masked, cmap='viridis', alpha=0.6, extent=[0, dem.shape[1], dem.shape[0], 0])

    # Add contour lines
    contour_levels = np.linspace(np.nanmin(dem_clean), np.nanmax(dem_clean), 15)
    contours = ax.contour(dem_clean, levels=contour_levels, colors='k', linewidths=0.4, alpha=0.25)
    ax.clabel(contours, inline=True, fontsize=6, fmt='%d')

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Flow depth (m)")
    ax.set_title("Flow depth over hillshade and topography")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()




    #  Normalize DEM for display 
    dem_clean = np.where(np.isnan(dem), np.nanmin(dem), dem)
    dem_norm = (dem_clean - np.nanmin(dem_clean)) / (np.nanmax(dem_clean) - np.nanmin(dem_clean))

    #  Generate faint topographic RGB image using terrain colormap 
    topo_rgb = plt.cm.terrain(dem_norm)[..., :3]

    #  Hillshade 
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem_clean, vert_exag=1, dx=30, dy=30)

    #  Blend hillshade into topo colors (10% intensity) 
    blended = topo_rgb * 0.9 + hillshade[..., np.newaxis] * 0.25

    #  Convert travel time to minutes 
    travel_time_minutes = travel_time_raster / 60.0

    #  Mask non-flow areas 
    travel_masked = np.ma.masked_where(np.isnan(travel_time_minutes), travel_time_minutes)

    #  Plot 
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(blended, extent=[0, dem.shape[1], dem.shape[0], 0])
    im = ax.imshow(travel_masked, cmap='inferno', alpha=0.6, extent=[0, dem.shape[1], dem.shape[0], 0])

    # Add contour lines
    contours = ax.contour(dem_clean, levels=contour_levels, colors='k', linewidths=0.4, alpha=0.25)
    ax.clabel(contours, inline=True, fontsize=6, fmt='%d')

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Travel time (minutes)")
    ax.set_title("Travel time over hillshade and topography")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


    

def debrisfast_save(output_dir, dem_path,
                                        flow_mask, flow_depth_raster,
                                        velocity_raster, travel_time_raster,
                                        erosion_total):
#     Save debrisfast output rasters as GeoTIFFs using the input DEM for georeferencing.
#     Parameters:
#         output_dir (str): Directory to save output files.
#         dem_path (str): Path to original DEM.
#         flow_mask, flow_depth_raster, velocity_raster, travel_time_raster, erosion_total: numpy arrays

    os.makedirs(output_dir, exist_ok=True)

    # Load georeferencing from DEM
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)

    output_data = {
        'flow_mask.tif': flow_mask.astype(np.float32),
        'flow_depth.tif': flow_depth_raster.astype(np.float32),
        'velocity.tif': velocity_raster.astype(np.float32),
        'travel_time.tif': travel_time_raster.astype(np.float32),
        'erosion.tif': erosion_total.astype(np.float32)
    }

    for filename, data in output_data.items():
        path = os.path.join(output_dir, filename)
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1)
