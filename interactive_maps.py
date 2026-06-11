"""
Interactive Map Functions for Elevation From DEMs
==============================================
Provides interactive leaflet maps for selecting points and transects.
Uses ipyleaflet for map display in Jupyter notebooks.
"""

import time
from IPython.display import clear_output, display
from ipyleaflet import (
    Map, Marker, Polyline, FullScreenControl,
    GeoJSON, WidgetControl, AwesomeIcon, basemaps
)
from ipywidgets import HTML, Button, HBox, Layout, VBox, widgets
from pyproj import Transformer
from rasterio import warp
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from geopy.distance import distance as geopy_distance

from elevation_utils import transform_bounds_to_wgs84


def select_point_interactive(timeout=2):
    """
    Interactive map to select a single point with draggable marker.
    
    Displays a leaflet map where users can drag a marker and click
    'Confirm' to select coordinates.
    
    Parameters
    ----------
    timeout : int
        Maximum seconds to wait for user confirmation
        
    Returns
    -------
    tuple
        (longitude, latitude) in WGS84
    """
    global coords
    coords = (-54.2, 75)
    mid_lat = 75.0
    mid_lon = -54.2

    # Create a leaflet map widget to find an AOI for the spatial query of the STAC API
    m = Map(
        basemap=basemaps.Esri.WorldImagery,
        scroll_wheel_zoom=True,
        center=(mid_lat, mid_lon),
        zoom=6,
        layout=Layout(height="380px", width="700px"),
    )
    m.add_control(FullScreenControl())
    result_output = widgets.Output()

    # Create info display
    info_html = HTML(
        value="<b>Drag the marker, then click Confirm</b>",
        layout=Layout(padding="10px"),
    )

    # Create confirmation button
    confirm_button = Button(
        description="Confirm Location",
        button_style="success",
        disabled=False,
        tooltip="Click after placing marker",
    )

    # Create markers (initially slightly offset from center)
    centre_marker = Marker(
        # location=(center_y, center_x - 0.1),
        location=(mid_lat, mid_lon),  # Leaflet expects (lat, lon)
        draggable=True,
        name="Location",
        icon=AwesomeIcon(name="map-pin", marker_color="red"),
    )
    m.add_layer(centre_marker)

    # Update info display when marker moves
    def update_display(*args):
        # Remember: Leaflet uses (lat, lon) format, but we want to store as (lon, lat)
        centre_lat, centre_lon = centre_marker.location[0], centre_marker.location[1]
        info_html.value = (
            f"<b>Current Position:</b><br>"
            f"<span style='color:red'>Location:</span> Lon: {centre_lon:.3f}, Lat: {centre_lat:.3f}<br>"
        )

    # Handle confirmation
    def on_confirm(b):
        global coords
        # Store as (lon, lat) for consistency with GIS conventions
        centre_lat, centre_lon = centre_marker.location[0], centre_marker.location[1]
        with result_output:
            result_output.clear_output()
            print(f"Centre confirmed! (lon, lat): {centre_lon}, {centre_lat}")
            global coords
            coords = (centre_lon, centre_lat)

    # Connect callbacks
    confirm_button.on_click(on_confirm)
    centre_marker.observe(update_display, names=["location"])
    update_display()  # Initial display update

    # Display all components
    display(VBox([info_html, m, HBox([confirm_button]), result_output]))

    # Wait for confirmation with timeout
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        time.sleep(0.1)

    clear_output(wait=True)  # Clean up the display
    return coords



def select_transect_interactive(# initial_start=(-54.25, 75.025), 
                            #    initial_end=(-54.35, 75.025),
                            #    zoom=6, 
                            timeout=30,
                            raster_data=None, raster_metadata=None, mask=None):
    """Interactive map to select a transect with two draggable markers.
    
    Displays a leaflet map where users can drag start and end markers
    to define a transect. Optionally overlays raster footprint for context.
    
    Parameters
    ----------
    timeout : int
        Maximum seconds to wait for user confirmation
    raster_data : ndarray, optional
        2D raster array for displaying footprint
    raster_metadata : dict, optional
        Dictionary with 'bounds', 'transform', 'crs' keys
    mask : ndarray, optional
        Binary mask for valid raster areas (1=valid, 0=invalid)
        
    Returns
    -------
    tuple
        ((start_lon, start_lat), (end_lon, end_lat)) in WGS84
    """
    global coords
    coords = (-54.2, 75, -54.3, 75.05)
    center_lon = -54.25
    center_lat = 75.025
    mid_lat = center_lat
    start_lon = coords[0] + (coords[2] - coords[0]) / 3
    end_lon = coords[0] + 2 * (coords[2] - coords[0]) / 3

    # Create a leaflet map widget to find an AOI for the spatial query of the STAC API
    m = Map(
        basemap=basemaps.Esri.WorldImagery,
        scroll_wheel_zoom=True,
        center=(center_lat, center_lon),
        zoom=6,
        layout=Layout(height="380px", width="700px"),
    )
    m.add_control(FullScreenControl())
    
    # If raster data is provided, overlay its footprint
    if raster_data is not None and raster_metadata is not None and mask is not None:
        _add_raster_footprint_to_map(m, raster_metadata, mask)
        
        # Update center based on raster bounds
        bounds_wgs84 = transform_bounds_to_wgs84(
            raster_metadata['bounds'], 
            raster_metadata['crs']
        )
        center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2
        center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
        m.center = (center_lat, center_lon)
        
        # Set initial marker positions at 1/3 and 2/3 of raster width
        start_lon = bounds_wgs84[0] + (bounds_wgs84[2] - bounds_wgs84[0]) / 3
        end_lon = bounds_wgs84[0] + 2 * (bounds_wgs84[2] - bounds_wgs84[0]) / 3
        start_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
        end_lat = start_lat
    
    # Create widgets
    result_output = widgets.Output()
    
    info_html = HTML(
        value="<b>Drag the markers, then click Confirm</b>",
        layout=Layout(padding="10px"),
    )
    
    confirm_button = Button(
        description="Confirm Selected Points",
        button_style="success",
        disabled=False,
        tooltip="Click after placing markers",
    )
    
    # Create start marker (red play icon)
    start_marker = Marker(
        location=(mid_lat, start_lon),
        draggable=True,
        name="Start (A)",
        icon=AwesomeIcon(name="play", marker_color="red"),
    )
    
    # Create end marker (blue stop icon)
    end_marker = Marker(
        location=(mid_lat, end_lon),
        draggable=True,
        name="End (B)",
        icon=AwesomeIcon(name="stop", marker_color="blue"),
    )
    
    m.add_layer(start_marker)
    m.add_layer(end_marker)
    
    # Create line connecting markers
    line = Polyline(
        locations=[(mid_lat, start_lon), (mid_lat, end_lon)],
        color="purple",
        weight=3
    )
    m.add_layer(line)
    
    # Update line when markers move
    def update_line(*args):
        line.locations = [
            (start_marker.location[0], start_marker.location[1]),
            (end_marker.location[0], end_marker.location[1]),
        ]

    start_marker.observe(update_line, names=["location"])
    end_marker.observe(update_line, names=["location"])

    # Update info display
    def update_display(*args):
        s_lat, s_lon = start_marker.location[0], start_marker.location[1]
        e_lat, e_lon = end_marker.location[0], end_marker.location[1]
        
        # Calculate distance
        dist = geopy_distance((s_lat, s_lon), (e_lat, e_lon)).kilometers
        
        info_html.value = (
            f"<b>Current Positions:</b><br>"
            f"<span style='color:red'>Start (A):</span> "
            f"Lon: {s_lon:.3f}, Lat: {s_lat:.3f}<br>"
            f"<span style='color:blue'>End (B):</span> "
            f"Lon: {e_lon:.3f}, Lat: {e_lat:.3f}<br>"
            f"<span style='color:purple'>Distance:</span> {dist:.2f} km"
        )
    
    # Handle confirmation
    def on_confirm(b):
        global coords
        # Store as (lon, lat) for consistency with GIS conventions
        start_coords = (start_marker.location[1], start_marker.location[0])
        end_coords = (end_marker.location[1], end_marker.location[0])
        with result_output:
            result_output.clear_output()
            print(
                f"Points confirmed!\n\
                    Start (lon, lat): {start_coords[0]:.3f}, {start_coords[1]:.3f}\n\
                    End (lon, lat): {end_coords[0]:.3f}, {end_coords[1]:.3f}"
            )
            # Store results in global variable
            global coords
            coords = (start_coords, end_coords)

    # Connect callbacks
    confirm_button.on_click(on_confirm)
    # start_marker.observe(update_line, names=["location"])
    # end_marker.observe(update_line, names=["location"])
    start_marker.observe(update_display, names=["location"])
    end_marker.observe(update_display, names=["location"])
    update_display()  # Initial display
    
    # Show the map
    display(VBox([info_html, m, HBox([confirm_button]), result_output]))
    
    # Wait for confirmation
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    clear_output(wait=True)
    
    # if selected_points[0] is None:
    #     print(f"⏰ Timeout after {timeout}s. Using default transect.")
    #     return (initial_start, initial_end)
    
    return coords


def _add_raster_footprint_to_map(m, raster_metadata, mask):
    """Add raster footprint as a GeoJSON layer to the map.
    
    Parameters
    ----------
    m : ipyleaflet.Map
        Map to add the layer to
    raster_metadata : dict
        Dictionary with 'bounds', 'transform', 'crs' keys
    mask : ndarray
        Binary mask (1=valid, 0=invalid)
    """
    transform = raster_metadata['transform']
    crs_src = raster_metadata['crs']
    
    # Extract shapes from mask (this can be slow for large rasters)
    shapes_gen = shapes(mask, transform=transform)
    polygons = [shape(geom) for geom, val in shapes_gen if val == 1]
    
    if not polygons:
        print("Warning: No valid polygons found in raster footprint")
        return
    
    # Add each polygon as a GeoJSON layer
    for i, poly in enumerate(polygons):
        # Transform to WGS84 for display
        wgs84_poly = warp.transform_geom(crs_src, "EPSG:4326", mapping(poly))
        
        geojson_poly = GeoJSON(
            data=wgs84_poly,
            style={
                "color": "lime",
                "fillColor": "lime",
                "opacity": 0.8,
                "fillOpacity": 0.2,
                "weight": 2,
            },
            name=f"Raster Boundary {i+1}",
        )
        m.add_layer(geojson_poly)


def select_point_with_raster_context(raster_data, raster_metadata, mask, 
                                    timeout=30):
    """Select a point on a map with raster footprint for context.
    
    Useful when you want to see the DEM extent while selecting coordinates.
    
    Parameters
    ----------
    raster_data : ndarray
        2D raster array
    raster_metadata : dict
        Dictionary with 'bounds', 'transform', 'crs', 'nodata' keys
    mask : ndarray
        Binary mask for valid data (1=valid)
    timeout : int
        Maximum seconds to wait
        
    Returns
    -------
    tuple
        (lon, lat) in WGS84
        
    Examples
    --------
    >>> with rio.open('dem.tif') as src:
    ...     data = src.read(1)
    ...     mask = (data != src.nodata).astype(int)
    ...     metadata = {
    ...         'bounds': src.bounds,
    ...         'transform': src.transform,
    ...         'crs': src.crs,
    ...     }
    >>> coords = select_point_with_raster_context(data, metadata, mask)
    """
    # Calculate center from raster bounds
    bounds_wgs84 = transform_bounds_to_wgs84(
        raster_metadata['bounds'], 
        raster_metadata['crs']
    )
    center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2
    center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
    
    # Use the transect selection but with both markers
    # (we'll just use the start marker's position)
    transect = select_transect_interactive(
        initial_start=(center_lon - 0.05, center_lat),
        initial_end=(center_lon + 0.05, center_lat),
        raster_data=raster_data,
        raster_metadata=raster_metadata,
        mask=mask,
        timeout=timeout
    )
    
    # Return just the start point
    return transect[0]


def select_transect_with_raster_context(raster_data, raster_metadata, mask,
                                       timeout=30):
    """Select a transect on a map with raster footprint for context.
    
    Parameters
    ----------
    raster_data : ndarray
        2D raster array
    raster_metadata : dict
        Dictionary with 'bounds', 'transform', 'crs', 'nodata' keys
    mask : ndarray
        Binary mask for valid data (1=valid)
    timeout : int
        Maximum seconds to wait
        
    Returns
    -------
    tuple
        ((start_lon, start_lat), (end_lon, end_lat))
        
    Examples
    --------
    >>> with rio.open('dem.tif') as src:
    ...     data = src.read(1)
    ...     mask = (data != src.nodata).astype(int)
    ...     metadata = {
    ...         'bounds': src.bounds,
    ...         'transform': src.transform,
    ...         'crs': src.crs,
    ...     }
    >>> transect = select_transect_with_raster_context(data, metadata, mask)
    """
    bounds_wgs84 = transform_bounds_to_wgs84(
        raster_metadata['bounds'], 
        raster_metadata['crs']
    )
    center_lon = (bounds_wgs84[0] + bounds_wgs84[2]) / 2
    center_lat = (bounds_wgs84[1] + bounds_wgs84[3]) / 2
    
    # Set initial markers at 1/3 and 2/3 of raster width
    width = bounds_wgs84[2] - bounds_wgs84[0]
    start_lon = bounds_wgs84[0] + width / 3
    end_lon = bounds_wgs84[0] + 2 * width / 3
    
    return select_transect_interactive(
        initial_start=(start_lon, center_lat),
        initial_end=(end_lon, center_lat),
        raster_data=raster_data,
        raster_metadata=raster_metadata,
        mask=mask,
        timeout=timeout
    )