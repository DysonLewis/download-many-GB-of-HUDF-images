"""
query.py - HUDF Coordinate Resolution and HST Image Search

This module handles coordinate resolution for the Hubble Ultra Deep Field (HUDF)
and queries the MAST archive for HST observations in specified filter bandpasses.
"""

import logging
from typing import List, Optional, Tuple
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.mast import Observations
from astropy.table import Table

Observations.MAST_TIMEOUT = 60  # sets timeout globally

# Configure module logger
logger = logging.getLogger(__name__)


def resolve_hudf_coordinates() -> SkyCoord:
    """
    Resolve coordinates for the Hubble Ultra Deep Field using Astropy/Sesame.
    
    Returns:
        SkyCoord: Coordinates of the HUDF field center.
        
    Raises:
        ValueError: If coordinates cannot be resolved.
    """
    logger.info("Resolving 'HUDF' coordinates via Astropy (Sesame)...")
    
    try:
        hudf_coord = SkyCoord.from_name("HUDF")
        logger.info(f"Resolved HUDF coordinates:")
        logger.info(f"  RA  = {hudf_coord.ra.to_string(unit=u.hour, sep=':')}")
        logger.info(f"  Dec = {hudf_coord.dec.to_string(unit=u.deg, sep=':')}")
        logger.info(f"  RA  = {hudf_coord.ra.deg:.6f}°, Dec = {hudf_coord.dec.deg:.6f}°")
        return hudf_coord
        
    except Exception as e:
        error_msg = f"Could not resolve 'HUDF' coordinates: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def find_hst_images(
    coordinates: SkyCoord,
    filter_name: str,
    radius_deg: float = 0.1,
    max_results: Optional[int] = 10
) -> Table:
    """
    Query MAST for HST observations near specified coordinates containing a filter.
    Returns observations that have drizzled (_drz.fits or _drc.fits) products.
    
    Args:
        coordinates: Target sky coordinates for the search.
        filter_name: Name of the filter to search for (e.g., 'F435W', 'F850LP;CLEAR2L').
        radius_deg: Search radius in degrees. Default is 0.1 degrees.
        max_results: Maximum number of results to return. None for all results.
        
    Returns:
        Table: Astropy table of observations matching the filter criteria.
        
    Raises:
        TypeError: If coordinates is not a SkyCoord object.
        ValueError: If filter_name is empty or radius_deg is non-positive.
        RuntimeError: If no drizzled products are found.
    """
    if not isinstance(coordinates, SkyCoord):
        raise TypeError("coordinates must be a SkyCoord object")
    
    if not filter_name or not isinstance(filter_name, str):
        raise ValueError("filter_name must be a non-empty string")
    
    if radius_deg <= 0:
        raise ValueError("radius_deg must be positive")
    
    logger.info(f"Searching for HST images with filter '{filter_name}'...")
    logger.debug(f"  Search radius: {radius_deg} degrees")
    
    try:
        import time
        max_retries = 3
        retry_delay = 2
        
        obs = None
        for attempt in range(max_retries):
            try:
                logger.debug(f"  Query attempt {attempt + 1}/{max_retries}")
                obs = Observations.query_criteria(
                    coordinates=coordinates,
                    radius=f"{radius_deg} deg",
                    obs_collection="HST",
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"  Query attempt {attempt + 1} failed: {e}")
                    logger.warning(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise
        
        if obs is None:
            raise RuntimeError("All query attempts failed")
        
        # Filter by requested filter
        mask = [(filter_name.upper() in str(filt).upper()) if filt is not None else False
                for filt in obs["filters"]]
        filtered_obs = obs[mask]
        
        if max_results is not None and len(filtered_obs) > max_results:
            filtered_obs = filtered_obs[:max_results]
        
        logger.info(f"  Found {len(filtered_obs)} observations with filter '{filter_name}'")
        if len(filtered_obs) > 0:
            instruments = sorted(set(filtered_obs["instrument_name"]))
            logger.info(f"  Instruments: {', '.join(instruments)}")
        else:
            logger.warning(f"  No matching observations found for filter '{filter_name}'")
        
        # Filter for observations that have drizzled products
        obs_with_drizzled = []
        for obs_row in filtered_obs:
            products = Observations.get_product_list(obs_row)
            # Check if any drizzled products exist
            has_drizzled = any(
                p['productFilename'].lower().endswith('_drz.fits') or 
                p['productFilename'].lower().endswith('_drc.fits')
                for p in products
            )
            if has_drizzled:
                obs_with_drizzled.append(obs_row)
        
        if not obs_with_drizzled:
            raise RuntimeError(f"No drizzled products (_drz.fits or _drc.fits) found for filter '{filter_name}'")
        
        obs_with_drizzled_table = Table(rows=obs_with_drizzled, names=obs_with_drizzled[0].colnames)
        logger.info(f"  {len(obs_with_drizzled_table)} observations have drizzled products")
        return obs_with_drizzled_table
    
    except Exception as e:
        error_msg = f"Error querying MAST for filter '{filter_name}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def search_multiple_filters(
    coordinates: SkyCoord,
    filter_list: List[str],
    radius_deg: float = 0.1
) -> dict:
    """
    Search for HST observations in multiple filter bandpasses.
    
    Args:
        coordinates: Target sky coordinates for the search.
        filter_list: List of filter names to search for.
        radius_deg: Search radius in degrees. Default is 0.1 degrees.
        
    Returns:
        dict: Dictionary mapping filter names to observation tables.
        
    Raises:
        ValueError: If filter_list is empty or contains invalid entries.
    """
    if not filter_list:
        raise ValueError("filter_list must contain at least one filter name")
    
    if not all(isinstance(f, str) and f for f in filter_list):
        raise ValueError("All filter names must be non-empty strings")
    
    logger.info(f"Beginning search for {len(filter_list)} filters: {', '.join(filter_list)}")
    
    results = {}
    
    for i, filter_name in enumerate(filter_list, 1):
        try:
            logger.info(f"Filter {i}/{len(filter_list)}: {filter_name}")
            obs_table = find_hst_images(coordinates, filter_name, radius_deg)
            results[filter_name] = obs_table
            
            # Small delay between queries to avoid rate limiting
            if i < len(filter_list):
                import time
                time.sleep(0.1)
            
        except Exception as e:
            # Log the error but continue with other filters
            logger.error(f"Failed to search for filter '{filter_name}': {e}")
            logger.info(f"Continuing with remaining filters...")
            results[filter_name] = Table()  # Empty table for failed searches
    
    logger.info("Filter search complete")
    return results


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the module
    try:
        hudf_coord = resolve_hudf_coordinates()
        filters = ["F435W", "F606W", "F850LP"]
        results = search_multiple_filters(hudf_coord, filters)
        
        print("\nSearch Summary:")
        for filt, obs_table in results.items():
            print(f"  {filt}: {len(obs_table)} observations")
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise