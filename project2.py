"""
project.py - HUDF Multi-Band Image Analysis Main Script

This is the main entry point for the Hubble Ultra Deep Field (HUDF) 
multi-band image analysis project. It coordinates the workflow:
1. Resolves HUDF coordinates
2. Queries MAST for HST observations in multiple filters (F435W, F606W, F850LP)
3. Downloads the first available drizzled (_drz.fits) science image for each filter
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict
import matplotlib.pyplot as plt

# Import project modules
try:
    from query import resolve_hudf_coordinates, search_multiple_filters
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure query.py is in the same directory.")
    sys.exit(1)

# Import MAST tools for downloading
try:
    from astroquery.mast import Observations
except ImportError as e:
    print(f"Error importing astroquery: {e}")
    print("Please install astroquery: pip install astroquery")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hudf_project.log')
    ]
)

logger = logging.getLogger(__name__)


class HUDFProject:
    """
    Main project class for HUDF multi-band image analysis.
    """
    
    def __init__(self, filters: Optional[list] = None, search_radius: float = 0.1):
        """
        Initialize the HUDF project.
        
        Args:
            filters: List of filter names to search for. 
                    Default: ['F435W', 'F606W', 'F850LP']
            search_radius: Search radius in degrees for MAST queries.
        """
        self.filters = filters or ['F435W', 'F606W', 'F850LP']
        self.search_radius = search_radius
        self.hudf_coord = None
        self.observations = {}
        
        logger.info("="*60)
        logger.info("HUDF Multi-Band Image Analysis Project")
        logger.info("="*60)
        logger.info(f"Filters: {', '.join(self.filters)}")
        logger.info(f"Search radius: {self.search_radius} degrees")
    
    def step1_resolve_coordinates(self):
        """
        Step 1: Resolve HUDF field center coordinates.
        
        Raises:
            ValueError: If coordinates cannot be resolved.
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Resolving HUDF Coordinates")
        logger.info("="*60)
        
        try:
            self.hudf_coord = resolve_hudf_coordinates()
            logger.info("Coordinates resolved successfully")
            return self.hudf_coord
            
        except ValueError as e:
            logger.error("Failed to resolve coordinates")
            raise
    
    def step2_search_observations(self):
        """
        Step 2: Search MAST for HST observations in specified filters.
        
        Returns:
            Dictionary mapping filter names to observation tables.
            
        Raises:
            RuntimeError: If no coordinates have been resolved yet.
        """
        if self.hudf_coord is None:
            raise RuntimeError("Must resolve coordinates before searching observations")
        
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Searching for HST Observations")
        logger.info("="*60)
        
        try:
            self.observations = search_multiple_filters(
                self.hudf_coord,
                self.filters,
                self.search_radius
            )
            
            # Log summary
            total_obs = sum(len(obs) for obs in self.observations.values())
            logger.info(f"Search complete: {total_obs} total observations found")
            
            for filter_name, obs_table in self.observations.items():
                logger.info(f"  - {filter_name}: {len(obs_table)} observations")
            
            return self.observations
            
        except Exception as e:
            logger.error("Search failed")
            raise
    
    def step3_download_images(self):
        """
        Step 3: Download FITS images for each filter.

        Returns:
            Dictionary mapping filter names to lists of local FITS file paths.

        Raises:
            RuntimeError: If no observations have been found yet.
        """
        if not self.observations:
            raise RuntimeError("Must search for observations before downloading")

        logger.info("\n" + "="*60)
        logger.info("STEP 3: Downloading FITS Images")
        logger.info("="*60)

        # Use current directory for downloads
        download_dir = Path.cwd() / "hudf_data"
        download_dir.mkdir(exist_ok=True)
        
        # Clean up files that don't match our filter criteria
        logger.info("Cleaning up unwanted files...")
        logger.info(f"Scanning directory: {download_dir}")
        
        # Check if directory exists and has content
        if not download_dir.exists():
            logger.info("Download directory doesn't exist yet")
        else:
            all_files = list(download_dir.rglob("*"))
            logger.info(f"Found {len(all_files)} total items (files and directories)")
            
            files_removed = 0
            for filepath in all_files:
                if filepath.is_file():
                    filename = filepath.name.lower()
                    # Remove files that are NOT drizzled FITS or contain unwanted patterns
                    should_remove = False
                    
                    # Check if it's a FITS file first
                    if filename.endswith('.fits'):
                        # If it's a FITS file but NOT drizzled, remove it
                        if not (filename.endswith('_drz.fits') or filename.endswith('_drc.fits')):
                            should_remove = True
                    # Remove catalog and auxiliary files
                    elif any(x in filename for x in ['_cat.', '_point-cat.', '.ecsv', '.txt', '.xml']):
                        should_remove = True
                    
                    if should_remove:
                        try:
                            filepath.unlink()
                            logger.info(f"  Removed: {filepath.relative_to(download_dir)}")
                            files_removed += 1
                        except Exception as e:
                            logger.error(f"  Failed to remove {filepath.name}: {e}")
            
            if files_removed > 0:
                logger.info(f"Removed {files_removed} unwanted files")
            else:
                logger.info("No unwanted files found")

        fits_files = {}
        download_dir.mkdir(exist_ok=True)

        for filter_name in self.filters:
            logger.info(f"Processing {filter_name}...")

            obs_table = self.observations.get(filter_name)
            if obs_table is None or len(obs_table) == 0:
                logger.warning(f"No observations found for {filter_name}")
                continue

            fits_files[filter_name] = []

            try:
                # Get the first observation
                obs_row = obs_table[0]
                obs_id = obs_row['obsid']
                logger.info(f"  Selected observation: {obs_id}")

                # Get associated data products
                products = Observations.get_product_list(obs_row)
                
                logger.info(f"  Total products found: {len(products)}")

                # Filter for ONLY drizzled products (_drz.fits or _drc.fits)
                science_products = []
                for p in products:
                    filename = p['productFilename'].lower()
                    
                    # ONLY accept drizzled FITS files
                    if filename.endswith('_drz.fits') or filename.endswith('_drc.fits'):
                        science_products.append(p)
                        logger.info(f"    Found drizzled: {p['productFilename']}")

                if len(science_products) == 0:
                    logger.warning(f"No _drz.fits or _drc.fits files found for {filter_name}")
                    # List what WAS found to debug
                    logger.info(f"  Available products:")
                    for p in products[:5]:  # Show first 5
                        logger.info(f"    - {p['productFilename']}")
                    continue

                logger.info(f"  Found {len(science_products)} drizzled products to download")

                # Download all science products
                for idx, product in enumerate(science_products, 1):
                    filename = product['productFilename']
                    logger.info(f"  Product {idx}/{len(science_products)}: {filename}")
                    
                    # Check if file already exists in download directory
                    existing_files = list(download_dir.rglob(f"*{filename}"))
                    
                    if existing_files:
                        local_path = str(existing_files[0])
                        logger.info(f"    Already exists: {local_path}")
                        logger.info(f"    Skipping, downloading next file")
                        fits_files[filter_name].append(local_path)
                        continue
                    
                    # Download the product
                    try:
                        logger.info(f"    Downloading...")
                        manifest = Observations.download_products(
                            product,
                            download_dir=str(download_dir)
                        )

                        # Get the local path
                        local_path = manifest['Local Path'][0]
                        fits_files[filter_name].append(local_path)
                        logger.info(f"    Downloaded to: {local_path}")

                    except Exception as e:
                        logger.error(f"    Failed to download {filename}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Failed to process {filter_name}: {e}")
                continue

        # Log summary
        total_files = sum(len(files) for files in fits_files.values())
        logger.info(f"\nDownload summary:")
        for filter_name, files in fits_files.items():
            logger.info(f"  {filter_name}: {len(files)} files")
        
        if len(fits_files) < len(self.filters):
            logger.warning(f"Only downloaded files for {len(fits_files)}/{len(self.filters)} filters")
        else:
            logger.info(f"Successfully downloaded files for all {len(fits_files)} filters")

        return fits_files
    
    
    def run_full_pipeline(self):
        """
        Run the complete analysis pipeline automatically.
        No user input required - downloads data and creates RGB composite.
        """
        logger.info("\n" + "#"*60)
        logger.info("# STARTING FULL PIPELINE")
        logger.info("#"*60)
        
        try:
            # Step 1: Resolve coordinates
            self.step1_resolve_coordinates()
            
            # Step 2: Search for observations
            self.step2_search_observations()
            
            # Step 3: Download FITS images
            fits_files = self.step3_download_images()
            
            # Check if we have all required files
            missing_filters = [f for f in self.filters if f not in fits_files]
            if missing_filters:
                logger.error(f"Missing FITS files for: {', '.join(missing_filters)}")
                logger.error("Cannot create RGB composite without all three filters")
                raise RuntimeError("Incomplete data - missing required filters")
            
            logger.info("\n" + "#"*60)
            logger.info("# PIPELINE COMPLETE")
            logger.info("#"*60)
            
        except Exception as e:
            logger.error("\n" + "#"*60)
            logger.error("# PIPELINE FAILED")
            logger.error("#"*60)
            logger.error(f"Error: {e}")
            raise


def main():
    """
    Main function - automatically runs the complete pipeline.
    """
    logger.info("="*60)
    logger.info("HUDF RGB Composite Image Creator")
    logger.info("="*60)
    logger.info("This program will:")
    logger.info("  1. Resolve HUDF coordinates")
    logger.info("  2. Search for HST observations in F435W, F606W, F850LP")
    logger.info("  3. Download drizzled FITS images")
    logger.info("="*60)
    
    # Create project instance with default settings
    project = HUDFProject(
        filters=['F435W', 'F606W', 'F850LP'],
        search_radius=0.1
    )
    
    # Run the complete pipeline
    try:
        project.run_full_pipeline()
        return 0
        
    except Exception as e:
        logger.error(f"Project failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())