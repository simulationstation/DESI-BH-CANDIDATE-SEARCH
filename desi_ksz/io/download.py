"""
Data download utilities for kSZ analysis pipeline.

This module provides functions and manifests for downloading required
data products from DESI, ACT, and Planck archives.
"""

from pathlib import Path
from typing import Optional, List, Dict, Union
import urllib.request
import urllib.error
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Data Manifests
# =============================================================================

DATA_MANIFEST = {
    # DESI DR1 LSS Catalogs
    "desi_lss": {
        "base_url": "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/",
        "files": {
            "BGS_BRIGHT_N": "BGS_BRIGHT_N_clustering.dat.fits",
            "BGS_BRIGHT_S": "BGS_BRIGHT_S_clustering.dat.fits",
            "BGS_FAINT_N": "BGS_FAINT_N_clustering.dat.fits",
            "BGS_FAINT_S": "BGS_FAINT_S_clustering.dat.fits",
            "LRG_N": "LRG_N_clustering.dat.fits",
            "LRG_S": "LRG_S_clustering.dat.fits",
            "ELG_LOP_N": "ELG_LOPnotqso_N_clustering.dat.fits",
            "ELG_LOP_S": "ELG_LOPnotqso_S_clustering.dat.fits",
        },
        "randoms_pattern": "{tracer}_{region}_{i}_clustering.ran.fits",
        "n_randoms": 18,
    },

    # ACT DR6 Maps
    "act_dr6": {
        "base_url": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6.02/maps/",
        "files": {
            "f090_map": "act_dr6.02_coadd_f090_daynight_map.fits",
            "f090_ivar": "act_dr6.02_coadd_f090_daynight_ivar.fits",
            "f150_map": "act_dr6.02_coadd_f150_daynight_map.fits",
            "f150_ivar": "act_dr6.02_coadd_f150_daynight_ivar.fits",
            "f220_map": "act_dr6.02_coadd_f220_daynight_map.fits",
            "f220_ivar": "act_dr6.02_coadd_f220_daynight_ivar.fits",
        },
        "mask_url": "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6.02/masks/",
        "masks": {
            "point_source": "act_dr6.02_ps_mask.fits",
            "galactic": "act_dr6.02_gal_mask.fits",
        },
    },

    # Planck PR4 (NPIPE) Maps
    "planck_pr4": {
        "base_url": "https://portal.nersc.gov/project/cmb/planck2020/",
        "files": {
            "commander": "npipe6v20_comm_full_map_n2048.fits",
            "sevem": "npipe6v20_sevem_full_map_n2048.fits",
        },
        "alt_url": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/",
        "alt_files": {
            "commander": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
            "sevem": "COM_CMB_IQU-sevem_2048_R3.00_full.fits",
            "nilc": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
            "smica": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
        },
        "mask_url": "https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/masks/",
        "masks": {
            "common": "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits",
        },
    },
}


def download_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
    timeout: int = 300,
    max_retries: int = 3,
) -> bool:
    """
    Download a single file with retry logic.

    Parameters
    ----------
    url : str
        URL to download
    output_path : Path
        Local path to save file
    overwrite : bool
        If True, overwrite existing files
    timeout : int
        Download timeout in seconds
    max_retries : int
        Maximum number of retry attempts

    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.info(f"File exists, skipping: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading: {url}")
            logger.info(f"  -> {output_path}")

            # Download to temporary file first
            temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

            urllib.request.urlretrieve(url, temp_path)

            # Move to final location
            temp_path.rename(output_path)

            logger.info(f"Download complete: {output_path.name}")
            return True

        except urllib.error.URLError as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            continue

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    logger.error(f"Download failed after {max_retries} attempts: {url}")
    return False


def download_desi_lss(
    tracers: List[str] = ["LRG"],
    regions: List[str] = ["N", "S"],
    output_dir: Union[str, Path] = Path("data/ksz/catalogs/"),
    include_randoms: bool = True,
    n_randoms: int = 1,
    overwrite: bool = False,
) -> Dict[str, bool]:
    """
    Download DESI DR1 LSS galaxy catalogs.

    Parameters
    ----------
    tracers : list of str
        Galaxy tracers to download: BGS_BRIGHT, BGS_FAINT, LRG, ELG_LOP
    regions : list of str
        Galactic cap regions: N, S
    output_dir : Path
        Output directory for catalogs
    include_randoms : bool
        Whether to download random catalogs
    n_randoms : int
        Number of random files to download (0-17)
    overwrite : bool
        Overwrite existing files

    Returns
    -------
    dict
        Dictionary mapping filenames to download success status
    """
    output_dir = Path(output_dir)
    manifest = DATA_MANIFEST["desi_lss"]
    base_url = manifest["base_url"]

    results = {}

    for tracer in tracers:
        for region in regions:
            key = f"{tracer}_{region}"

            # Data catalog
            if key in manifest["files"]:
                filename = manifest["files"][key]
                url = f"{base_url}{filename}"
                output_path = output_dir / filename
                results[filename] = download_file(url, output_path, overwrite)
            else:
                logger.warning(f"Unknown tracer/region combination: {key}")

            # Random catalogs
            if include_randoms:
                for i in range(n_randoms):
                    filename = f"{tracer}_{region}_{i}_clustering.ran.fits"
                    url = f"{base_url}{filename}"
                    output_path = output_dir / filename
                    results[filename] = download_file(url, output_path, overwrite)

    return results


def download_act_maps(
    frequencies: List[int] = [150],
    output_dir: Union[str, Path] = Path("data/ksz/maps/"),
    include_ivar: bool = True,
    include_masks: bool = True,
    overwrite: bool = False,
) -> Dict[str, bool]:
    """
    Download ACT DR6 temperature maps.

    Parameters
    ----------
    frequencies : list of int
        Frequency bands to download: 90, 150, 220
    output_dir : Path
        Output directory for maps
    include_ivar : bool
        Whether to download inverse variance maps
    include_masks : bool
        Whether to download masks
    overwrite : bool
        Overwrite existing files

    Returns
    -------
    dict
        Dictionary mapping filenames to download success status
    """
    output_dir = Path(output_dir)
    manifest = DATA_MANIFEST["act_dr6"]
    base_url = manifest["base_url"]

    results = {}

    for freq in frequencies:
        # Temperature map
        map_key = f"f{freq:03d}_map"
        if map_key in manifest["files"]:
            filename = manifest["files"][map_key]
            url = f"{base_url}{filename}"
            output_path = output_dir / filename
            results[filename] = download_file(url, output_path, overwrite)

        # Inverse variance map
        if include_ivar:
            ivar_key = f"f{freq:03d}_ivar"
            if ivar_key in manifest["files"]:
                filename = manifest["files"][ivar_key]
                url = f"{base_url}{filename}"
                output_path = output_dir / filename
                results[filename] = download_file(url, output_path, overwrite)

    # Masks
    if include_masks:
        mask_url = manifest["mask_url"]
        for mask_name, filename in manifest["masks"].items():
            url = f"{mask_url}{filename}"
            output_path = output_dir / filename
            results[filename] = download_file(url, output_path, overwrite)

    return results


def download_planck_maps(
    components: List[str] = ["commander"],
    output_dir: Union[str, Path] = Path("data/ksz/maps/"),
    include_masks: bool = True,
    overwrite: bool = False,
    use_alt_source: bool = True,
) -> Dict[str, bool]:
    """
    Download Planck PR4/PR3 CMB temperature maps.

    Parameters
    ----------
    components : list of str
        Component separation methods: commander, sevem, nilc, smica
    output_dir : Path
        Output directory for maps
    include_masks : bool
        Whether to download masks
    overwrite : bool
        Overwrite existing files
    use_alt_source : bool
        Use alternative source (IRSA) if primary fails

    Returns
    -------
    dict
        Dictionary mapping filenames to download success status
    """
    output_dir = Path(output_dir)
    manifest = DATA_MANIFEST["planck_pr4"]

    results = {}

    for component in components:
        # Try primary source first
        if component in manifest["files"]:
            filename = manifest["files"][component]
            url = f"{manifest['base_url']}{filename}"
            output_path = output_dir / filename

            success = download_file(url, output_path, overwrite)

            # Try alternative source if primary fails
            if not success and use_alt_source and component in manifest.get("alt_files", {}):
                alt_filename = manifest["alt_files"][component]
                alt_url = f"{manifest['alt_url']}{alt_filename}"
                output_path = output_dir / alt_filename
                success = download_file(alt_url, output_path, overwrite)
                filename = alt_filename

            results[filename] = success
        else:
            logger.warning(f"Unknown Planck component: {component}")

    # Masks
    if include_masks and "mask_url" in manifest:
        for mask_name, filename in manifest.get("masks", {}).items():
            url = f"{manifest['mask_url']}{filename}"
            output_path = output_dir / filename
            results[filename] = download_file(url, output_path, overwrite)

    return results


def verify_checksum(
    file_path: Path,
    expected_md5: str,
) -> bool:
    """
    Verify file MD5 checksum.

    Parameters
    ----------
    file_path : Path
        Path to file
    expected_md5 : str
        Expected MD5 hash

    Returns
    -------
    bool
        True if checksum matches
    """
    if not file_path.exists():
        return False

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)

    actual_md5 = md5.hexdigest()
    return actual_md5.lower() == expected_md5.lower()


def check_data_availability(
    data_dir: Union[str, Path] = Path("data/ksz/"),
) -> Dict[str, Dict[str, bool]]:
    """
    Check which data files are available locally.

    Parameters
    ----------
    data_dir : Path
        Base data directory

    Returns
    -------
    dict
        Nested dictionary of data availability by source and file
    """
    data_dir = Path(data_dir)
    catalogs_dir = data_dir / "catalogs"
    maps_dir = data_dir / "maps"

    availability = {
        "desi_lss": {},
        "act_dr6": {},
        "planck_pr4": {},
    }

    # Check DESI catalogs
    desi_manifest = DATA_MANIFEST["desi_lss"]
    for key, filename in desi_manifest["files"].items():
        availability["desi_lss"][key] = (catalogs_dir / filename).exists()

    # Check ACT maps
    act_manifest = DATA_MANIFEST["act_dr6"]
    for key, filename in act_manifest["files"].items():
        availability["act_dr6"][key] = (maps_dir / filename).exists()

    # Check Planck maps
    planck_manifest = DATA_MANIFEST["planck_pr4"]
    for key, filename in planck_manifest["files"].items():
        availability["planck_pr4"][key] = (maps_dir / filename).exists()
    # Also check alt files
    for key, filename in planck_manifest.get("alt_files", {}).items():
        if key not in availability["planck_pr4"] or not availability["planck_pr4"][key]:
            availability["planck_pr4"][key] = (maps_dir / filename).exists()

    return availability


def print_data_status(data_dir: Union[str, Path] = Path("data/ksz/")) -> None:
    """Print formatted data availability status."""
    availability = check_data_availability(data_dir)

    print("\n" + "=" * 60)
    print("DATA AVAILABILITY STATUS")
    print("=" * 60)

    for source, files in availability.items():
        print(f"\n{source}:")
        for filename, available in files.items():
            status = "OK" if available else "MISSING"
            symbol = "+" if available else "-"
            print(f"  [{symbol}] {filename}: {status}")

    print("\n" + "=" * 60)
