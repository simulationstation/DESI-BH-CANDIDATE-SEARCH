"""
Tests for I/O modules.

Tests catalog loading, CMB map handling, and caching.
"""

import pytest
import numpy as np
import tempfile
import os


class TestDESICatalogs:
    """Tests for DESI catalog loading."""

    def test_catalog_dataclass(self, mock_catalog):
        """Test DESIGalaxyCatalog dataclass."""
        from desi_ksz.io import DESIGalaxyCatalog

        catalog = DESIGalaxyCatalog(
            ra=mock_catalog['ra'],
            dec=mock_catalog['dec'],
            z=mock_catalog['z'],
            weights=mock_catalog['weights'],
        )

        assert len(catalog.ra) == len(mock_catalog['ra'])
        assert catalog.positions is None  # Not computed yet

    def test_catalog_compute_positions(self, mock_catalog):
        """Test comoving position computation."""
        from desi_ksz.io import DESIGalaxyCatalog

        catalog = DESIGalaxyCatalog(
            ra=mock_catalog['ra'],
            dec=mock_catalog['dec'],
            z=mock_catalog['z'],
            weights=mock_catalog['weights'],
        )

        catalog.compute_comoving_distances()
        catalog.compute_positions()

        assert catalog.positions is not None
        assert catalog.positions.shape == (len(catalog.ra), 3)

        # Positions should be finite
        assert np.all(np.isfinite(catalog.positions))

    def test_catalog_redshift_selection(self, mock_catalog):
        """Test redshift bin selection."""
        from desi_ksz.io import DESIGalaxyCatalog

        catalog = DESIGalaxyCatalog(
            ra=mock_catalog['ra'],
            dec=mock_catalog['dec'],
            z=mock_catalog['z'],
            weights=mock_catalog['weights'],
        )

        z_min, z_max = 0.3, 0.6
        selected = catalog.select_redshift_bin(z_min, z_max)

        # Check all selected are within range
        assert np.all(selected.z >= z_min)
        assert np.all(selected.z <= z_max)

        # Should have fewer galaxies
        assert len(selected.ra) <= len(catalog.ra)


class TestCMBMaps:
    """Tests for CMB map handling."""

    def test_cmb_map_base_class(self):
        """Test CMBTemperatureMap base class."""
        from desi_ksz.io.cmb_maps import CMBTemperatureMap

        # Should not be instantiable directly
        with pytest.raises(TypeError):
            CMBTemperatureMap()

    @pytest.mark.requires_healpy
    def test_planck_map_mock(self, mock_healpix_map):
        """Test PlanckMap with mock data."""
        try:
            from desi_ksz.io import PlanckMap
        except ImportError:
            pytest.skip("healpy not available")

        # Create temporary file with mock map
        import healpy as hp

        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            hp.write_map(f.name, mock_healpix_map['data'], overwrite=True)

            map_obj = PlanckMap(f.name)

            assert map_obj.nside == mock_healpix_map['nside']
            assert len(map_obj.data) == len(mock_healpix_map['data'])

            os.unlink(f.name)

    @pytest.mark.requires_healpy
    def test_temperature_extraction(self, mock_healpix_map, mock_catalog):
        """Test temperature extraction at galaxy positions."""
        try:
            import healpy as hp
            from desi_ksz.io import PlanckMap
        except ImportError:
            pytest.skip("healpy not available")

        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            hp.write_map(f.name, mock_healpix_map['data'], overwrite=True)

            map_obj = PlanckMap(f.name)
            temps = map_obj.get_temperature_at_positions(
                mock_catalog['ra'],
                mock_catalog['dec']
            )

            assert len(temps) == len(mock_catalog['ra'])
            assert np.all(np.isfinite(temps))

            os.unlink(f.name)


class TestCache:
    """Tests for caching functionality."""

    def test_cache_creation(self):
        """Test HDF5 cache creation."""
        from desi_ksz.io import DataCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)
            assert cache is not None
            assert os.path.isdir(tmpdir)

    def test_cache_store_retrieve(self, mock_pksz_data):
        """Test storing and retrieving from cache."""
        from desi_ksz.io import DataCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            # Store data
            key = 'test_pksz'
            cache.store(key, mock_pksz_data['p_ksz'])

            # Retrieve data
            retrieved = cache.retrieve(key)

            np.testing.assert_array_equal(retrieved, mock_pksz_data['p_ksz'])

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from desi_ksz.io import DataCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            result = cache.retrieve('nonexistent_key')
            assert result is None

    def test_cache_metadata(self):
        """Test storing metadata with cached data."""
        from desi_ksz.io import DataCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DataCache(cache_dir=tmpdir)

            data = np.array([1.0, 2.0, 3.0])
            metadata = {'tracer': 'LRG', 'z_bin': '0.4-0.6'}

            cache.store('test', data, metadata=metadata)

            # Check metadata is stored
            retrieved_meta = cache.get_metadata('test')
            assert retrieved_meta['tracer'] == 'LRG'


class TestDownload:
    """Tests for download functionality."""

    def test_data_manifest_structure(self):
        """Test DATA_MANIFEST has expected structure."""
        from desi_ksz.io import DATA_MANIFEST

        assert 'desi_lss' in DATA_MANIFEST
        assert 'act_dr6' in DATA_MANIFEST
        assert 'planck_pr4' in DATA_MANIFEST

        for key in DATA_MANIFEST:
            assert 'base_url' in DATA_MANIFEST[key]
            assert 'files' in DATA_MANIFEST[key]

    def test_download_file_stub(self):
        """Test download_file function exists."""
        from desi_ksz.io import download_file

        # Just check it's callable
        assert callable(download_file)
