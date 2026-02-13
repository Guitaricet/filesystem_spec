"""Tests for HNS (Hierarchical Namespace) support in gcsfs mkdir operations.

These tests verify that HNSGCSFileSystem._mkdir correctly handles HNS-enabled
buckets by creating real directory objects via the GCS Folders API, while
preserving the existing no-op behavior for flat-namespace buckets.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Skip all tests if gcsfs is not importable
gcsfs = pytest.importorskip("gcsfs")

from gcsfs.retry import HttpError
from fsspec.implementations.gcsfs_hns import HNSGCSFileSystem


def _make_fs():
    """Create an HNSGCSFileSystem with mocked credentials (no real GCS connection)."""
    with patch("gcsfs.core.GoogleCredentials"):
        fs = HNSGCSFileSystem(project="test-project", token="anon")
    # Mock the session so _call doesn't try to make real HTTP requests
    fs._session = MagicMock()
    return fs


class TestIsHnsEnabled:
    """Tests for HNSGCSFileSystem._is_hns_enabled."""

    @pytest.mark.asyncio
    async def test_hns_enabled_bucket(self):
        fs = _make_fs()
        fs._call = AsyncMock(
            return_value={"name": "hns-bucket", "hierarchicalNamespace": {"enabled": True}}
        )
        result = await fs._is_hns_enabled("hns-bucket")
        assert result is True
        fs._call.assert_called_once_with("GET", "b/hns-bucket", json_out=True)

    @pytest.mark.asyncio
    async def test_hns_disabled_bucket(self):
        fs = _make_fs()
        fs._call = AsyncMock(
            return_value={"name": "flat-bucket", "hierarchicalNamespace": {"enabled": False}}
        )
        result = await fs._is_hns_enabled("flat-bucket")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_hns_field(self):
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "old-bucket"})
        result = await fs._is_hns_enabled("old-bucket")
        assert result is False

    @pytest.mark.asyncio
    async def test_caching(self):
        fs = _make_fs()
        fs._call = AsyncMock(
            return_value={"name": "hns-bucket", "hierarchicalNamespace": {"enabled": True}}
        )
        # First call queries the API
        result1 = await fs._is_hns_enabled("hns-bucket")
        assert result1 is True
        assert fs._call.call_count == 1

        # Second call should use cache, not call API again
        result2 = await fs._is_hns_enabled("hns-bucket")
        assert result2 is True
        assert fs._call.call_count == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_caching_false_result(self):
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "flat-bucket"})
        result1 = await fs._is_hns_enabled("flat-bucket")
        assert result1 is False
        # False results should also be cached
        result2 = await fs._is_hns_enabled("flat-bucket")
        assert result2 is False
        assert fs._call.call_count == 1

    @pytest.mark.asyncio
    async def test_api_error_returns_false(self):
        fs = _make_fs()
        fs._call = AsyncMock(side_effect=OSError("Forbidden"))
        result = await fs._is_hns_enabled("private-bucket")
        assert result is False

    @pytest.mark.asyncio
    async def test_api_error_is_cached(self):
        fs = _make_fs()
        fs._call = AsyncMock(side_effect=OSError("Forbidden"))
        result1 = await fs._is_hns_enabled("private-bucket")
        assert result1 is False
        # Error result should be cached as False
        result2 = await fs._is_hns_enabled("private-bucket")
        assert result2 is False
        assert fs._call.call_count == 1

    @pytest.mark.asyncio
    async def test_different_buckets_cached_independently(self):
        fs = _make_fs()

        async def mock_call(method, path, **kwargs):
            if "hns-bucket" in path:
                return {"name": "hns-bucket", "hierarchicalNamespace": {"enabled": True}}
            return {"name": "flat-bucket"}

        fs._call = AsyncMock(side_effect=mock_call)
        assert await fs._is_hns_enabled("hns-bucket") is True
        assert await fs._is_hns_enabled("flat-bucket") is False
        assert fs._call.call_count == 2


class TestMkdirHns:
    """Tests for HNSGCSFileSystem._mkdir_hns."""

    @pytest.mark.asyncio
    async def test_create_folder(self):
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "folders/mydir"})
        await fs._mkdir_hns("mybucket/mydir", "mybucket", "mydir", create_parents=False)
        fs._call.assert_called_once_with(
            "POST",
            "b/{}/folders",
            "mybucket",
            folder="mydir",
            recursive="false",
            json_out=True,
        )

    @pytest.mark.asyncio
    async def test_create_folder_with_parents(self):
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "folders/a/b/c"})
        await fs._mkdir_hns(
            "mybucket/a/b/c", "mybucket", "a/b/c", create_parents=True
        )
        fs._call.assert_called_once_with(
            "POST",
            "b/{}/folders",
            "mybucket",
            folder="a/b/c",
            recursive="true",
            json_out=True,
        )

    @pytest.mark.asyncio
    async def test_strips_trailing_slash_from_key(self):
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "folders/mydir"})
        await fs._mkdir_hns("mybucket/mydir/", "mybucket", "mydir/", create_parents=False)
        # The folder parameter should have the trailing slash stripped
        fs._call.assert_called_once_with(
            "POST",
            "b/{}/folders",
            "mybucket",
            folder="mydir",
            recursive="false",
            json_out=True,
        )

    @pytest.mark.asyncio
    async def test_folder_already_exists_409(self):
        fs = _make_fs()
        error = HttpError({"code": 409, "message": "Folder already exists"})
        fs._call = AsyncMock(side_effect=error)
        # Should not raise - folder already exists is OK
        await fs._mkdir_hns("mybucket/mydir", "mybucket", "mydir", create_parents=False)

    @pytest.mark.asyncio
    async def test_precondition_failure_without_create_parents(self):
        fs = _make_fs()
        # 412 is mapped to FileExistsError by validate_response, but for
        # the Folders API it means the parent doesn't exist
        fs._call = AsyncMock(side_effect=FileExistsError("mybucket/a/b"))
        with pytest.raises(FileNotFoundError, match="Parent directory does not exist"):
            await fs._mkdir_hns(
                "mybucket/a/b", "mybucket", "a/b", create_parents=False
            )

    @pytest.mark.asyncio
    async def test_precondition_failure_with_create_parents_raises(self):
        fs = _make_fs()
        fs._call = AsyncMock(side_effect=FileExistsError("mybucket/a/b"))
        with pytest.raises(FileExistsError):
            await fs._mkdir_hns(
                "mybucket/a/b", "mybucket", "a/b", create_parents=True
            )

    @pytest.mark.asyncio
    async def test_other_http_error_propagates(self):
        fs = _make_fs()
        error = HttpError({"code": 500, "message": "Internal server error"})
        fs._call = AsyncMock(side_effect=error)
        with pytest.raises(HttpError):
            await fs._mkdir_hns("mybucket/mydir", "mybucket", "mydir")

    @pytest.mark.asyncio
    async def test_invalidates_parent_cache(self):
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "folders/a/b"})
        fs.invalidate_cache = MagicMock()
        await fs._mkdir_hns("mybucket/a/b", "mybucket", "a/b")
        # Should invalidate the parent directory's cache
        fs.invalidate_cache.assert_called_once_with("mybucket/a")


class TestMkdirWithHns:
    """Tests for HNSGCSFileSystem._mkdir with HNS support."""

    @pytest.mark.asyncio
    async def test_sub_bucket_path_hns_bucket(self):
        """mkdir on sub-bucket path for HNS bucket should create the folder."""
        fs = _make_fs()
        fs._exists = AsyncMock(return_value=True)
        fs._is_hns_enabled = AsyncMock(return_value=True)
        fs._mkdir_hns = AsyncMock()

        await fs._mkdir("mybucket/mydir", create_parents=False)

        fs._is_hns_enabled.assert_called_once_with("mybucket")
        fs._mkdir_hns.assert_called_once_with(
            "mybucket/mydir", "mybucket", "mydir", False
        )

    @pytest.mark.asyncio
    async def test_sub_bucket_path_flat_bucket(self):
        """mkdir on sub-bucket path for non-HNS bucket should be a no-op."""
        fs = _make_fs()
        fs._exists = AsyncMock(return_value=True)
        fs._is_hns_enabled = AsyncMock(return_value=False)
        fs._mkdir_hns = AsyncMock()

        await fs._mkdir("mybucket/mydir", create_parents=False)

        fs._is_hns_enabled.assert_called_once_with("mybucket")
        fs._mkdir_hns.assert_not_called()

    @pytest.mark.asyncio
    async def test_sub_bucket_path_create_parents_hns(self):
        """makedirs on sub-bucket path for HNS bucket creates folders recursively."""
        fs = _make_fs()
        fs._exists = AsyncMock(return_value=True)
        fs._is_hns_enabled = AsyncMock(return_value=True)
        fs._mkdir_hns = AsyncMock()

        await fs._mkdir("mybucket/a/b/c", create_parents=True)

        fs._mkdir_hns.assert_called_once_with(
            "mybucket/a/b/c", "mybucket", "a/b/c", True
        )

    @pytest.mark.asyncio
    async def test_sub_bucket_path_create_parents_flat(self):
        """makedirs on sub-bucket path for non-HNS bucket is still a no-op."""
        fs = _make_fs()
        fs._exists = AsyncMock(return_value=True)
        fs._is_hns_enabled = AsyncMock(return_value=False)
        fs._mkdir_hns = AsyncMock()

        await fs._mkdir("mybucket/a/b/c", create_parents=True)

        fs._mkdir_hns.assert_not_called()

    @pytest.mark.asyncio
    async def test_sub_bucket_bucket_not_found_no_create_parents(self):
        """mkdir with non-existent bucket and create_parents=False raises."""
        fs = _make_fs()
        fs._exists = AsyncMock(return_value=False)

        with pytest.raises(FileNotFoundError):
            await fs._mkdir("nonexistent/dir", create_parents=False)

    @pytest.mark.asyncio
    async def test_sub_bucket_bucket_not_found_create_parents(self):
        """mkdir with non-existent bucket and create_parents=True creates bucket."""
        fs = _make_fs()
        fs._exists = AsyncMock(return_value=False)
        fs._call = AsyncMock(return_value={"name": "nonexistent"})

        await fs._mkdir("nonexistent/dir", create_parents=True)

        # Should have called POST to create the bucket
        fs._call.assert_called_once()
        call_kwargs = fs._call.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["path"] == "b"
        assert call_kwargs["json"] == {"name": "nonexistent"}
        assert call_kwargs["json_out"] is True

    @pytest.mark.asyncio
    async def test_bucket_only_path_creates_bucket(self):
        """mkdir with just a bucket name creates the bucket."""
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "newbucket"})

        await fs._mkdir("newbucket")

        fs._call.assert_called_once()
        call_kwargs = fs._call.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["path"] == "b"
        assert call_kwargs["json"] == {"name": "newbucket"}
        assert call_kwargs["json_out"] is True

    @pytest.mark.asyncio
    async def test_empty_bucket_raises(self):
        """mkdir with empty path raises ValueError."""
        fs = _make_fs()
        with pytest.raises(ValueError, match="Cannot create root bucket"):
            await fs._mkdir("")

    @pytest.mark.asyncio
    async def test_hns_not_checked_for_bucket_only_path(self):
        """HNS check should not be performed for bucket-only paths."""
        fs = _make_fs()
        fs._call = AsyncMock(return_value={"name": "mybucket"})
        fs._is_hns_enabled = AsyncMock()

        await fs._mkdir("mybucket")

        fs._is_hns_enabled.assert_not_called()

    @pytest.mark.asyncio
    async def test_hns_cache_populated_in_init(self):
        """HNSGCSFileSystem.__init__ should initialize the HNS cache."""
        fs = _make_fs()
        assert hasattr(fs, "_hns_enabled_cache")
        assert fs._hns_enabled_cache == {}


class TestMkdirHnsIntegration:
    """Integration-style tests combining _mkdir with _is_hns_enabled."""

    @pytest.mark.asyncio
    async def test_mkdir_exists_roundtrip_hns(self):
        """mkdir followed by exists should work on HNS buckets.

        This is the core issue: previously mkdir was a no-op, so exists
        would return False. With HNS support, mkdir creates the folder
        and exists should return True.
        """
        fs = _make_fs()

        # Mock: bucket exists and is HNS-enabled
        fs._exists = AsyncMock(return_value=True)
        call_results = [
            # First call: GET bucket metadata (HNS check)
            {"name": "mybucket", "hierarchicalNamespace": {"enabled": True}},
            # Second call: POST folders (create folder)
            {"name": "folders/mydir"},
        ]
        fs._call = AsyncMock(side_effect=call_results)

        # mkdir should create the folder (not be a no-op)
        await fs._mkdir("mybucket/mydir", create_parents=False)

        # Verify the Folders API was called
        assert fs._call.call_count == 2
        # First call: HNS check
        fs._call.assert_any_call("GET", "b/mybucket", json_out=True)
        # Second call: create folder
        fs._call.assert_any_call(
            "POST",
            "b/{}/folders",
            "mybucket",
            folder="mydir",
            recursive="false",
            json_out=True,
        )

    @pytest.mark.asyncio
    async def test_repeated_mkdir_uses_cached_hns_status(self):
        """Multiple mkdir calls should cache the HNS status."""
        fs = _make_fs()
        fs._exists = AsyncMock(return_value=True)

        call_count = 0

        async def mock_call(method_or_method_kw=None, path=None, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if method_or_method_kw == "GET" and path and path.startswith("b/"):
                return {"name": "mybucket", "hierarchicalNamespace": {"enabled": True}}
            return {"name": "folders/dir"}

        fs._call = AsyncMock(side_effect=mock_call)

        # First mkdir: HNS check + folder creation = 2 calls
        await fs._mkdir("mybucket/dir1", create_parents=False)
        assert call_count == 2

        # Second mkdir: no HNS check (cached) + folder creation = 1 call
        await fs._mkdir("mybucket/dir2", create_parents=False)
        assert call_count == 3  # Only 1 more call (folder creation only)
