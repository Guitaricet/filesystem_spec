"""GCSFileSystem subclass with HNS (Hierarchical Namespace) support.

GCS buckets with HNS enabled support real, first-class directory objects
via the Folders API. The base GCSFileSystem treats all sub-bucket mkdir
calls as no-ops, which breaks workflows that depend on mkdir + exists
roundtripping correctly on HNS-enabled buckets.

This subclass overrides _mkdir to detect HNS-enabled buckets and create
actual directory objects via the GCS Folders API, while preserving the
existing no-op behavior for flat-namespace buckets.

Usage::

    from fsspec.implementations.gcsfs_hns import HNSGCSFileSystem
    fs = HNSGCSFileSystem(project="my-project")
    fs.mkdir("my-hns-bucket/some/directory", create_parents=True)

See: https://cloud.google.com/storage/docs/hns-overview
     https://cloud.google.com/storage/docs/json_api/v1/folders
"""

import logging

from gcsfs.core import GCSFileSystem
from gcsfs.retry import HttpError
from fsspec import asyn

logger = logging.getLogger("gcsfs")


class HNSGCSFileSystem(GCSFileSystem):
    """GCSFileSystem with Hierarchical Namespace (HNS) support.

    Extends GCSFileSystem so that mkdir on sub-bucket paths creates real
    directory objects for HNS-enabled buckets via the GCS Folders API.
    Non-HNS buckets retain the standard no-op behavior.

    The bucket's HNS status is cached after the first lookup since it
    cannot change after bucket creation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hns_enabled_cache = {}

    async def _is_hns_enabled(self, bucket):
        """Check if a bucket has Hierarchical Namespace (HNS) enabled.

        Results are cached since a bucket's HNS status does not change.

        Parameters
        ----------
        bucket : str
            Name of the bucket to check.

        Returns
        -------
        bool
            True if the bucket has HNS enabled, False otherwise.
        """
        if bucket in self._hns_enabled_cache:
            return self._hns_enabled_cache[bucket]
        try:
            bucket_meta = await self._call("GET", f"b/{bucket}", json_out=True)
            hns = bucket_meta.get("hierarchicalNamespace", {})
            enabled = bool(hns.get("enabled", False))
        except Exception:
            logger.debug(
                "Could not determine HNS status for bucket '%s', assuming non-HNS",
                bucket,
            )
            enabled = False
        self._hns_enabled_cache[bucket] = enabled
        return enabled

    async def _mkdir_hns(self, path, bucket, key, create_parents=False):
        """Create a folder in an HNS-enabled bucket via the GCS Folders API.

        Parameters
        ----------
        path : str
            Full path including bucket, for cache invalidation.
        bucket : str
            Bucket name.
        key : str
            Object key / folder path within the bucket.
        create_parents : bool
            If True, creates parent folders as needed (recursive).
        """
        folder_path = key.rstrip("/")
        try:
            await self._call(
                "POST",
                "b/{}/folders",
                bucket,
                folder=folder_path,
                recursive=str(create_parents).lower(),
                json_out=True,
            )
            self.invalidate_cache(self._parent(path))
        except FileExistsError:
            # validate_response maps HTTP 412 to FileExistsError, but for the
            # Folders API this indicates a precondition failure (e.g., parent
            # directory does not exist when create_parents is False).
            if not create_parents:
                raise FileNotFoundError(
                    f"Parent directory does not exist for '{path}'"
                )
            raise
        except HttpError as e:
            if e.code in (409, "409"):
                # Folder already exists - not an error
                logger.debug(
                    "HNS folder already exists: %s/%s", bucket, folder_path
                )
                return
            raise

    async def _mkdir(
        self,
        path,
        acl="projectPrivate",
        default_acl="bucketOwnerFullControl",
        location=None,
        create_parents=False,
        enable_versioning=False,
        enable_object_retention=False,
        iam_configuration=None,
        **kwargs,
    ):
        """Create a directory or bucket, with HNS support.

        For sub-bucket paths on HNS-enabled (Hierarchical Namespace) buckets,
        creates a real directory using the GCS Folders API. For non-HNS
        buckets, sub-bucket paths are a no-op since GCS flat-namespace buckets
        don't have real directories. If path is just a bucket name, creates
        the bucket.

        Parameters
        ----------
        path : str
            bucket name or full path (e.g., 'mybucket/some/directory').
            For HNS-enabled buckets, sub-bucket directories will be created.
            For non-HNS buckets, sub-bucket paths have no effect.
        acl : str
            access for the bucket itself
        default_acl : str
            default ACL for objects created in this bucket
        location : str, optional
            Location where buckets are created, like 'US' or 'EUROPE-WEST3'.
        create_parents : bool
            If True, creates the bucket if it doesn't exist.
            For HNS buckets, also creates any missing parent directories.
        enable_versioning : bool
            If True, creates the bucket with object versioning enabled.
        enable_object_retention : bool
            If True, creates the bucket with object retention enabled.
        iam_configuration : dict, optional
            If provided, sets the IAM policy for the bucket.
        **kwargs
            Additional parameters passed to the bucket creation API call.
        """
        bucket, key, generation = self.split_path(path)
        if bucket in ["", "/"]:
            raise ValueError("Cannot create root bucket")
        if "/" in path:
            bucket_exists = await self._exists(bucket)
            if not bucket_exists:
                if not create_parents:
                    raise FileNotFoundError(bucket)
                # create_parents=True: fall through to create the bucket below
            else:
                # Bucket exists: check if HNS-enabled and create folder if so
                if key and await self._is_hns_enabled(bucket):
                    await self._mkdir_hns(path, bucket, key, create_parents)
                # For non-HNS buckets, sub-bucket paths are a no-op
                return

        json_data = {"name": bucket}
        location = location or self.default_location
        if location:
            json_data["location"] = location
        if enable_versioning:
            json_data["versioning"] = {"enabled": True}
        if iam_configuration:
            json_data["iamConfiguration"] = iam_configuration
            acl = None
            default_acl = None
        if kwargs:
            json_data.update(kwargs)

        await self._call(
            method="POST",
            path="b",
            predefinedAcl=acl,
            project=self.project,
            predefinedDefaultObjectAcl=default_acl,
            enableObjectRetention=str(enable_object_retention).lower(),
            json=json_data,
            json_out=True,
        )
        self.invalidate_cache(bucket)

    mkdir = asyn.sync_wrapper(_mkdir)
