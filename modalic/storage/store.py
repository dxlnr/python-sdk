from typing import Any, Dict, Optional

from minio import Minio

# from minio.error import S3Error


class Storage:
    r"""Instantiate Storage Object that enables access to S3 layer.

    Parameters:
    ------------------------
        cfg (dict): dictionary containing the configurations.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.client = Minio(
            endpoint=cfg["s3_server"],
            access_key=cfg["access_key"],
            secret_key=cfg["secret_access_key"],
            secure=False,
        )

    def upload(
        self,
        object: Any,
        object_name: str,
        length: int = -1,
        metadata: Optional[Dict[Any, Any]] = None,
    ) -> None:
        r"""Uploads data object to s3 storage bucket."""
        self.client.put_object(
            self.cfg["bucket"], object_name, object, length=length, metadata=metadata
        )

    # def create_bucket(self):
    #     r"""Instantiates and creates new bucket where all the object will be stored."""
    #     pass
