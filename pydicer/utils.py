import hashlib


def hash_uid(uid, truncate=6):
    """Hash a UID and truncate it

    Args:
        uid (str): The UID to hash
        truncate (int, optional): The number of the leading characters to keep. Defaults to 6.

    Returns:
        str: The hashed and trucated UID
    """

    hash_sha = hashlib.sha256()
    hash_sha.update(uid.encode("UTF-8"))
    return hash_sha.hexdigest()[:truncate]


def find_linked_image(struct_dir):
    """Returns the image file linked to a structure directory

    Args:
        struct_dir (pathlib.Path): The structure directory

    Returns:
        pathlib.Path: The image linked to the structure. None if not image is found.
    """

    img_id = struct_dir.name.split("_")[1]

    img_links = list(struct_dir.parent.parent.glob(f"images/*{img_id}.nii.gz"))

    # If we have multiple linked images (not sure if this can happen but it might?)
    # then take the first one. If we find no linked images log and error and don't
    # visualise for now
    if len(img_links) == 0:
        return None

    return img_links[0]
