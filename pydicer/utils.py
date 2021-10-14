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
