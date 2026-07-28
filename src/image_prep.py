

import io
from PIL import Image


class ImageLoadError(Exception):
    """
    Raised when a file cannot be decoded as an image at all -- corrupt,
    truncated, or (as with optera_doc_33.jpg in this dataset) an HTML error
    page saved with an image extension after a broken download link.

    Distinct from gemini_client.GeminiError: this is a bad *input file*,
    caught before any paid API call is made, not a bad model response.
    """
    pass


def validate_image(path: str) -> None:
    """
    Cheap structural check: can this file be decoded as an image at all?
    Used by the baseline pipeline (which otherwise never opens the file,
    it just ships raw bytes) so a non-image file never gets sent to a paid
    API declaring a fake `image/jpeg` mime type. Raises ImageLoadError if not.
    """
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise ImageLoadError(f"{path}: not a decodable image ({e})") from e


def load_and_resize(path: str, max_dimension: int, jpeg_quality: int) -> bytes:
    """
    Load an image, downscale so its longest edge is <= max_dimension
    (no-op if already smaller), re-encode as JPEG, return raw bytes.
    Raises ImageLoadError if the file can't be decoded as an image.
    """
    try:
        img = Image.open(path)
        img.load()  # force full decode now -- Image.open() is lazy and won't
                    # always raise until pixel data is actually accessed
    except Exception as e:
        raise ImageLoadError(f"{path}: not a decodable image ({e})") from e

    img = img.convert("RGB")  # normalize mode (some phone JPEGs are CMYK/palette)

    w, h = img.size
    longest = max(w, h)
    if longest > max_dimension:
        scale = max_dimension / longest
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return buf.getvalue()


def load_original_bytes(path: str) -> bytes:
    """
    Used by the baseline pipeline -- no resizing, send exactly what arrived.
    Still validates the file is a real image first: the baseline being
    "naive" means no cost optimization, not no input checking -- shipping
    raw HTML bytes to Gemini under a hardcoded image/jpeg mime type just
    burns a paid call to get back a 400.
    """
    validate_image(path)
    with open(path, "rb") as f:
        return f.read()
