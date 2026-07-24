from typing import Protocol
from PIL import Image


class FrameLoader(Protocol):
    def load(self, path: str) -> object: ...
    def downscale(self, image: object, max_side: int) -> object: ...
    def crop(self, image: object, bbox) -> object: ...
    def size(self, image: object) -> tuple: ...


class PILFrameLoader:
    def load(self, path: str):
        with Image.open(path) as im:
            return im.convert("RGB")

    def size(self, image):
        return image.size

    def downscale(self, image, max_side: int):
        w, h = image.size
        longest = max(w, h)
        if longest <= max_side:
            return image
        s = max_side / float(longest)
        return image.resize((max(1, round(w * s)), max(1, round(h * s))))

    def crop(self, image, bbox):
        return image.crop(tuple(bbox))
