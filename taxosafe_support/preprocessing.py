"""Explicit training augmentation; evaluation never augments images."""
from PIL import Image, ImageOps


class Letterbox:
    def __init__(self, size=224):
        self.size = int(size)

    def __call__(self, image):
        image = image.convert("RGB")
        resampling = getattr(Image, "Resampling", Image)
        contained = ImageOps.contain(image, (self.size, self.size), method=resampling.BICUBIC)
        result = Image.new("RGB", (self.size, self.size), (123, 117, 104))
        result.paste(contained, ((self.size-contained.width)//2, (self.size-contained.height)//2))
        return result


def get_transform(data, training=False):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    mode = data.get("resize_mode", "center_crop")
    if mode not in ("center_crop", "letterbox"):
        raise ValueError("resize_mode must be center_crop or letterbox")
    operations = [transforms.Lambda(_rgb)]
    if mode == "letterbox":
        operations.append(Letterbox(224))
    else:
        operations.extend([transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                           transforms.CenterCrop(224)])
    if training and data.get("augment", True):
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    operations.extend([transforms.ToTensor(), transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    return transforms.Compose(operations)


def _rgb(image):
    return image.convert("RGB")
