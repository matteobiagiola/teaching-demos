from torch import Tensor
from PIL import Image
from torchvision import transforms
import numpy as np
import io


class CompressionDefence:

    def __init__(self, jpeg_quality: int = 95) -> None:
        self.jpeg_quality = jpeg_quality

    def defend(self, image: Tensor) -> Tensor:

        # assuming image is standard normalized; remove batch and channel dims for PIL processing
        image_clone = image.clone().squeeze().squeeze()
        image_clone = image_clone * 0.3081 + 0.1307

        image_np = image_clone.squeeze().detach().cpu().numpy()
        # L mode for grayscale
        img = Image.fromarray((image_np * 255).clip(0, 255).astype("uint8"), mode="L")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.jpeg_quality)
        buffer.seek(0)

        decoded_image = np.array(Image.open(buffer).convert("L"))

        # renormalize perturbed image back to original scale
        transform_compose = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        decoded_tensor_normalized = transform_compose(decoded_image)

        return decoded_tensor_normalized.unsqueeze(0)  # add batch dim back
