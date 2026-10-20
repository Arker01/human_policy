import os
import torch
import torch.nn as nn
import torchvision.transforms
from PIL import Image

import sys
# TODO(roger): temporary hack. Entire CET should be treated as a package
sys.path.append("../")

import hdt.constants

# From WildLMA
class MaskclipBackbone(nn.Module):
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()
        import maskclip_onnx
        self.model_name = model_name
        self.model, self.preprocess = maskclip_onnx.clip.load(
            model_name,
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.patch_size = self.model.visual.patch_size
        self.num_channels = 512
    
    @property
    def num_patches(self):
        # TODO(roger): compute this automatically
        # (336, 448) -> 588
        # (240, 320) -> 300
        # (480, 640) -> 1200
        return 300
    
    @property
    def hidden_size(self):
        if self.model_name == "ViT-B/16":
            return 512
        else:
            raise ValueError(f"Unknown model name {self.model_name}")

    def forward(self, img):
        with torch.no_grad():
            b, _, input_size_h, input_size_w = img.shape
            patch_h = input_size_h // self.patch_size
            patch_w = input_size_w // self.patch_size
            features = self.model.get_patch_encodings(img).to(torch.float32)  # (B, num_patches, 512)
            # clip_aligned_feats_bchw = features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)
            return features

def make_visual_encoder(policy_class, policy_config):
    if policy_class == 'ACT':
        # ACT from OpenTV builds its own visual encoder
        visual_encoder = None
        def visual_preprocessor(image_nchw):
            # Input: NCHW images. N is the number of cameras. np.uint8, [0, 255].
            # Output: processed NCHW that can be used for inference
            image_nchw = torch.from_numpy(image_nchw).float()
            return image_nchw / 255
    elif policy_config["visual_backbone"] == "MASKCLIP":
        # TODO(roger): load different backbones from policy config
        visual_encoder = MaskclipBackbone()
        normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        def visual_preprocessor(image_nchw):
            # Input: NCHW images. N is the number of cameras. np.uint8, [0, 255].
            # Output: processed NCHW that can be used for inference
            image_nchw = torch.from_numpy(image_nchw).float() / 255
            return normalizer(image_nchw)
        return visual_encoder, visual_preprocessor
    elif policy_config["visual_backbone"] == "SIGLIP":
        from .modeling_siglip import SiglipVisionTower
        # Create visual encoder
        SIGLIP_PATH = "google/siglip-so400m-patch14-384"

        visual_encoder = SiglipVisionTower(SIGLIP_PATH, args=None)

        vision_int_mean = tuple(int(x*255) for x in visual_encoder.image_processor.image_mean)

        def visual_preprocessor(image_nchw):
            image_nhwc = image_nchw.transpose((0, 2, 3, 1))
            image_tensor_list = []
            for camera_idx in range(image_nhwc.shape[0]):
                # convert image to a list of PIL images
                img_hwc = image_nhwc[camera_idx]
                image_pil = Image.fromarray(img_hwc)
                image_pil = visual_encoder.expand2square(image_pil, vision_int_mean)
                image_tensor_single = visual_encoder.image_processor.preprocess(image_pil, return_tensors='pt')['pixel_values'][0]
                image_tensor_list.append(image_tensor_single)

            return torch.stack(image_tensor_list, dim=0)
    else:
        raise ValueError(f"Unknown policy class {policy_class}")
    
    return visual_encoder, visual_preprocessor
