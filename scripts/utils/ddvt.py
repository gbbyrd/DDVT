import torch
import numpy as np
from einops import rearrange

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

class DDVT:
    """ DDVT 3 Dimensional Translation Class. Creates the model and provides
    API for easy translation inference.
    """
    def __init__(self, config, ckpt, sensor_params, latent_shape, eval_mode=True, gpu=True):
        """Setup for DDVT class.

        Args:
            config (OmegaConf object): Configuration file for the model
            ckpt (_type_): _description_
            eval_mode (_type_): _description_
            gpu (_type_): _description_
            sensor_params (_type_): _description_
        """
        self.sensor_params = sensor_params
        self.model, global_step = self._load_model(config, ckpt, 
                                              gpu, eval_mode)
        self.latent_shape = latent_shape
        self.sampler = DDIMSampler(self.model)
        
    def translate(self, rgb_img, depth_img, translation_label, preprocessed):
        """Translates an inital rgb-depth img to a new relative location
        specified by the translation label.

        Args:
            rgb_img (numpy array): rgb image to translate from
            depth_img (numpy array): depth image to translate from
            translation_label (dict): contains the translation parameters
            preprocessed (bool): specifies whether your translation images and
                translation labels are already preprocessed
                
        Returns:
            translated_img_rgbd (numpy array): translated rgbd image
        """
        
        # pre process the translation data if needed
        if not preprocessed:
            rgbd_img, translation_label = self.preprocess_translation_data(rgb_img, depth_img, translation_label)
        else:
            rgbd_img = np.concatenate(rgb_img, depth_img, axis=1)
            
        # pass the preprocessed conditional image through the trained autoencoder
        # to go to the latent space for the diffusion process
        latent_conditioning = self.model.get_learned_conditioning(rgbd_img.to(self.model.device))
        
        # get translated, latent space image from the model
        latent_translated_img, intermediates = self.sampler.sample(200,
                                                                   1,
                                                                   shape=self.latent_shape,
                                                                   conditioning=latent_conditioning,
                                                                   verbose=False,
                                                                   translation_label=translation_label,
                                                                   eta=1.0)
        
        normalized_img = self.model.decode_first_stage(latent_translated_img)
        
        translated_img_rgbd = self.postprocess_translated_latent_img(normalized_img)
        
        return translated_img_rgbd
        
    def preprocess_translation_data(self, rgb_img, depth_img, translation_label):
        """Preprocess the frame data and get input for the diff view translation
        model.

        This function is modelled after the __getitem__ in the 
        ldm.dataset.custom_dataset.RGBDepthDatasetBase class
        
        """
        
        def preprocess_depth(depth_img):
            """ Normalize depth image and return h x w x 1 numpy array."""
            depth_img = depth_img[:,:,2] + 256 * depth_img[:,:,1] + 256 * 256 * depth_img[:,:,0]
            depth_img = depth_img / (256 * 256 * 256 - 1)
            
            # the distribution of depth values was HEAVILY skewed towards the lower end
            # therfore we will try to improve the distribution by clipping between
            # 0 and a threshold and normalizing based on these
            
            # need to test with clip_coefficient = 2
            clip_coefficient = 4

            depth_img = np.clip(depth_img, 0, 1/clip_coefficient)

            depth_img = depth_img * clip_coefficient

            depth_img = depth_img * 2 - 1

            return np.expand_dims(depth_img, axis=-1)

        # front_img_rgb = frame_info['front_rgb'] / 127.5 - 1
        aerial_img_rgb = rgb_img / 127.5 - 1

        aerial_img_depth = preprocess_depth(depth_img)

        # front_rgbd = np.concatenate((front_img_rgb, front_img_depth), axis=2)
        aerial_rgbd = np.concatenate((aerial_img_rgb, aerial_img_depth), axis=2)

        aerial_rgbd = torch.tensor(aerial_rgbd).unsqueeze(0)

        # the below is how the model arranges the condition.. taken directly from
        # ldm/model/diffusion/ddpm.py - get_input(self, batch, k)
        ############################################################################
        x = aerial_rgbd
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        ############################################################################

        aerial_rgbd = x

        # normalize the translation label before returning model input
        for key in translation_label:
            min_val = self.sensor_params['translation_limits_cube'][key][0]
            max_val = self.sensor_params['translation_limits_cube'][key][1]
            if min_val != max_val:
                translation_label[key] = ((translation_label[key] - min_val) / (max_val - min_val)) * 2 - 1

            # ensure that the normalized translation label is between -1 and 1
            assert translation_label[key] >= -1 and translation_label[key] <= 1, (\
                "Error: Normalized translation label outside acceptable bounds ([-1, 1])...")

        x = translation_label['x']
        y = translation_label['y']
        z = translation_label['z']
        yaw = translation_label['yaw']

        translation_label = torch.tensor(np.array([[x, y, z, yaw]], dtype='float32')).unsqueeze(0)

        return aerial_rgbd, translation_label
    
    def postprocess_translated_latent_img(self, img):
        # process translated img
        img = img.squeeze()
        translated_img = torch.clamp(img, -1., 1.)
        translated_img = translated_img.cpu().numpy()
        translated_img = np.transpose(translated_img, (1, 2, 0))
        def denormalize(img): 
            """ Takes an img normalized between [-1, 1] and denormalizes to between 
            [0, 255]
            """
            img = (((img + 1.0) / 2.0) * 255).astype(np.uint8)

            return img
        translated_img = denormalize(translated_img)

        return translated_img
        
    def _load_model(self, config, ckpt, gpu, eval_mode):
        print('LOADING DDVT MODEL...')
        def load_model_from_config(config, sd):
            model = instantiate_from_config(config)
            model.load_state_dict(sd,strict=False)
            model.cuda()
            model.eval()
            return model

        if ckpt:
            print(f"Loading model from {ckpt}")
            pl_sd = torch.load(ckpt, map_location="cpu")
            global_step = pl_sd["global_step"]
        else:
            pl_sd = {"state_dict": None}
            global_step = None
        model = load_model_from_config(config.model,
                                    pl_sd["state_dict"])

        return model, global_step