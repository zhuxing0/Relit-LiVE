
from diffsynth import ModelManager, WanVideoRelitlivePipeline
from torchvision.transforms import v2
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from natsort import natsorted
from typing import List

import torch, os, imageio, argparse, torchvision, pdb, cv2, pathlib, glob, pyexr
import torch.nn as nn
import numpy as np

def stitch_frames(video_frames, rows=2, cols=5):
    num_frames = len(video_frames[0])
    assert all(len(frames) == num_frames for frames in video_frames), "All video frames must have the same number of frames."
    
    frame_width, frame_height = video_frames[0][0].size
    stitched_images = []
    for frame_idx in range(num_frames):
        stitched_image = Image.new('RGB', (cols * frame_width, rows * frame_height))
        for i in range(rows):
            for j in range(cols):
                video_idx = i * cols + j
                if video_idx < len(video_frames):
                    img = video_frames[video_idx][frame_idx]
                    x = j * frame_width
                    y = i * frame_height
                    stitched_image.paste(img, (x, y))
        stitched_images.append(stitched_image)
    
    return stitched_images

def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

def save_frames(frames, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(save_path, f"{i}.png"))
        
def rotate_panorama_around_horizontal_axis(
    panorama_img, 
    pitch_angle=10
):
    if isinstance(panorama_img, Image.Image):
        panorama = np.array(panorama_img.convert('RGB'))
    elif isinstance(panorama_img, np.ndarray):
        panorama = panorama_img.copy()
        if len(panorama.shape) != 3 or panorama.shape[2] != 3:
            raise ValueError("The NumPy array for the panoramic image must be in RGB format with dimensions (H, W, 3).")
    else:
        raise TypeError("The panoramic image must be a PIL.Image or a numpy.ndarray.")
    
    pan_h, pan_w = panorama.shape[:2]

    pitch_rad = np.radians(pitch_angle)

    lon = np.linspace(0, 2 * np.pi, pan_w)
    lat = np.linspace(-np.pi/2, np.pi/2, pan_h)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.cos(lon_grid)
    coords = np.stack([x, y, z], axis=-1)

    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])
    coords_rot = np.dot(coords, R_pitch.T)

    lon_new = np.arctan2(coords_rot[..., 0], coords_rot[..., 2])
    lat_new = np.arcsin(np.clip(coords_rot[..., 1], -1, 1))

    u_new = (lon_new / (2 * np.pi) + 0.5) * pan_w
    v_new = (lat_new / np.pi + 0.5) * pan_h

    u_new = u_new.astype(np.float32)
    v_new = v_new.astype(np.float32)
    
    new_panorama = cv2.remap(
        panorama,
        u_new, v_new,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP
    )

    new_panorama = np.hstack([new_panorama[:, pan_w//2:, ...], new_panorama[:, :pan_w//2, ...]])
    return new_panorama

class PBRVideo_img_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, env_map_path=None, dataset_type='relit-live', \
    use_ref_image=False, full_resolution=False, padding_resolution=False, drop_mr=False, args=None):

        self.dataset_type = dataset_type
        self.base_path = base_path
        self.args = args
        self.num_frames = num_frames
        
        p = pathlib.Path(base_path) 
        if not p.is_dir():
            raise NotADirectoryError(f"{base_path}' is not a valid dir")
        
        if self.dataset_type == "relit-live":
            self.path: List[pathlib.Path] = natsorted([item for item in p.iterdir() if item.is_dir()])
            self.frame_interval = frame_interval if self.num_frames!=1 else 0

        print(f'============= Load {len(self.path)}seqs from {base_path} =============')

        self.max_num_frames = max_num_frames
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.env_map_path = env_map_path
        self.use_ref_image = use_ref_image
        self.full_resolution = full_resolution
        self.padding_resolution = padding_resolution
        self.drop_mr = drop_mr
        
    def crop_and_resize(self, image, std_shape=None, env_reshape=False, align=None):
        if env_reshape:
            image = torchvision.transforms.functional.resize(
                image,
                (self.height, self.width),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
        else:
            if std_shape is not None:
                height, width = std_shape
            else:
                width, height = image.size
            if self.padding_resolution:
                scale = min(self.width / width, self.height / height)
            else:
                scale = max(self.width / width, self.height / height)
            if align is not None:
                if align == 'width':
                    scale = self.width / width
                elif align == 'height':
                    scale = self.height / height
            image = torchvision.transforms.functional.resize(
                image,
                (round(height*scale), round(width*scale)), 
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR
            )
        return image

    def load_frames_using_imageio(self, file_path, start_frame_id, interval, num_frames, frame_process, std_shape=None, env_reshape=False, return_shape=False):
        reader = imageio.get_reader(file_path)
        image_num = reader.count_frames()
        
        frames = []
        for frame_id in range(num_frames):
            sample_list = list(range(image_num)) + list(range(image_num-2, 0, -1))
            sample_id = sample_list[((start_frame_id + frame_id * interval) % len(sample_list))]
            frame = reader.get_data(sample_id)
            img_shape = frame.shape[:2]
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, std_shape, env_reshape)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if return_shape:
            return frames, img_shape
        return frames

    def load_frames_using_imageio_from_imgdir(self, file_path, start_frame_id, interval, num_frames, frame_process, divided_max=False, std_shape=None, env_reshape=False, return_shape=False):
        image_paths = glob.glob(os.path.join(file_path, '*.png')) + glob.glob(os.path.join(file_path, '*.jpg')) + glob.glob(os.path.join(file_path, '*.exr')) 
        if len(image_paths) == 0:
            return None

        sorted_image_paths = natsorted(image_paths)
        image_num = len(sorted_image_paths)

        frames = []
        for frame_id in range(num_frames):
            sample_list = list(range(image_num)) + list(range(image_num-2, 0, -1))
            sample_id = sample_list[((start_frame_id + frame_id * interval) % len(sample_list))]

            if ".exr" in sorted_image_paths[sample_id]:
                frame = pyexr.open(sorted_image_paths[sample_id]).get()
            else:
                frame = imageio.imread(sorted_image_paths[sample_id])

            if frame.shape[-1] == 4:
                frame = frame[:,:,:3]
            elif frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)

            img_shape = frame.shape[:2]
            if divided_max:
                bg_mask = (frame > 1000.0)
                if len(frame[bg_mask]) > 0:
                    frame_max = float(np.percentile(frame, 99))
                    frame = np.clip(frame / frame_max, 0, 1)
                else:
                    frame = frame / frame.max()
            
            if frame.min() < 0:
                frame = frame * 0.5 + 0.5
            
            if frame.dtype != 'uint8':
                frame = (frame * 255).astype(np.uint8)

            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, std_shape, env_reshape)
            frame = frame_process(frame)
            frames.append(frame)

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if return_shape:
            return frames, img_shape
        return frames

    def load_frames_using_imageio_from_imgpath(self, file_path, start_frame_id, interval, num_frames, frame_process, divided_max=False, std_shape=None, env_reshape=False, return_shape=False, align=None):
        image_paths = [file_path]
        if len(image_paths) == 0:
            return None

        sorted_image_paths = natsorted(image_paths)
        image_num = len(sorted_image_paths)

        frames = []
        for frame_id in range(num_frames):
            sample_list = list(range(image_num)) + list(range(image_num-2, 0, -1))
            sample_id = sample_list[((start_frame_id + frame_id * interval) % len(sample_list))]

            if ".exr" in sorted_image_paths[sample_id]:
                frame = pyexr.open(sorted_image_paths[sample_id]).get()
            else:
                frame = imageio.imread(sorted_image_paths[sample_id])

            if frame.shape[-1] == 4:
                frame = frame[:,:,:3]
            elif frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)

            img_shape = frame.shape[:2]

            if divided_max:
                bg_mask = (frame > 1000.0)
                if len(frame[bg_mask]) > 0:
                    frame_max = float(np.percentile(frame, 99))
                    frame = np.clip(frame / frame_max, 0, 1)
                else:
                    frame = frame / frame.max()
            
            if frame.min() < 0:
                frame = frame * 0.5 + 0.5
            
            if frame.dtype != 'uint8':
                frame = (frame * 255).astype(np.uint8)

            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, std_shape, env_reshape, align=align)
            frame = frame_process(frame)
            frames.append(frame)
        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        if return_shape:
            return frames, img_shape
        return frames
        
    def __getitem__(self, index):
        data_id = index % len(self.path)
        dir_path = self.path[data_id]

        RGB_path = os.path.join(dir_path, "images_4")
        basecolor_path = os.path.join(dir_path, "Base Color")
        depth_path = os.path.join(dir_path, "depth")
        metallic_path = os.path.join(dir_path, "Metallic")
        normal_path = os.path.join(dir_path, "normal")
        roughness_path = os.path.join(dir_path, "Roughness")

        fixed_env = True
        if self.env_map_path is not None:
            ldr_path = os.path.join(self.env_map_path, "ldr_video_fix_first_frame.mp4")
            hdr_log_path = os.path.join(self.env_map_path, "hdr_log_video_fix_first_frame.mp4")
            env_dir_path = os.path.join(self.env_map_path, "env_dir_video_fix_first_frame.mp4")
        else:
            ldr_path = os.path.join(dir_path, "env", "ldr_video_fix_first_frame.mp4")
            hdr_log_path = os.path.join(dir_path, "env", "hdr_log_video_fix_first_frame.mp4")
            env_dir_path = os.path.join(dir_path, "env", "env_dir_video_fix_first_frame.mp4")
            fixed_env = False

        start_frame_id = 0

        if self.full_resolution:
            source = self.load_frames_using_imageio_from_imgdir(RGB_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
            
            basecolor= self.load_frames_using_imageio_from_imgdir(basecolor_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
            depth= self.load_frames_using_imageio_from_imgdir(depth_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, divided_max=True, env_reshape=True)
            metallic= self.load_frames_using_imageio_from_imgdir(metallic_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
            
            normal= self.load_frames_using_imageio_from_imgdir(normal_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
            roughness = self.load_frames_using_imageio_from_imgdir(roughness_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
        else:
            source, std_shape = self.load_frames_using_imageio_from_imgdir(RGB_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, return_shape=True)

            basecolor= self.load_frames_using_imageio_from_imgdir(basecolor_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, std_shape=std_shape)
            depth= self.load_frames_using_imageio_from_imgdir(depth_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, divided_max=True, std_shape=std_shape)
            metallic= self.load_frames_using_imageio_from_imgdir(metallic_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, std_shape=std_shape)
            
            normal= self.load_frames_using_imageio_from_imgdir(normal_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, std_shape=std_shape)
            roughness = self.load_frames_using_imageio_from_imgdir(roughness_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, std_shape=std_shape)

        if metallic is None:
            metallic = source * 0.0
        if roughness is None:
            roughness = source * 0.0

        ldr = self.load_frames_using_imageio(ldr_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
        hdr_log = self.load_frames_using_imageio(hdr_log_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
        env_dir = self.load_frames_using_imageio(env_dir_path, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)

        if fixed_env:
            ldr = ldr[:,0:1].repeat(1, self.num_frames, 1, 1)
            hdr_log = hdr_log[:,0:1].repeat(1, self.num_frames, 1, 1)

        if self.drop_mr:
            metallic_weight = torch.tensor(0.0)
            roughness_weight = torch.tensor(0.0)
        else:
            metallic_weight = torch.tensor(1.0)
            roughness_weight = torch.tensor(1.0)

        env_self_weight = torch.tensor(1.0)
        env_cross_weight = torch.tensor(1.0)

        # print(f"haven't use prompt for {RGB_path}")
        prompt = ""

        if self.use_ref_image:

            image_paths = glob.glob(os.path.join(basecolor_path, '*.png')) + glob.glob(os.path.join(basecolor_path, '*.jpg')) + glob.glob(os.path.join(basecolor_path, '*.exr')) 
            max_idx = (len(image_paths)-1) - (self.num_frames-1) * self.frame_interval
            if max_idx == 0:
                start_frame_id_ref = 0
            else:
                start_frame_id_ref = start_frame_id

            ref_RGB_path = RGB_path
            if self.full_resolution:
                ref_source = self.load_frames_using_imageio_from_imgdir(ref_RGB_path, start_frame_id_ref, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
            else:
                ref_source = self.load_frames_using_imageio_from_imgdir(ref_RGB_path, start_frame_id_ref, self.frame_interval, self.num_frames, self.frame_process, std_shape=std_shape)

            if self.args.use_fixed_frame_and_w_rotate_light or self.args.use_fixed_frame_and_h_rotate_light or self.num_frames == 1:
                ref_idx = 0
            else:
                ref_idx = ref_source.shape[1] // 2 # Use the mid frame as the reference image
            
            ref_image = ref_source[:, ref_idx:ref_idx+1, :, :] 

            if self.args.ref_image_path_with_idddx is not None:
                ref_image_path_with_idx = self.args.ref_image_path_with_idddx.replace("idddx", f"{index}")
                if self.full_resolution:
                    ref_image = self.load_frames_using_imageio_from_imgpath(ref_image_path_with_idx, self.max_num_frames, start_frame_id_ref, self.frame_interval, self.num_frames, self.frame_process, env_reshape=True)
                else:
                    ref_image = self.load_frames_using_imageio_from_imgpath(ref_image_path_with_idx, self.max_num_frames, start_frame_id_ref, self.frame_interval, self.num_frames, self.frame_process, std_shape=None, align='width')
                ref_image = ref_image[:, -1:, :, :]

            data = {
                "source_video": source,
                "basecolor": basecolor,
                "depth": depth,
                "metallic":metallic,
                "normal": normal,
                "roughness":roughness,
                'metallic_weight': metallic_weight,
                'roughness_weight': roughness_weight,
                'env_self_weight': env_self_weight,
                'env_cross_weight': env_cross_weight,
                "ldr": ldr,
                "hdr_log": hdr_log,
                "env_dir": env_dir,
                "ref_image": ref_image,
                "ref_video": ref_source,
                "path": str(dir_path),
                "prompt": prompt,
            }
        else:
            data = {
                "source_video": source,
                "basecolor": basecolor,
                "depth": depth,
                "metallic":metallic,
                "normal": normal,
                "roughness":roughness,
                'metallic_weight': metallic_weight,
                'roughness_weight': roughness_weight,
                'env_self_weight': env_self_weight,
                'env_cross_weight': env_cross_weight,
                "ldr": ldr,
                "hdr_log": hdr_log,
                "env_dir": env_dir,
                "path": str(dir_path),
                "prompt": prompt,
            }

        return data

    def __len__(self):
        return len(self.path)

def parse_args():
    parser = argparse.ArgumentParser(description="diffusion renderer Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./example_test_data",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--env_map_path",
        type=str,
        default=None,
        help="env_map_path.",
    )
    parser.add_argument(
        "--use_ref_image",
        default=False,
        action="store_true",
        help="Whether to use reference image.",
    )
    parser.add_argument(
        "--use_muti_ref_image",
        default=False,
        action="store_true",
        help="Whether to use multiple reference images.",
    )
    parser.add_argument(
        "--ref_image_path_with_idddx",
        type=str,
        default=None,
        help="Path to the reference image with index.",
    )
    parser.add_argument(
        "--full_resolution",
        default=False,
        action="store_true",
        help="Whether to use full_resolution.",
    )
    parser.add_argument(
        "--padding_resolution",
        default=False,
        action="store_true",
        help="Whether to use padding_resolution.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="relit-live",
        help="Dataset format to load. Use 'relit-live' for the default Relit-LiVE directory layout.",
    )
    parser.add_argument(
        "--drop_mr",
        default=False,
        action="store_true",
        help="Ignore metallic and roughness inputs by setting their conditioning weights to zero.",
    )
    parser.add_argument(
        "--use_rotate_light",
        default=False,
        action="store_true",
        help="Enable light rotation mode during inference.",
    )
    parser.add_argument(
        "--use_fixed_frame_and_w_rotate_light",
        default=False,
        action="store_true",
        help="Repeat the first frame and rotate the environment map along the width axis across frames.",
    )
    parser.add_argument(
        "--use_fixed_frame_and_h_rotate_light",
        default=False,
        action="store_true",
        help="Repeat the first frame and rotate the environment map along the height axis across frames.",
    )
    parser.add_argument(
        "--h_rotate_light",
        type=int,
        default=0,
        help="Rotate the environment map vertically by this many degrees for every frame.",
    )
    parser.add_argument(
        "--w_rotate_light",
        type=int,
        default=0,
        help="Rotate the environment map horizontally by this many pixels for every frame.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps used during inference.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Sampling interval between frames read from the input sequence.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the fine-tuned checkpoint to load into the pipeline.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional explicit output file path. Supports .mp4 and .png.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--wo_ref_weight",
        type=float,
        default=0.0,
        help="Weight applied to the branch without reference-image conditioning.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=5,
        help="Video encoding quality passed to imageio when saving mp4 outputs.",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Load Wan2.1 pre-trained models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoRelitlivePipeline.from_model_manager(model_manager, device="cuda")

    # Initialize additional modules introduced in relit live
    dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    for block in pipe.dit.blocks:
        block.env_encoder = nn.Sequential(
        nn.Conv3d(
        48, dim, kernel_size=(1,2,2), stride=(1,2,2)),
        nn.SiLU()
        )
        block.env_encoder[0].weight.data.zero_()
        block.env_encoder[0].bias.data.zero_()
        block.projector = nn.Linear(dim, dim)
        block.projector.weight = nn.Parameter(torch.eye(dim))
        block.projector.bias = nn.Parameter(torch.zeros(dim))

    # if args.use_ref_image:
    dim = pipe.dit.patch_embedding.out_channels
    pipe.dit.ref_conv = nn.Conv2d(16, dim, kernel_size=(2,2), stride=(2,2))

    # Load checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=False) # True
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    output_dir = os.path.join(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = PBRVideo_img_Dataset(
        base_path=args.dataset_path,
        max_num_frames=args.num_frames,
        frame_interval=args.frame_interval,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        env_map_path=args.env_map_path,
        dataset_type=args.dataset_type,
        use_ref_image=args.use_ref_image,
        full_resolution=args.full_resolution,
        padding_resolution=args.padding_resolution,
        drop_mr=args.drop_mr,
        args=args,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # Inference
    ldr_img, log_img, ref_img = None, None, None
    for batch_idx, batch in enumerate(dataloader):

        input_rgb = batch["source_video"] # b c t h w
        basecolor = batch["basecolor"]
        depth = batch["depth"]
        metallic = batch["metallic"]
        normal = batch["normal"]
        roughness = batch["roughness"]
        ldr = batch["ldr"]
        hdr_log = batch["hdr_log"]
        env_dir = batch["env_dir"]
        dir_path = batch["path"][0]
        prompt = batch["prompt"]
        print(f'processing {dir_path}, prompt: {prompt}.')

        if args.h_rotate_light != 0:
            for frame_idx in range(args.num_frames):
                ldr_now = np.array(ldr[0,:,frame_idx,...].permute(1, 2, 0)) # c h w -> h w c
                hdr_log_now = np.array(hdr_log[0,:,frame_idx,...].permute(1, 2, 0))
                ldr_now = rotate_panorama_around_horizontal_axis(ldr_now, args.h_rotate_light)
                hdr_log_now = rotate_panorama_around_horizontal_axis(hdr_log_now, args.h_rotate_light)
                ldr[:,:,frame_idx:frame_idx+1,...] = torch.tensor(ldr_now).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                hdr_log[:,:,frame_idx:frame_idx+1,...] = torch.tensor(hdr_log_now).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            batch["ldr"] = ldr
            batch["hdr_log"] = hdr_log
        elif args.w_rotate_light != 0:
            for frame_idx in range(args.num_frames):
                ldr_now = np.array(ldr[0,:,frame_idx,...].permute(1, 2, 0)) # c h w -> h w c
                hdr_log_now = np.array(hdr_log[0,:,frame_idx,...].permute(1, 2, 0))
                ldr_rotate = np.concatenate((ldr_now[:,args.w_rotate_light:], ldr_now[:,:args.w_rotate_light]), axis=1)
                hdr_log_rotate = np.concatenate((hdr_log_now[:,args.w_rotate_light:], hdr_log_now[:,:args.w_rotate_light]), axis=1)
                ldr[:,:,frame_idx:frame_idx+1,...] = torch.tensor(ldr_rotate).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                hdr_log[:,:,frame_idx:frame_idx+1,...] = torch.tensor(hdr_log_rotate).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            batch["ldr"] = ldr
            batch["hdr_log"] = hdr_log
            
        if (args.use_fixed_frame_and_w_rotate_light or args.use_fixed_frame_and_h_rotate_light or args.use_rotate_light) and args.num_frames == 1:
            continue

        if (args.use_fixed_frame_and_w_rotate_light or args.use_fixed_frame_and_h_rotate_light or args.use_rotate_light) and args.num_frames != 1:
            # Repeat the first frame for num_frames times
            if not args.use_rotate_light:
                input_rgb = input_rgb[:, :, :1, :, :].repeat(1, 1, args.num_frames, 1, 1)
                batch["source_video"] = input_rgb
                basecolor = basecolor[:, :, :1, :, :].repeat(1, 1, args.num_frames, 1, 1)
                batch["basecolor"] = basecolor
                depth = depth[:, :, :1, :, :].repeat(1, 1, args.num_frames, 1, 1)
                batch["depth"] = depth
                metallic = metallic[:, :, :1, :, :].repeat(1, 1, args.num_frames, 1, 1)
                batch["metallic"] = metallic
                normal = normal[:, :, :1, :, :].repeat(1, 1, args.num_frames, 1, 1)
                batch["normal"] = normal
                roughness = roughness[:, :, :1, :, :].repeat(1, 1, args.num_frames, 1, 1)
                batch["roughness"] = roughness

            if args.use_fixed_frame_and_h_rotate_light:
                ldr_first = np.array(ldr[0,:,0,...].permute(1, 2, 0)) # c h w -> h w c
                hdr_log_first = np.array(hdr_log[0,:,0,...].permute(1, 2, 0))
                x_rotate_list = [int(360 / (args.num_frames-1) * (i+1)) for i in range(args.num_frames-1)]
                for frame_idx, x_rotate in enumerate(x_rotate_list):
                    ldr_now = rotate_panorama_around_horizontal_axis(ldr_first, x_rotate)
                    hdr_log_now = rotate_panorama_around_horizontal_axis(hdr_log_first, x_rotate)
                    ldr[:,:,frame_idx+1:frame_idx+2,...] = torch.tensor(ldr_now).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                    hdr_log[:,:,frame_idx+1:frame_idx+2,...] = torch.tensor(hdr_log_now).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
                batch["ldr"] = ldr
                batch["hdr_log"] = hdr_log
            else:
                ldr_first = ldr[:,:,:1,...]
                hdr_log_first = hdr_log[:,:,:1,...]
                _, c, _, h, w = ldr_first.shape
                y_rotate_list = [int(w / (args.num_frames-1) * (i+1)) for i in range(args.num_frames-1)]
                for frame_idx, y_rotate in enumerate(y_rotate_list): 
                    try:
                        ldr_now = torch.zeros_like(ldr_first)
                        ldr_now[:, :, :, :, -y_rotate:] = ldr_first[:, :, :, :, :y_rotate]
                        ldr_now[:, :, :, :, :-y_rotate] = ldr_first[:, :, :, :, y_rotate:]
                        hdr_log_now = torch.zeros_like(hdr_log_first)
                        hdr_log_now[:, :, :, :, -y_rotate:] = hdr_log_first[:, :, :, :, :y_rotate]
                        hdr_log_now[:, :, :, :, :-y_rotate] = hdr_log_first[:, :, :, :, y_rotate:]
                        ldr[:,:,frame_idx+1:frame_idx+2,...] = ldr_now
                        hdr_log[:,:,frame_idx+1:frame_idx+2,...] = hdr_log_now
                    except:
                        pdb.set_trace()
                batch["ldr"] = ldr
                batch["hdr_log"] = hdr_log
            
        seq_name = os.path.basename(dir_path)
        
        if args.use_rotate_light:
            seq_name += "_w_rotate_light"
        elif args.use_fixed_frame_and_w_rotate_light:
            seq_name += "_fixed_frame_and_w_rotate_light"
        elif args.use_fixed_frame_and_h_rotate_light:
            seq_name += "_fixed_frame_and_h_rotate_light"

        model_name = os.path.basename(args.ckpt_path).replace("ckpt", f'{seq_name}_{args.height}_{args.width}')
    
        if not args.use_ref_image:
            model_name += f'_no_ref_image'
        elif args.use_muti_ref_image:
            model_name += f'_ref_muti_image'
        else:
            model_name += f'_ref_mid_image'
        
        if args.env_map_path is not None:
            env_str = os.path.basename(args.env_map_path)
            model_name += f'_envdir_{env_str}'

        if args.drop_mr:
            model_name += f'_drop_mr'

        if args.padding_resolution:
            model_name += f'_padding'

        if args.full_resolution:
            model_name += f'_full'
        
        if args.num_inference_steps != 50:
            model_name += f'_steps{args.num_inference_steps}'

        if args.wo_ref_weight != 0.0:
            args.wo_ref_weight = round(args.wo_ref_weight, 2)
            model_name += f'_worw{args.wo_ref_weight}'
        
        if args.h_rotate_light != 0:
            model_name += f'_h-rotation-{args.h_rotate_light}'

        if args.w_rotate_light != 0:
            model_name += f'_w-rotation-{args.w_rotate_light}'

        if args.ref_image_path_with_idddx is not None:
            model_name += f'_ref_hr_{batch_idx}'

        model_name += f'_frames{args.num_frames}'

        if args.num_frames in [1]:
            save_path = os.path.join(output_dir, f"{model_name}_cfg{args.cfg_scale}_render.png")
        else:
            save_path = os.path.join(output_dir, f"{model_name}_cfg{args.cfg_scale}_video.mp4")

        video, envs = pipe(
                prompt=prompt,
                negative_prompt="",
                batch=batch,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                seed=0, tiled=True, wo_ref_weight=args.wo_ref_weight, use_muti_ref_image=args.use_muti_ref_image
            )

        input_rgb = pipe.tensor2video(input_rgb[0])
        basecolor = pipe.tensor2video(basecolor[0])
        depth = pipe.tensor2video(depth[0])
        metallic = pipe.tensor2video(metallic[0])
        normal = pipe.tensor2video(normal[0])
        roughness = pipe.tensor2video(roughness[0])
        ldr = pipe.tensor2video(ldr[0])
        hdr_log = pipe.tensor2video(hdr_log[0])
        env_dir = pipe.tensor2video(env_dir[0])
        
        if args.use_ref_image:
            ref_image = batch["ref_image"].repeat(1, 1, args.num_frames, 1, 1) # b c t h w
            ref_image = pipe.tensor2video(ref_image[0])
            stitched_results = stitch_frames([basecolor, metallic, roughness, depth, normal, ldr, ref_image, video, input_rgb, env_dir, envs, env_dir], rows=4, cols=3) 
        else:
            stitched_results = stitch_frames([basecolor, metallic, roughness, depth, normal, ldr, env_dir, video, input_rgb, env_dir, envs, env_dir], rows=4, cols=3) 
        
        print(f'Finish the inference of {model_name}.')

        if args.num_frames in [1]:
            output = video[0]
            output_all = stitched_results[0]
            output.save(os.path.join(output_dir, f"{model_name}_cfg{args.cfg_scale}_render.png"))
            output_all.save(os.path.join(output_dir, f"{model_name}_cfg{args.cfg_scale}.png"))
        else:
            save_video(video, os.path.join(output_dir, f"{model_name}_cfg{args.cfg_scale}_video.mp4"), fps=30, quality=args.quality)
            save_video(stitched_results, os.path.join(output_dir, f"{model_name}_cfg{args.cfg_scale}.mp4"), fps=30, quality=5)

        if args.output_path is not None:
            if '.mp4' in args.output_path:
                save_video(video, args.output_path, fps=30, quality=args.quality)
            elif '.png' in args.output_path:
                output = video[0]
                output.save(args.output_path)
            else:
                print(f'Error output_path: {args.output_path}')