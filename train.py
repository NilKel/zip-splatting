#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch.modules.lpips import LPIPS
import torchvision
from os import makedirs
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# ZIP method: NerfAcc occupancy grid
try:
    from nerfacc import OccGridEstimator
    from utils.occupancy_utils import (
        scatter_gaussians_to_occupancy,
        lookup_precomputed_occupancy
    )
    from gaussian_renderer.zip_render import render_zip
    NERFACC_AVAILABLE = True
except ImportError:
    NERFACC_AVAILABLE = False
    print("Warning: NerfAcc not available. ZIP method will not work.")

# Neural Interval Splatting with MLP-NeRF
try:
    import tinycudann as tcnn
    from gaussian_renderer.neural_render import render_neural_intervals
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("Warning: tiny-cuda-nn not available. MLP-NeRF blending will not work.")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Compute scene AABB for NGP-style intervals
    scene_aabb = None
    if dataset.method == "zip" and opt.use_ngp_intervals:
        scene_aabb = scene.compute_aabb(margin=0.1)
        print(f"Computed scene AABB: {scene_aabb.cpu().numpy()}")

    # Initialize occupancy grid for ZIP method
    estimator = None
    precomputed_occ = None
    # DISABLED: Old remnant code
    # if dataset.method == "zip":
    #     if not NERFACC_AVAILABLE:
    #         sys.exit("ZIP method requires NerfAcc, but it's not available. Please install nerfacc.")

    #     print(f"Initializing OccGridEstimator for ZIP method...")
    #     print(f"  Resolution: {opt.grid_resolution}")
    #     print(f"  Levels: {opt.grid_levels}")
    #     print(f"  Update interval: {opt.grid_update_interval}")
    #     print(f"  Warmup steps: {opt.grid_warmup_steps}")

    #     # Compute scene AABB
    #     aabb = scene.compute_aabb(margin=0.1)
    #     print(f"  Scene AABB: {aabb.cpu().numpy()}")

    #     # Initialize estimator
    #     estimator = OccGridEstimator(
    #         roi_aabb=aabb.cpu().tolist(),  # NerfAcc expects list
    #         resolution=opt.grid_resolution,
    #         levels=opt.grid_levels
    #     ).to("cuda")
    #     estimator.train()

    #     print("OccGridEstimator initialized successfully.")

    # Initialize MLP for neural interval rendering (mlp-nerf mode)
    mlp = None
    mlp_optimizer = None
    if opt.blending == "mlp-nerf":
        if not TCNN_AVAILABLE:
            sys.exit("MLP-NeRF blending requires tiny-cuda-nn, but it's not available. Please install tiny-cuda-nn.")

        print(f"Initializing MLP for Neural Interval Splatting...")
        print(f"  Feature dim: {gaussians.feature_dim}")
        print(f"  Output: 4 (RGB + density)")

        # Create tiny-cuda-nn MLP
        # Input: neural features [feature_dim]
        # Output: 4 (RGB + density)
        mlp_config = {
            "encoding": {
                "otype": "Identity"  # No encoding, just pass features through
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        }

        mlp = tcnn.NetworkWithInputEncoding(
            n_input_dims=gaussians.feature_dim,
            n_output_dims=4,  # RGB + density
            encoding_config=mlp_config["encoding"],
            network_config=mlp_config["network"]
        ).to("cuda")

        # Create optimizer for MLP
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        print("MLP initialized successfully.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ZIP method: Update occupancy grid
        # DISABLED: Old remnant code
        # if dataset.method == "zip" and estimator is not None:
        #     if iteration % opt.grid_update_interval == 0:
        #         # Scatter Gaussians into voxel occupancy grid
        #         with torch.no_grad():
        #             precomputed_occ = scatter_gaussians_to_occupancy(
        #                 gaussians,
        #                 grid_resolution=estimator.resolution,
        #                 grid_aabb=torch.tensor(estimator.aabbs[0].cpu().numpy(), device="cuda"),
        #                 levels=opt.grid_levels
        #             )

        #             # Create occ_eval_fn for NerfAcc
        #             def occ_eval_fn(positions):
        #                 """
        #                 Lookup precomputed occupancy at query positions.
        #                 positions: [N, 3] world-space coordinates
        #                 returns: [N, 1] occupancy values
        #                 """
        #                 return lookup_precomputed_occupancy(
        #                     positions,
        #                     precomputed_occ,
        #                     grid_resolution=estimator.resolution,
        #                     grid_aabb=torch.tensor(estimator.aabbs[0].cpu().numpy(), device="cuda"),
        #                     levels=opt.grid_levels
        #                 )

        #             # Update NerfAcc estimator
        #             estimator.update_every_n_steps(
        #                 step=iteration,
        #                 occ_eval_fn=occ_eval_fn,
        #                 occ_thre=1e-2,
        #                 ema_decay=0.95,
        #                 warmup_steps=opt.grid_warmup_steps,
        #                 n=opt.grid_update_interval
        #             )

        #             # Log grid statistics
        #             if iteration % (opt.grid_update_interval * 10) == 0:
        #                 occupied_voxels = (estimator.occs > 0).sum().item()
        #                 total_voxels = estimator.occs.shape[0]
        #                 occupancy_pct = 100.0 * occupied_voxels / total_voxels
        #                 print(f"[Iter {iteration}] Grid occupancy: {occupancy_pct:.2f}% ({occupied_voxels}/{total_voxels} voxels)")

        # Render: Use appropriate renderer based on method and blending mode
        if dataset.method == "zip" and opt.blending == "mlp-nerf":
            # Use Neural Interval Splatting renderer
            render_pkg = render_neural_intervals(
                viewpoint_cam, gaussians, mlp, pipe, bg,
                num_intervals=opt.num_intervals,
                near_depth=opt.near_plane,
                far_depth=opt.far_plane,
                feature_dim=gaussians.feature_dim,
                scaling_modifier=1.0,
                override_color=None,
                debug_iteration=None,  # Disable debug output
                use_ngp_intervals=opt.use_ngp_intervals,
                scene_aabb=scene_aabb
            )
        elif dataset.method == "zip":
            # Use ZIP renderer with standard interval blending
            render_pkg = render_zip(
                viewpoint_cam, gaussians, pipe, bg,
                estimator=estimator,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                use_intervals=True,  # Always use intervals for ZIP method
                blending_mode=opt.blending,
                num_intervals=opt.num_intervals,
                near_depth=opt.near_plane,
                far_depth=opt.far_plane,
                use_ngp_intervals=opt.use_ngp_intervals,
                scene_aabb=scene_aabb
            )
        else:
            # Use baseline renderer
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE
            )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # if viewpoint_cam.alpha_mask is not None:
        #     alpha_mask = viewpoint_cam.alpha_mask.cuda()
        #     image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        # For neural rendering, create dummy gradients for screenspace points
        # since they're not connected to the loss (we optimize neural features instead)
        if opt.blending == "mlp-nerf" and viewspace_point_tensor.grad is None:
            viewspace_point_tensor.grad = torch.zeros_like(viewspace_point_tensor)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    old_count = gaussians.get_xyz.shape[0]
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    new_count = gaussians.get_xyz.shape[0]
                    print(f"\n[ITER {iteration}] Densification: {old_count} -> {new_count} Gaussians (delta: {new_count - old_count:+d})")
                    # Synchronize CUDA after densification to ensure all operations complete
                    torch.cuda.synchronize()

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    torch.cuda.synchronize()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                # MLP optimizer step for neural interval rendering
                if mlp_optimizer is not None:
                    mlp_optimizer.step()
                    mlp_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # Render test images at the end of training
    if opt.test_image_stride > 0:
        render_test_images(scene, gaussians, pipe, background, dataset, opt, estimator, SPARSE_ADAM_AVAILABLE, mlp)

def render_test_images(scene, gaussians, pipeline, background, dataset, opt, estimator, separate_sh, mlp=None):
    """Render test images with GT and rendered views separately"""
    stride = opt.test_image_stride
    print("\n[Rendering test images with stride {}]".format(stride))
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("No test cameras found, skipping test image rendering.")
        return

    # Create output directory
    test_images_path = os.path.join(dataset.model_path, "test_images")
    makedirs(test_images_path, exist_ok=True)
    
    # Get indices to render: 0, stride, 2*stride, ...
    indices_to_render = list(range(0, len(test_cameras), stride))
    cameras_to_render = [test_cameras[i] for i in indices_to_render]
    print(f"Rendering {len(cameras_to_render)} out of {len(test_cameras)} test images")
    
    # Prepare metrics
    metrics_path = os.path.join(test_images_path, "metrics.txt")
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    num_imgs = 0
    
    # LPIPS model expects inputs in [-1, 1]
    lpips_model = LPIPS().cuda().eval()
    
    with torch.no_grad():
        with open(metrics_path, "w") as mf:
            mf.write("index, PSNR, SSIM, LPIPS\n")
            for actual_idx, view in zip(indices_to_render, tqdm(cameras_to_render, desc="Rendering test images")):
                # Use appropriate renderer based on method and blending mode
                if dataset.method == "zip" and opt.blending == "mlp-nerf" and mlp is not None:
                    # Use Neural Interval Splatting renderer
                    rendering = render_neural_intervals(
                        view, gaussians, mlp, pipeline, background,
                        num_intervals=16,
                        near_depth=0.1,
                        far_depth=100.0,
                        feature_dim=gaussians.feature_dim,
                        scaling_modifier=1.0,
                        override_color=None
                    )["render"]
                elif dataset.method == "zip":
                    # Use ZIP renderer with standard interval blending
                    rendering = render_zip(
                        view, gaussians, pipeline, background,
                        estimator=estimator,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=separate_sh,
                        use_intervals=True,
                        blending_mode=opt.blending
                    )["render"]
                else:
                    # Use baseline renderer
                    rendering = render(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh)["render"]
                gt = view.original_image[0:3, :, :]

                if dataset.train_test_exp:
                    rendering = rendering[..., rendering.shape[-1] // 2:]
                    gt = gt[..., gt.shape[-1] // 2:]
                
                # Clamp to [0, 1]
                rendering = torch.clamp(rendering, 0.0, 1.0)
                gt = torch.clamp(gt, 0.0, 1.0)
                
                # Save GT and rendered images in same folder with index-based naming
                torchvision.utils.save_image(gt, os.path.join(test_images_path, '{0}_gt.png'.format(actual_idx)))
                torchvision.utils.save_image(rendering, os.path.join(test_images_path, '{0}_r.png'.format(actual_idx)))

                # Metrics
                psnr_val = psnr(rendering.unsqueeze(0), gt.unsqueeze(0)).mean().item()
                try:
                    from fused_ssim import fused_ssim as _fssim
                    ssim_val = _fssim(rendering.unsqueeze(0), gt.unsqueeze(0)).item()
                except Exception:
                    ssim_val = ssim(rendering, gt).item() if hasattr(ssim(rendering, gt), 'item') else float(ssim(rendering, gt))
                lpips_val = lpips_model(2.0 * rendering.unsqueeze(0) - 1.0, 2.0 * gt.unsqueeze(0) - 1.0).mean().item()

                psnr_sum += psnr_val
                ssim_sum += ssim_val
                lpips_sum += lpips_val
                num_imgs += 1

                mf.write(f"{actual_idx}, {psnr_val:.6f}, {ssim_val:.6f}, {lpips_val:.6f}\n")
            if num_imgs > 0:
                mf.write("AVERAGES\n")
                mf.write(f"PSNR: {psnr_sum/num_imgs:.6f}\n")
                mf.write(f"SSIM: {ssim_sum/num_imgs:.6f}\n")
                mf.write(f"LPIPS: {lpips_sum/num_imgs:.6f}\n")
    
    print(f"Test images saved to {test_images_path}")

def prepare_output_and_logger(dataset, opt):
    if not dataset.model_path:
        if dataset.name:
            # Extract dataset and scene names from source path
            source_path = os.path.abspath(dataset.source_path)
            path_parts = source_path.split(os.sep)

            # Try to find dataset name (e.g., "nerf_synthetic") and scene name (e.g., "drums")
            # Look for common patterns: .../dataset/scene or .../dataset/.../scene
            dataset_name = "unknown"
            scene_name = "unknown"

            # Common dataset names to look for
            dataset_keywords = ["nerf_synthetic", "nerf_synthetic_colmap", "mipnerf360", "tanksandtemples", "deepblending"]
            for keyword in dataset_keywords:
                if keyword in path_parts:
                    dataset_idx = path_parts.index(keyword)
                    dataset_name = keyword
                    # Scene name is typically the next directory or the last part
                    if dataset_idx + 1 < len(path_parts):
                        scene_name = path_parts[dataset_idx + 1]
                    elif len(path_parts) > 0:
                        scene_name = path_parts[-1]
                    break

            # If no dataset keyword found, use parent directory as dataset and last part as scene
            if dataset_name == "unknown" and len(path_parts) >= 2:
                dataset_name = path_parts[-2] if len(path_parts) >= 2 else "unknown"
                scene_name = path_parts[-1]

            # Construct output path
            # For ZIP method: output/dataset/scene/method/blending/name
            # For other methods: output/dataset/scene/method/name
            if dataset.method == "zip":
                dataset.model_path = os.path.join("./output/", dataset_name, scene_name, dataset.method, opt.blending, dataset.name)
            else:
                dataset.model_path = os.path.join("./output/", dataset_name, scene_name, dataset.method, dataset.name)
        else:
            # Fallback to original UUID-based naming
            if os.getenv('OAR_JOB_ID'):
                unique_str=os.getenv('OAR_JOB_ID')
            else:
                unique_str = str(uuid.uuid4())
            dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
