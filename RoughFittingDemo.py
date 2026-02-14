import logging
import os
import shutil
import tempfile
from pathlib import Path
import click
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Set global PyTorch print precision to 2 decimal places
torch.set_printoptions(precision=2)

from comotion_demo.models import comotion
from comotion_demo.utils import dataloading, helper
from comotion_demo.utils import track as track_utils

# Import existing modules from demo.py
from demo import device, use_mps, prepare_scene, add_pose_to_scene

# Try to import aitviewer for rendering
try:
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.smpl import SMPLLayer
    aitviewer_available = True
except ModuleNotFoundError:
    print("WARNING: aitviewer not available, skipping visualization rendering")
    aitviewer_available = False

# DEFAULT PARAMETERS - Modify these as needed
DEFAULT_INPUT_PATH = Path(r"C:\\Users\\genia\\Work\\ml-comotion\\TW_DTL_FULL.mp4")  # Change this to your video path
DEFAULT_OUTPUT_DIR = Path(r"C:\\Users\\genia\\Work\\ml-comotion\\output\\rough_fitting")
DEFAULT_FRAME_INTERVAL = 40
DEFAULT_ANALYZE_ONLY = False  # Set to True to only analyze existing results
DEFAULT_PERSON_HEIGHT = 1.85  # Person's actual height in meters (set to None to disable constraint)

def render_frame_with_pose(image, K, betas, pose, trans, smpl_layer, output_path, input_path):
    """
    Render a single frame with the fitted SMPL pose overlaid.
    
    Args:
        image: Input image tensor [3, H, W]
        K: Camera intrinsics matrix [3, 3]
        betas: SMPL beta parameters [10]
        pose: SMPL pose parameters [72]
        trans: SMPL translation parameters [3]
        smpl_layer: SMPLLayer for rendering
        output_path: Path to save the rendered image
        input_path: Original input path (for temp image saving)
    """
    viewer = None
    temp_image_path = None
    
    try:
        # Convert tensor image to PIL and save temporarily
        from comotion_demo.utils.dataloading import convert_tensor_to_image
        image_np = convert_tensor_to_image(image)
        temp_image_path = str(output_path).replace('.png', '_temp.jpg')
        Image.fromarray(image_np).save(temp_image_path)
        
        # Get image dimensions
        image_height, image_width = image.shape[-2:]
        
        # Initialize viewer with explicit cleanup
        viewer = HeadlessRenderer(size=(image_width, image_width))
        
        # Clear any existing scene
        viewer.reset()
        
        prepare_scene(viewer, image_width, image_height, K.cpu().numpy(), [temp_image_path])
        
        # Add SMPL pose to scene
        add_pose_to_scene(
            viewer,
            smpl_layer,
            betas.unsqueeze(0),  # Add batch dimension
            pose.unsqueeze(0),   # Add batch dimension
            trans.unsqueeze(0),  # Add batch dimension
        )
        
        # Save rendered scene
        viewer.save_frame(str(output_path))
        
        logging.info(f"Rendered frame saved to: {output_path}")
        
    except Exception as e:
        logging.warning(f"Failed to render frame: {e}")
        
    finally:
        # Clean up resources
        if viewer is not None:
            try:
                viewer.close()  # Properly close the viewer
            except:
                pass  # Ignore cleanup errors
                
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass  # Ignore cleanup errors

def render_frame_with_shared_viewer(image, K, betas, pose, trans, smpl_layer, viewer, output_path, input_path):
    """
    Render a single frame with the fitted SMPL pose overlaid using a shared viewer.
    
    Args:
        image: Input image tensor [3, H, W]
        K: Camera intrinsics matrix [3, 3]
        betas: SMPL beta parameters [10]
        pose: SMPL pose parameters [72]
        trans: SMPL translation parameters [3]
        smpl_layer: SMPLLayer for rendering
        viewer: Shared HeadlessRenderer instance
        output_path: Path to save the rendered image
        input_path: Original input path (for temp image saving)
    """
    temp_image_path = None
    
    try:
        # Convert tensor image to PIL and save temporarily
        from comotion_demo.utils.dataloading import convert_tensor_to_image
        image_np = convert_tensor_to_image(image)
        temp_image_path = str(output_path).replace('.png', '_temp.jpg')
        Image.fromarray(image_np).save(temp_image_path)
        
        # Get image dimensions
        image_height, image_width = image.shape[-2:]
        
        # Clear any existing scene state
        viewer.reset()
        
        prepare_scene(viewer, image_width, image_height, K.cpu().numpy(), [temp_image_path])
        
        # Add SMPL pose to scene
        add_pose_to_scene(
            viewer,
            smpl_layer,
            betas.unsqueeze(0),  # Add batch dimension
            pose.unsqueeze(0),   # Add batch dimension
            trans.unsqueeze(0),  # Add batch dimension
        )
        
        # Save rendered scene
        viewer.save_frame(str(output_path))
        
        logging.info(f"Rendered frame saved to: {output_path}")
        
    except Exception as e:
        logging.warning(f"Failed to render frame: {e}")
        
    finally:
        # Clean up temp file only (keep viewer alive)
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass  # Ignore cleanup errors

def constrain_betas_to_height(betas, target_height_meters, smpl_layer):
    """
    Constrain beta parameters to match a target height by adjusting the first beta.
    
    Args:
        betas: [10] tensor of SMPL beta parameters
        target_height_meters: Target height in meters
        smpl_layer: SMPLLayer for computing height
    
    Returns:
        Constrained beta parameters
    """
    if target_height_meters is None:
        return betas
        
    try:
        with torch.no_grad():
            # Use neutral T-pose to measure height
            neutral_pose = torch.zeros(1, 72)
            neutral_trans = torch.zeros(1, 3)
            
            # Test current height with these betas
            betas_batch = betas.unsqueeze(0)
            smpl_output = smpl_layer(
                betas=betas_batch,
                body_pose=neutral_pose[:, 3:],
                global_orient=neutral_pose[:, :3],
                transl=neutral_trans
            )
            vertices = smpl_output.vertices[0]
            current_height = vertices[:, 1].max() - vertices[:, 1].min()
            
            # Calculate required scale factor
            scale_factor = target_height_meters / current_height.item()
            
            # Adjust first beta (overall scale) - empirically derived relationship
            # Beta[0] roughly corresponds to overall body scale
            constrained_betas = betas.clone()
            height_adjustment = torch.log(torch.tensor(scale_factor)) * 2.0  # Empirical scaling
            constrained_betas[0] = height_adjustment
            
            # Verify the adjustment worked
            constrained_batch = constrained_betas.unsqueeze(0)
            test_output = smpl_layer(
                betas=constrained_batch,
                body_pose=neutral_pose[:, 3:],
                global_orient=neutral_pose[:, :3], 
                transl=neutral_trans
            )
            test_vertices = test_output.vertices[0]
            final_height = test_vertices[:, 1].max() - test_vertices[:, 1].min()
            
            logging.info(f"Height constraint: {current_height}m -> {final_height}m (target: {target_height_meters}m)")
            
            return constrained_betas
            
    except Exception as e:
        logging.warning(f"Failed to constrain height, using original betas: {e}")
        return betas

def rough_fitting_sparse_frames(
    input_path, 
    cache_path, 
    start_frame=0, 
    num_frames=1000000000, 
    frame_interval=10,  # Process every 10th frame
    model=None,
    save_renderings=True,  # New parameter to enable/disable rendering
    person_height_meters=None  # Height constraint for beta parameters
):
    """
    Perform rough fitting on sparse frames to estimate average betas and camera info.
    
    Args:
        input_path: Path to video or image directory
        cache_path: Where to save results
        start_frame: Starting frame index
        num_frames: Maximum frames to process
        frame_interval: Process every N-th frame (10 = every 10th frame)
        model: CoMotion model (will create if None)
        save_renderings: Whether to save rendered frames with fitted poses
        person_height_meters: Target height in meters to constrain beta parameters
    
    Returns:
        dict with: average_betas, frame_results, processing_info
    """
    if model is None:
        model = comotion.CoMotion(use_coreml=use_mps)
    model.to(device).eval()
    
    logging.info(f"Starting rough fitting on every {frame_interval}th frame")
    
    beta_estimates = []
    pose_estimates = []
    trans_estimates = []
    confidences = []
    frame_indices = []
    Ks = []
    frame_images = []  # Store original images for rendering
    constrained_betas_list = []  # Store height-constrained betas separately
    
    processed_count = 0
    initialized = False
    
    # Create output directory for frame renderings if needed
    if save_renderings and aitviewer_available:
        render_dir = cache_path.parent / f"{cache_path.stem}_frames"
        render_dir.mkdir(exist_ok=True)
        logging.info(f"Frame renderings will be saved to: {render_dir}")
        
        # Initialize SMPL layer for rendering
        smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
        
        # Initialize a single viewer for all rendering (create it once, reuse it)
        viewer = None
    else:
        render_dir = None
        smpl_layer = None
        viewer = None
    
    # Process sparse frames
    for frame_idx, (image, K) in enumerate(tqdm(
        dataloading.yield_image_and_K(input_path, start_frame, num_frames, 1),
        desc="Rough fitting (sparse frames)"
    )):
        # Skip frames that aren't at our interval
        if frame_idx % frame_interval != 0:
            continue
            
        if not initialized:
            image_res = image.shape[-2:]
            model.init_tracks(image_res)
            initialized = True
        
        # Run detection on this frame
        detection, track = model(image, K, use_mps=use_mps)
        
        # Extract single-person detection (highest confidence)
        if detection["pose"].shape[0] > 0:
            confidences_frame = detection.get("confidence", torch.ones(detection["pose"].shape[0]))
            best_idx = confidences_frame.argmax()
            
            # Store results (both original and constrained betas)
            beta_estimates.append(detection["betas"][best_idx].cpu())  # Original for comparison
            pose_estimates.append(detection["pose"][best_idx].cpu())
            trans_estimates.append(detection["trans"][best_idx].cpu())
            confidences.append(confidences_frame[best_idx].cpu())
            frame_indices.append(frame_idx)
            Ks.append(K.cpu())
            
            # Constrain betas to target height if specified
            constrained_betas = detection["betas"][best_idx].cpu()
            if person_height_meters is not None and smpl_layer is not None:
                constrained_betas = constrain_betas_to_height(
                    constrained_betas, person_height_meters, smpl_layer
                )
            constrained_betas_list.append(constrained_betas)
            
            # Store original image for rendering
            if save_renderings and aitviewer_available:
                frame_images.append(image.cpu())
                
                # Initialize viewer on first frame
                if viewer is None:
                    image_height, image_width = image.shape[-2:]
                    viewer = HeadlessRenderer(size=(image_width, image_height))
                
                # Render this frame with fitted pose using constrained betas
                render_frame_with_shared_viewer(
                    image, K, 
                    constrained_betas,  # Use constrained betas for rendering
                    detection["pose"][best_idx], 
                    detection["trans"][best_idx],
                    smpl_layer,
                    viewer,
                    render_dir / f"frame_{frame_idx:06d}_fitted.png",
                    input_path
                )
            
            processed_count += 1
            logging.info(f"Processed frame {frame_idx}, confidence: {confidences_frame[best_idx]}")
        
        # Stop if we've processed enough frames
        if processed_count >= 20:  # Limit to ~20 frames for rough estimation
            break
    
    # Clean up the shared viewer
    if viewer is not None:
        try:
            viewer.close()
        except:
            pass  # Ignore cleanup errors
    
    if len(beta_estimates) == 0:
        raise ValueError("No detections found in any processed frames")
    
    # Calculate average betas weighted by confidence (use constrained betas if available)
    betas_for_averaging = constrained_betas_list if constrained_betas_list else beta_estimates
    betas_tensor = torch.stack(betas_for_averaging)  # [N, 10] - use constrained betas
    confidences_tensor = torch.stack(confidences)  # [N]
    
    # Also keep original betas for comparison
    original_betas_tensor = torch.stack(beta_estimates)  # [N, 10] - original unconstrained
    
    # Normalize confidences to sum to 1
    confidence_weights = confidences_tensor / confidences_tensor.sum()
    
    # Weighted average of betas
    average_betas = (betas_tensor * confidence_weights.unsqueeze(-1)).sum(0)
    
    logging.info(f"Computed average betas from {len(betas_for_averaging)} frames:")
    logging.info(f"Average betas (constrained): {average_betas}")
    if person_height_meters is not None:
        original_average = (original_betas_tensor * confidence_weights.unsqueeze(-1)).sum(0)
        logging.info(f"Average betas (original): {original_average}")
        logging.info(f"Height constraint applied: {person_height_meters}m")
    logging.info(f"Confidence range: {confidences_tensor.min()} - {confidences_tensor.max()}")
    
    # Estimate rough scale from poses (for camera transform estimation)
    poses_tensor = torch.stack(pose_estimates)
    trans_tensor = torch.stack(trans_estimates)
    
    # Simple camera transform estimation (very rough)
    # This is placeholder - you'd need proper multi-view data for real camera estimation
    estimated_camera_info = {
        "average_translation": trans_tensor.mean(0),
        "translation_std": trans_tensor.std(0),
        "average_K": torch.stack(Ks).mean(0),
        "processing_frames": frame_indices,
        "num_detections": len(beta_estimates)
    }
    
    # Package results
    results = {
        "average_betas": average_betas,
        "beta_estimates": betas_tensor,  # Constrained betas
        "original_beta_estimates": original_betas_tensor,  # Original unconstrained betas
        "pose_estimates": poses_tensor,
        "trans_estimates": trans_tensor,
        "confidences": confidences_tensor,
        "frame_indices": frame_indices,
        "camera_info": estimated_camera_info,
        "processing_info": {
            "frame_interval": frame_interval,
            "frames_processed": len(betas_for_averaging),
            "input_path": str(input_path),
            "height_constraint": person_height_meters
        }
    }
    
    # Save results
    torch.save(results, cache_path)
    logging.info(f"Rough fitting results saved to {cache_path}")
    
    return results

def estimate_person_height_from_betas(betas, smpl_decoder):
    """
    Estimate person height from beta parameters using SMPL model.
    
    Args:
        betas: [10] tensor of SMPL beta parameters
        smpl_decoder: SMPL model decoder
    
    Returns:
        height in meters
    """
    with torch.no_grad():
        # Use neutral T-pose
        neutral_pose = torch.zeros(1, 72)  # [1, 72]
        neutral_trans = torch.zeros(1, 3)   # [1, 3]
        betas_batch = betas.unsqueeze(0)     # [1, 10]
        
        # Forward pass through SMPL
        smpl_output = smpl_decoder(betas_batch, neutral_pose, neutral_trans)
        vertices = smpl_output.vertices[0]  # [6890, 3]
        
        # Calculate height (Y-axis difference)
        height = vertices[:, 1].max() - vertices[:, 1].min()
        return height.item()

def analyze_rough_fitting_results(results_path):
    """
    Analyze the results from rough fitting and provide insights.
    """
    results = torch.load(results_path, weights_only=False)
    
    print("\n" + "="*50)
    print("ROUGH FITTING ANALYSIS")
    print("="*50)
    
    avg_betas = results["average_betas"]
    beta_estimates = results["beta_estimates"]
    confidences = results["confidences"]
    
    print(f"Frames processed: {len(beta_estimates)}")
    print(f"Frame interval: {results['processing_info']['frame_interval']}")
    
    # Show height constraint info if applied
    height_constraint = results['processing_info'].get('height_constraint')
    if height_constraint:
        print(f"Height constraint: {height_constraint}m")
    
    print(f"Confidence range: {confidences.min()} - {confidences.max()}")
    print(f"Average confidence: {confidences.mean()}")
    
    print(f"\nAverage Beta Parameters (height-constrained):")
    for i, beta_val in enumerate(avg_betas):
        print(f"  β{i:02d}: {beta_val}")
    
    # Show original unconstrained betas for comparison if available
    if 'original_beta_estimates' in results:
        original_betas = results['original_beta_estimates']
        original_avg = (original_betas * (confidences / confidences.sum()).unsqueeze(-1)).sum(0)
        print(f"\nOriginal Beta Parameters (before height constraint):")
        for i, beta_val in enumerate(original_avg):
            print(f"  β{i:02d}: {beta_val}")
        
        # Show the difference in first beta (height parameter)
        height_diff = avg_betas[0] - original_avg[0]
        print(f"\nHeight adjustment (β00): {height_diff:+.3f}")
    
    # The stability stats aren't required, but keep code just in case.
    # print(f"\nBeta Parameter Stability (std dev):")
    # beta_stds = beta_estimates.std(0)
    # for i, std_val in enumerate(beta_stds):
    #     print(f"  β{i:02d}: {std_val}")
    
    # Identify most/least stable parameters
    # most_stable = beta_stds.argmin()
    # least_stable = beta_stds.argmax()
    # print(f"\nMost stable parameter: β{most_stable} (std: {beta_stds[0][most_stable]})")
    # print(f"Least stable parameter: β{least_stable} (std: {beta_stds[least_stable]})")
    
    print(f"\nCamera Info:")
    cam_info = results["camera_info"]
    print(f"  Average translation: {cam_info['average_translation']}")
    print(f"  Translation std: {cam_info['translation_std']}")
    
    # Check for rendered frames
    results_path_obj = Path(results_path)
    render_dir = results_path_obj.parent / f"{results_path_obj.stem}_frames"
    if render_dir.exists():
        rendered_files = list(render_dir.glob("*.png"))
        print(f"\nRendered Frames:")
        print(f"  Directory: {render_dir}")
        print(f"  Number of rendered frames: {len(rendered_files)}")
    else:
        print(f"\nNo rendered frames found (aitviewer may not be available)")
    
    return results

def run_with_defaults():
    """Run rough fitting with default parameters - can be called directly."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Suppress verbose aitviewer/OpenGL logging
    logging.getLogger('aitviewer').setLevel(logging.WARNING)
    logging.getLogger('moderngl').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Suppress specific shader/source loading messages if they appear
    for logger_name in ['_load_source', '_load_shader', 'shader', 'opengl']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    input_path = DEFAULT_INPUT_PATH
    output_dir = DEFAULT_OUTPUT_DIR
    frame_interval = DEFAULT_FRAME_INTERVAL
    analyze_only = DEFAULT_ANALYZE_ONLY
    person_height = DEFAULT_PERSON_HEIGHT
    
    print("🚀 Running Rough Fitting Demo with default parameters:")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_dir}")
    print(f"   Frame interval: {frame_interval}")
    print(f"   Person height constraint: {person_height}m" if person_height else "   No height constraint")
    print(f"   Analyze only: {analyze_only}")
    print("-" * 50)
    
    # Validate input path
    if not input_path.exists():
        print(f"❌ Error: Input path does not exist: {input_path}")
        print("Please update DEFAULT_INPUT_PATH in the code to point to your video file.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = input_path.stem
    cache_path = output_dir / f"{input_name}_rough_fitting.pt"
    
    if analyze_only and cache_path.exists():
        # Just analyze existing results
        print("📊 Analyzing existing results...")
        analyze_rough_fitting_results(cache_path)
    else:
        # Run rough fitting
        try:
            print("🎬 Starting rough fitting...")
            results = rough_fitting_sparse_frames(
                input_path=input_path,
                cache_path=cache_path,
                frame_interval=frame_interval,
                save_renderings=aitviewer_available,  # Enable if aitviewer available
                person_height_meters=person_height
            )
            
            # Automatically analyze results
            analyze_rough_fitting_results(cache_path)
            
            print(f"\n✅ Rough fitting complete!")
            print(f"📁 Results saved to: {cache_path}")
            print(f"🔍 Set DEFAULT_ANALYZE_ONLY = True to re-analyze these results")
            
        except Exception as e:
            logging.error(f"Rough fitting failed: {e}")
            print(f"❌ Error: {e}")
            raise

# CLI command (still available if needed)
@click.command()
@click.option(
    "-i", "--input-path", required=True, type=click.Path(exists=True, path_type=Path),
    help="Path to input video or image directory"
)
@click.option(
    "-o", "--output-dir", required=True, type=click.Path(path_type=Path),
    help="Output directory for results"
)
@click.option(
    "--frame-interval", default=10, type=int,
    help="Process every N-th frame (default: 10)"
)
@click.option(
    "--analyze", is_flag=True,
    help="Analyze existing rough fitting results"
)
@click.option(
    "--save-renderings", is_flag=True, default=True,
    help="Save individual frame renderings with fitted poses"
)
def main_rough_fitting(input_path, output_dir, frame_interval, analyze, save_renderings):
    """Run rough fitting on sparse frames to estimate average betas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_name = input_path.stem
    
    cache_path = output_dir / f"{input_name}_rough_fitting.pt"
    
    if analyze and cache_path.exists():
        # Just analyze existing results
        analyze_rough_fitting_results(cache_path)
    else:
        # Run rough fitting
        try:
            results = rough_fitting_sparse_frames(
                input_path=input_path,
                cache_path=cache_path,
                frame_interval=frame_interval,
                save_renderings=save_renderings and aitviewer_available
            )
            
            # Automatically analyze results
            analyze_rough_fitting_results(cache_path)
            
            print(f"\n✅ Rough fitting complete!")
            print(f"📁 Results saved to: {cache_path}")
            print(f"🔍 Use --analyze flag to re-analyze these results")
            
        except Exception as e:
            logging.error(f"Rough fitting failed: {e}")
            raise

# Main execution - runs when you press play in VSCode
if __name__ == "__main__":
    # Check if running with command line arguments
    import sys
    if len(sys.argv) > 1:
        # Running with CLI arguments
        main_rough_fitting()
    else:
        # Running directly (F5 in VSCode) - use defaults
        run_with_defaults()