#%%
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from comotion_demo.utils import dataloading, smpl_kinematics
import time
#%% Setup OSC client for Blender communication
from pythonosc import udp_client
client = udp_client.SimpleUDPClient("127.0.0.1", 9001)
client.send_message("/filter", 1)
#%
# Set global PyTorch print precision to 2 decimal places
torch.set_printoptions(precision=2)
#%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_mps = torch.mps.is_available()
print(f"Using device: {device}")
torch.set_printoptions(precision=2)

from aitviewer.renderables.smpl import SMPLLayer
from aitviewer.configuration import CONFIG
#%% SET DEFAULT VALUES
# DEFAULT PARAMETERS - Modify these as needed
DEFAULT_INPUT_PATH = Path(r"C:\\Users\\genia\\Work\\ml-comotion\\TW_DTL_FULL.mp4")  # Change this to your video path
DEFAULT_OUTPUT_DIR = Path(r"C:\\Users\\genia\\Work\\ml-comotion\\output\\rough_fitting")
DEFAULT_FRAME_INTERVAL = 40
DEFAULT_ANALYZE_ONLY = False  # Set to True to only analyze existing results
DEFAULT_PERSON_HEIGHT = 1.85  # Person's actual height in meters (set to None to disable constraint)
#%
beta_estimates = []
pose_estimates = []
trans_estimates = []
confidences = []
frame_indices = []

#% SETUP SMPL LAYER
CONFIG.smplx_models = Path(r"C:\\Users\\genia\\Work\\ml-comotion\\src\\comotion_demo\\data\\")
smpl_layer = SMPLLayer(model_type="smpl", gender="male")
smpl_kin = smpl_kinematics.SMPLKinematics()

#%% GET IMAGE FROM VIDEO INPUT PATH 
enumImages = dataloading.yield_image_and_K_and_Index(DEFAULT_INPUT_PATH, 40, 140, 40)
# Save filename and start frame for later identification
analysis_filename = DEFAULT_INPUT_PATH.name
analysis_view = 1 # 1 for first view DTL, 2 for second view FO, etc.
#%
(image, K, analysis_frame) = enumImages.__next__()
image_res = image.shape[-2:]
K = K.to(device)
output_path = DEFAULT_OUTPUT_DIR / f"{analysis_filename}_view{analysis_view}_frame{analysis_frame}.pt"
output_image_path = DEFAULT_OUTPUT_DIR / f"{analysis_filename}_view{analysis_view}_frame{analysis_frame}.png"
client.send_message("/bgImage", output_image_path.as_posix())
#%% Load nms from file
nms_out = torch.load(output_path)

#%% APPLY HEIGHT CONSTRAINT - LINEAR IMPLEMENTATION
# % ! %%time
print("\n" + "="*50)
print("APPLYING HEIGHT CONSTRAINT")
print("="*50)

#%% Get original parameters from detection
# ---------------------------------------------------------
original_betas = nms_out['betas'].clone().detach()
original_pose = nms_out['pose'].clone().detach()
original_trans = nms_out['trans'].clone().detach()

#%% Calculate SMPL defaults (zero translation, default pose, and betas)
# -------------------------------------------------------
zero_trans_list = np.zeros_like(original_trans[0, 0].cpu().numpy()).tolist()  # Convert to list
zero_pose = torch.zeros_like(original_pose)
zero_pose_list = zero_pose[0,0].cpu().numpy().tolist()  # Convert to list
zero_betas = torch.zeros_like(original_betas)
zero_betas_list = zero_betas[0, 0].cpu().numpy().tolist()  # Convert to list

#%% Send zero translation, pose, and betas to Blender
# -------------------------------------------------------
client.send_message("/applysmpltrans", zero_trans_list)
client.send_message("/applysmplpose", zero_pose_list)
client.send_message("/applysmplbetas", zero_betas_list)

#%% Get zero pose and joints from SMPL layer forward model
# -------------------------------------------------------
zero_verts, zero_joints = smpl_layer(zero_pose[0, :, 3:], zero_betas[0])
orig_verts, orig_joints = smpl_layer(original_pose[0, :, 3:], original_betas[0])

#%% Apply only translation and pelvis pose to Blender
# -------------------------------------------------------
original_trans_list = original_trans[0, 0].cpu().numpy().tolist()  # Convert to list
original_pose_list = original_pose[0, 0].cpu().numpy()
original_pose_list[3:] = 0
original_pose_list = original_pose_list.tolist()  # Convert to list

#%% Send pelvis only translation and pose to Blender
# -------------------------------------------------------
original_trans_list = original_trans[0, 0].cpu().numpy().tolist()  # Convert to list
client.send_message("/applysmpltrans", original_trans_list)
client.send_message("/applysmplpose", original_pose_list)

#%% Calculate original model vertices and joints
# -------------------------------------------------------
zero_betas[0][0][0] = 1.0 # Set beta[0] to 1 for zero model for testing
smpl_layer_zero_betas = zero_betas[0]
smpl_layer_zero_pose = zero_pose[0, :, 3:]
smpl_layer_orig_betas = original_betas[0]
smpl_layer_orig_pose = original_pose[0, :, 3:]
#%% TEST SMPL_LAYER
zero_verts, zero_joints = smpl_layer(smpl_layer_zero_pose, smpl_layer_zero_betas)
orig_verts, orig_joints = smpl_layer(smpl_layer_orig_pose, smpl_layer_orig_betas)

#%%
zero_verts_out = zero_verts[0].detach().cpu().numpy()
zero_joints_out = zero_joints[0].detach().cpu().numpy()
orig_verts_out = orig_verts[0].detach().cpu().numpy()
orig_joints_out = orig_joints[0].detach().cpu().numpy()
#% Output ZERO model vertices to Blender on /verts OSC
send_verts = zero_verts_out
send_joints = zero_joints_out
# send_verts = orig_verts_out
# send_joints = orig_joints_out

send_verts = send_verts - 2*send_joints[0]  # Center vertices around pelvis joint
send_joints = send_joints - 2*send_joints[0]  # Center joints around pelvis joint
#% Send selected vertices and joints to Blender
# -------------------------------------------------------
df_verts = pd.DataFrame(send_verts)
df_joints = pd.DataFrame(send_joints)
df_combined = pd.concat([df_verts, df_joints], axis=0)
output_csv_path = DEFAULT_OUTPUT_DIR / f"{analysis_filename}_view{analysis_view}_frame.csv"
df_combined.to_csv(output_csv_path, index=False, header=False)
#%
client.send_message("/verts", output_csv_path.as_posix())

#%% Set blenders model to the original estimation
# -------------------------------------------------------
original_betas = zero_betas
original_trans_list = original_trans[0, 0].cpu().numpy().tolist()  # Convert to list
client.send_message("/applysmpltrans", original_trans_list)
original_pose_list = original_pose[0, 0].cpu().numpy().tolist()  # Convert to list
original_betas_list = original_betas[0, 0].cpu().numpy().tolist()  # Convert to list

zero_joints_list = zero_joints_out.flatten().cpu().numpy().tolist()
# -------------------------------------------------------
client.send_message("/applysmplbetas", zero_betas_list)
time.sleep(0.5)  # Wait for Blender to process the message
client.send_message("/applysmplpose", zero_pose_list)
time.sleep(0.5)  # Wait for Blender to process the message
client.send_message("/adjustjoints", zero_joints_list)
time.sleep(0.5)  # Wait for Blender to process the message
client.send_message("/applysmplpose", original_pose_list)
time.sleep(0.5)  # Wait for Blender to process the message
client.send_message("/applysmpltrans", original_trans_list)
#%%
client.send_message("/set/joints/zero", zero_joints_list)
#%%
client.send_message("/set/pose/orig", original_pose_list)
time.sleep(0.5)  # Wait for Blender to process the message
client.send_message("/set/pose/test", original_pose_list)
time.sleep(0.5)  # Wait for Blender to process the message
client.send_message("/set/pose/zero", original_pose_list)

#%% CALCULATE HEIGHT CORRECTION
# -------------------------------------------------------
# STEP 1: Calculate current height of detected SMPL model
# Always use default pose for height calculation
# pose_for_height = smpl_kin.mean_pose.clone().to(original_betas.device)
pose_for_height = zero_pose.clone().to(original_betas.device)
# Ensure shape [1, 1, n]
# pose_for_height = pose_for_height.unsqueeze(0).unsqueeze(0)
#% Sweep beta[0] from -2 to 2 and record heights based on mesh vertices
import numpy as np
beta0_values = np.arange(2, -2.11, -0.3)
height_table = []

for beta0 in beta0_values:
    test_betas = zero_betas.clone()
    test_betas[..., 0] = beta0
    # Generate mesh vertices in canonical space (default pose)
    # mesh_vertices = smpl_kin(
    #     test_betas, 
    #     pose_for_height, 
    #     original_trans,
    #     output_format="vertices"
    # )
    # zero_verts, zero_joints = smpl_layer(smpl_layer_pose, smpl_layer_betas)
    mesh_vertices, mesh_joints = smpl_layer(pose_for_height[0,:,3:], test_betas[0])
    mesh_vertices = mesh_vertices.detach().cpu().numpy()
    # Height is the difference between max and min Y of all vertices
    # mesh_vertices shape: [batch, num_persons, num_vertices, 3]
    y_coords = mesh_vertices[..., 1]
    top = y_coords.max()
    bottom = y_coords.min()
    current_height = top - bottom
    height_table.append((beta0, current_height))

# Fit a linear model: height = a * beta0 + b
beta0s = np.array([b for b, h in height_table])
heights = np.array([h for b, h in height_table])
# Fit line: height = a * beta0 + b
a_bck, b_bck = np.polyfit(beta0s, heights, 1)

def linear_beta0_for_height(target_height):
    # Invert the linear model: beta0 = (height - b) / a
    return (target_height - b_bck) / a_bck
def linear_height_for_beta0(target_beta0):
    # Use the linear model: height = a * beta0 + b
    return target_beta0 * a_bck + b_bck

original_beta_0 = original_betas[..., 0].cpu().numpy().item()

linear_estimated_beta0 = linear_beta0_for_height(DEFAULT_PERSON_HEIGHT)
linear_estimated_height = linear_height_for_beta0(original_beta_0)

print(f"[Linear] Estimated beta[0] for target height {DEFAULT_PERSON_HEIGHT}m: {linear_estimated_beta0:.2f}")
print(f"[Linear] Estimated height for original beta[0] {original_beta_0:.2f}: {linear_estimated_height:.2f}")

#% STEP 2: Calculate scaling factors needed
scale_factors = DEFAULT_PERSON_HEIGHT / linear_estimated_height
print(f"Required scale factors: {scale_factors:.2f}")
#%%
# STEP 3: Adjust beta[0] to achieve the desired height scaling
# Beta[0] primarily controls overall scale
# Empirical relationship: log scaling works better than linear
beta_0_adjustment = linear_estimated_beta0 - original_beta_0
adjusted_betas = original_betas.clone()
adjusted_betas[..., 0] += beta_0_adjustment
print(f"Beta[0] adjustments: {beta_0_adjustment}")
#%%
# STEP 4: Adjust translation to maintain same apparent size after height scaling
# If person is now taller, move them farther away to maintain same apparent size
# If person is shorter, move them closer
adjusted_trans = original_trans.clone()
adjusted_trans[..., 2] = original_trans[..., 2] * scale_factors  # Scale depth

# STEP 5: Verify the adjustment worked by recalculating height
verification_mesh_vertices = model.smpl_decoder(
    adjusted_betas, 
    pose_for_height, 
    adjusted_trans,
    output_format="vertices"
)

# Calculate adjusted heights from verification_mesh_vertices
# Shape: [batch, num_persons, num_vertices, 3]
y_coords = verification_mesh_vertices[..., 1]
adjusted_heights = y_coords.max(dim=-1).values - y_coords.min(dim=-1).values
current_heights = torch.tensor([h for _, h in height_table], device=adjusted_heights.device)
print(f"Adjusted heights: {adjusted_heights.cpu().numpy()} meters")
print(f"Height error: {torch.abs(adjusted_heights - DEFAULT_PERSON_HEIGHT).cpu().numpy()} meters")


#%
# Send the adjusted trans value to Blender
adjusted_trans_list = adjusted_trans[0, 0].cpu().numpy().tolist()  # Convert to list
client.send_message("/cameraTrans", adjusted_trans_list)
#%%
# STEP 6: Update the detection results with adjusted parameters
height_constrained_out = {}
for k, v in nms_out.items():
    height_constrained_out[k] = v.clone()

height_constrained_out['betas'] = adjusted_betas
height_constrained_out['trans'] = adjusted_trans

# STEP 7: Recalculate 3D and 2D coordinates with new parameters
adjusted_3d = model.smpl_decoder(
    adjusted_betas,
    original_pose, 
    adjusted_trans,
    output_format="joints_face"
)

# Project to 2D using camera intrinsics  
from comotion_demo.utils import helper
import numpy as np
from comotion_demo.utils.dataloading import convert_tensor_to_image
from PIL import Image
adjusted_2d = helper.project_to_2d(K, adjusted_3d)

height_constrained_out['pred_3d'] = adjusted_3d
height_constrained_out['pred_2d'] = adjusted_2d

# STEP 8: Print detailed summary
print(f"\nDETAILED SUMMARY:")
print(f"Number of detections processed: {len(current_heights)}")

# Print changes for each detection (or just first few if many)
num_to_show = min(3, len(current_heights))
for i in range(num_to_show):
    print(f"\nDetection {i+1}:")
    print(f"  Original height: {current_heights[i].item():.3f}m")
    print(f"  Scale factor: {scale_factors[i].item():.3f}")
    print(f"  Final height: {adjusted_heights[i].item():.3f}m")
    print(f"  Beta[0]: {original_betas[0, i, 0].item():.3f} -> {adjusted_betas[0, i, 0].item():.3f} (Δ{beta_0_adjustment[i].item():.3f})")
    
    orig_trans = original_trans[0, i].cpu().numpy()
    adj_trans = adjusted_trans[0, i].cpu().numpy()
    print(f"  Translation: [{orig_trans[0]:.3f}, {orig_trans[1]:.3f}, {orig_trans[2]:.3f}] -> [{adj_trans[0]:.3f}, {adj_trans[1]:.3f}, {adj_trans[2]:.3f}]")
    print(f"  Depth change: {adj_trans[2] - orig_trans[2]:.3f}m")

if len(current_heights) > num_to_show:
    print(f"  ... and {len(current_heights) - num_to_show} more detections")

# Use the height-constrained results
final_detection = height_constrained_out
print(f"\n✓ Height constraint applied successfully!")
    
print("="*50)

#%% OPTIONAL: Compare original vs height-constrained results
if DEFAULT_PERSON_HEIGHT is not None and nms_out['betas'].shape[1] > 0:
    print("\nCOMPARISON - Original vs Height-Constrained:")
    print(f"Original betas[0]: {nms_out['betas'][0, 0, 0].item():.4f}")
    print(f"Adjusted betas[0]: {final_detection['betas'][0, 0, 0].item():.4f}")
    print(f"Original trans Z: {nms_out['trans'][0, 0, 2].item():.4f}")
    print(f"Adjusted trans Z: {final_detection['trans'][0, 0, 2].item():.4f}")

#%% Continue with final_detection for any further processing...
# final_detection now contains the height-constrained detection results
# You can use this instead of nms_out for rendering, tracking, etc.

print(f"\nFinal detection contains {final_detection['betas'].shape[1]} person(s)")
if final_detection['betas'].shape[1] > 0:
    print(f"First person's beta[0]: {final_detection['betas'][0, 0, 0].item():.4f}")
    print(f"First person's translation: {final_detection['trans'][0, 0].cpu().numpy()}")

#%%