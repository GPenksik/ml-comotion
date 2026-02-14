#%% Get Originals
from http import client


original_betas = nms_out['betas'].clone().detach()
original_pose = nms_out['pose'].clone().detach()
original_trans = nms_out['trans'].clone().detach()
# Calculate zero model for testing
zero_pose = torch.zeros_like(original_pose)
zero_betas = torch.zeros_like(original_betas)
# Define test params
test_betas = zero_betas.clone().detach()
test_betas[0][0][0] = 1.0  # Set beta[0] to 1 for test model for testing
test_pose = zero_pose.clone().detach()
# Calculate vertices and joints
orig_verts, orig_joints = smpl_layer(original_pose[0, :, 3:], original_betas[0])
zero_verts, zero_joints = smpl_layer(zero_pose[0, :, 3:], zero_betas[0])
test_verts, test_joints = smpl_layer(test_pose[0, :, 3:], test_betas[0])
# Convert to lists to send via OSC
orig_trans_list = original_trans[0, 0].cpu().numpy().tolist()
orig_pose_list = original_pose[0, 0].cpu().numpy().tolist()
orig_betas_list = original_betas[0, 0].cpu().numpy().tolist()
zero_trans_list = np.zeros_like(original_trans[0, 0].cpu().numpy()).tolist()
zero_pose_list = zero_pose[0, 0].cpu().numpy().tolist()
zero_betas_list = zero_betas[0, 0].cpu().numpy().tolist()
test_pose_list = test_pose[0, 0].cpu().numpy().tolist()  
test_betas_list = test_betas[0, 0].cpu().numpy().tolist()  

orig_verts_list = orig_verts[0].cpu().numpy().tolist()  
orig_joints_list = orig_joints[0].cpu().numpy().tolist()
zero_verts_list = zero_verts[0].cpu().numpy().tolist()  
zero_joints_list = zero_joints[0].cpu().numpy().tolist()  
test_verts_list = test_verts[0].cpu().numpy().tolist()  
test_joints_list = test_joints[0].cpu().numpy().tolist()  
# Save verts to .csv files
verts_orig_out_path = DEFAULT_OUTPUT_DIR / f"{analysis_filename}_view{analysis_view}_frame_orig.csv"
verts_zero_out_path = DEFAULT_OUTPUT_DIR / f"{analysis_filename}_view{analysis_view}_frame_zero.csv"
verts_test_out_path = DEFAULT_OUTPUT_DIR / f"{analysis_filename}_view{analysis_view}_frame_test.csv"
pd.DataFrame(orig_verts_list).to_csv(verts_orig_out_path, index=False, header=False)
pd.DataFrame(zero_verts_list).to_csv(verts_zero_out_path, index=False, header=False)
pd.DataFrame(test_verts_list).to_csv(verts_test_out_path, index=False, header=False)

#%% Send all data to Blender
client.send_message("/set/verts_path/orig", verts_orig_out_path.as_posix())
client.send_message("/set/verts_path/zero", verts_zero_out_path.as_posix())
client.send_message("/set/verts_path/test", verts_test_out_path.as_posix())
client.send_message("/set/joints/orig", orig_joints_list)
client.send_message("/set/joints/zero", zero_joints_list)
client.send_message("/set/joints/test", test_joints_list)
client.send_message("/set/joints/orig", orig_joints_list)
client.send_message("/set/joints/zero", zero_joints_list)
client.send_message("/set/joints/test", test_joints_list)
client.send_message("/set/pose/orig", orig_pose_list)
client.send_message("/set/pose/zero", zero_pose_list)
client.send_message("/set/pose/test", test_pose_list)
# client.send_message("/set/betas/orig", orig_betas_list)
# client.send_message("/set/betas/test", test_betas_list)
