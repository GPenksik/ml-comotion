import bpy

def check_smpl_objects():
    """
    Check if all required SMPL objects exist in the Blender scene.
    This function verifies the presence of SMPL Rigs, Armatures, Meshes, Vertices and Joints
    for all three variations: orig, zero, and test.
    
    Returns:
        dict: Dictionary with status of all object groups
    """
    # Define all expected objects
    expected_objects = {
        "rigs": ["SMPL_Rig_orig", "SMPL_Rig_zero", "SMPL_Rig_test"],
        "armatures": ["Armature_orig", "Armature_zero", "Armature_test"],
        "meshes": ["m_avg_orig", "m_avg_zero", "m_avg_test"],
        "vertices": ["smpl_verts_orig", "smpl_verts_zero", "smpl_verts_test"],
        "joints": ["smpl_joints_orig", "smpl_joints_zero", "smpl_joints_test"]
    }
    
    # Results dictionary
    results = {
        "orig": {"status": True, "missing": []},
        "zero": {"status": True, "missing": []},
        "test": {"status": True, "missing": []}
    }
    
    # Check for all objects
    for category, objects in expected_objects.items():
        for obj_name in objects:
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                # Determine which variation this belongs to
                if obj_name.endswith("_orig"):
                    results["orig"]["status"] = False
                    results["orig"]["missing"].append(obj_name)
                elif obj_name.endswith("_zero"):
                    results["zero"]["status"] = False
                    results["zero"]["missing"].append(obj_name)
                elif obj_name.endswith("_test"):
                    results["test"]["status"] = False
                    results["test"]["missing"].append(obj_name)
    
    # Print results to console
    print("\n=== SMPL OBJECTS CHECK ===")
    
    for variation in ["orig", "zero", "test"]:
        if results[variation]["status"]:
            print(f"✓ {variation.upper()} objects: All present")
        else:
            print(f"✗ {variation.upper()} objects: Missing {len(results[variation]['missing'])} objects")
            for missing in results[variation]["missing"]:
                print(f"  - {missing}")
    
    print("========================\n")
    return results

def register():
    """Register function for Blender add-on"""
    pass

def unregister():
    """Unregister function for Blender add-on"""
    pass

if __name__ == "__main__":
    check_smpl_objects()
