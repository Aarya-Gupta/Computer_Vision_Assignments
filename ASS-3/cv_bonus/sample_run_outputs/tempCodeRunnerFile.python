# -*- coding: utf-8 -*-
"""
2022006_5_improved.py

Complete script for Point Cloud Registration using RANSAC + Point-to-Plane ICP.
Processes a sequence of point clouds to estimate trajectory and build a global map.
"""

import numpy as np
import open3d as o3d
import copy
import os
from scipy.stats import ortho_group
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate # Optional, used only for experiment summary if run

# === Core Helper Functions ===

def load_point_cloud(file_path):
    """Load a point cloud from file."""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            print(f"Warning: No points found in {file_path}")
            return None
        return pcd
    except Exception as e:
        print(f"Error loading point cloud {file_path}: {e}")
        return None

def preprocess_point_cloud(pcd, voxel_size=0.05, estimate_normals=True, knn=30):
    """Preprocess point cloud: downsample and estimate normals."""
    if pcd is None:
        return None
    
    # Create a copy to avoid modifying the original
    processed_pcd = copy.deepcopy(pcd)

    # Downsample using voxel grid filter
    # print(f"    Downsampling with voxel size {voxel_size}...") # Verbose
    processed_pcd = processed_pcd.voxel_down_sample(voxel_size)
    if not processed_pcd.has_points():
        print("Warning: Point cloud has no points after downsampling.")
        return None

    # Estimate normals if requested
    if estimate_normals:
        # print(f"    Estimating normals with KNN={knn}...") # Verbose
        search_param = o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        processed_pcd.estimate_normals(search_param=search_param)
        # Orient normals consistently
        processed_pcd.orient_normals_consistent_tangent_plane(k=knn)
        if not processed_pcd.has_normals():
             print("Warning: Failed to estimate normals.")

    return processed_pcd

def get_random_orthogonal_matrix():
    """Generate a random valid 4x4 transformation matrix."""
    R = ortho_group.rvs(3)
    t = np.random.rand(3, 1) * 0.1 # Small random translation

    initial_transform = np.eye(4)
    initial_transform[:3, :3] = R
    initial_transform[:3, 3] = t.squeeze()

    return initial_transform

def get_initial_transform_from_ransac(source, target, voxel_size, with_normals=True):
    """Use RANSAC for initial alignment (global registration)."""
    if source is None or target is None or not source.has_points() or not target.has_points():
        print("Error: Invalid input point clouds for RANSAC.")
        return np.eye(4)

    distance_threshold = voxel_size * 1.5 # Max correspondence distance for RANSAC

    if with_normals:
        if not source.has_normals() or not target.has_normals():
            print("Error: Normals required for feature matching RANSAC but not found.")
            return np.eye(4)
            
        # print("    Computing FPFH features...") # Verbose
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

        if source_fpfh is None or target_fpfh is None:
             print("Error: FPFH feature computation failed.")
             return np.eye(4)

        # print("    Running RANSAC based on features...") # Verbose
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            mutual_filter=True, # Requires Open3D >= 0.13
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), # Point-to-Point for RANSAC transform estimation itself
            ransac_n=4, # Typically 3 or 4 points
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999) # Max iterations, confidence
            # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500) # Original higher params
        )
        return result.transformation
    else:
        # RANSAC based on correspondences (if pick_correspondences exists)
        # This part had issues in the original code, FPFH is generally preferred.
        print("Warning: RANSAC without normals is not fully supported/recommended here. Using FPFH method.")
        # Fallback to trying FPFH anyway if normals happen to be present
        if source.has_normals() and target.has_normals():
             return get_initial_transform_from_ransac(source, target, voxel_size, with_normals=True)
        else:
             print("Error: Cannot run RANSAC without normals or pick_correspondences.")
             return np.eye(4)


def run_icp(source, target, initial_transform,
            max_iteration=100, threshold=0.05,
            knn=30, voxel_size=0.05, use_point_to_plane=True):
    """Run ICP (Point-to-Plane recommended) for refinement."""

    if source is None or target is None or not source.has_points() or not target.has_points():
        print("Error: Invalid input point clouds for ICP.")
        return np.eye(4), 0.0, 0.0, 0.0, 0.0

    # Ensure point clouds have normals if using Point-to-Plane
    if use_point_to_plane:
        if not source.has_normals():
            print("    Estimating normals for source in ICP...")
            source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
            source.orient_normals_consistent_tangent_plane(k=knn)
        if not target.has_normals():
            print("    Estimating normals for target in ICP...")
            target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
            target.orient_normals_consistent_tangent_plane(k=knn)
        
        if not source.has_normals() or not target.has_normals():
             print("Warning: Failed to estimate normals for Point-to-Plane ICP. Falling back to Point-to-Point.")
             use_point_to_plane = False # Fallback

    if initial_transform is None:
        print("Warning: initial_transform is None in run_icp. Using identity.")
        initial_transform = np.eye(4) # Use identity, RANSAC should have provided one

    # Choose estimation method
    if use_point_to_plane:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    # Evaluate initial alignment (optional, for logging)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, initial_transform)
    initial_fitness = evaluation.fitness
    initial_rmse = evaluation.inlier_rmse
    # print(f"    ICP Initial - Fitness: {initial_fitness:.4f}, RMSE: {initial_rmse:.4f}") # Verbose

    # Run ICP
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, initial_transform,
        estimation_method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

    # Get results
    transformation = reg_result.transformation
    fitness = reg_result.fitness
    inlier_rmse = reg_result.inlier_rmse

    return transformation, initial_fitness, initial_rmse, fitness, inlier_rmse


# === Sequence Registration Function ===

def register_all_point_clouds_with_ransac(dataset_dir, pcd_files,
                                          ransac_voxel_size=0.05,
                                          icp_threshold=0.05,
                                          icp_max_iteration=200,
                                          knn=30):
    """
    Register all point clouds using RANSAC initial alignment + Point-to-Plane ICP refinement.
    """
    trajectory = []
    registered_pcds = [] # Stores original resolution clouds transformed to global frame
    registration_metrics = []

    # Load and prepare the first point cloud
    first_pcd_path = os.path.join(dataset_dir, pcd_files[0])
    first_pcd = load_point_cloud(first_pcd_path)
    if first_pcd is None:
        print("Error: Could not load the first point cloud. Aborting.")
        return None, None, None
        
    # Preprocess first cloud (downsample, estimate normals) - needed as target
    print(f"Preprocessing cloud 0 ({pcd_files[0]})...")
    first_pcd_processed = preprocess_point_cloud(first_pcd,
                                                 voxel_size=ransac_voxel_size,
                                                 estimate_normals=True, knn=knn)
    if first_pcd_processed is None:
        print("Error: Failed to preprocess the first point cloud. Aborting.")
        return None, None, None

    registered_pcds.append(copy.deepcopy(first_pcd)) # Store original high-res cloud
    global_transform = np.eye(4)
    trajectory.append(global_transform)

    prev_pcd_processed = first_pcd_processed # Keep track of the *processed* previous cloud

    # Process subsequent pairs
    for i in range(1, len(pcd_files)):
        print("-" * 40)
        print(f"Processing pair {i-1}-{i} : ({pcd_files[i-1]} -> {pcd_files[i]})")
        print("-" * 40)

        # Load current point cloud (original resolution)
        current_pcd_path = os.path.join(dataset_dir, pcd_files[i])
        current_pcd = load_point_cloud(current_pcd_path)
        if current_pcd is None:
            print(f"Warning: Skipping pair {i-1}-{i} due to loading error for {pcd_files[i]}. Using identity transform.")
            relative_transform = np.eye(4)
            final_fitness = 0.0
            final_rmse = -1.0 # Indicate failure
            init_f = 0.0
            init_e = 0.0
        else:
            # Preprocess current point cloud for RANSAC and ICP
            print(f"Preprocessing cloud {i}...")
            current_pcd_processed = preprocess_point_cloud(current_pcd,
                                                          voxel_size=ransac_voxel_size,
                                                          estimate_normals=True, knn=knn)
            if current_pcd_processed is None:
                 print(f"Warning: Skipping pair {i-1}-{i} due to preprocessing error for {pcd_files[i]}. Using identity transform.")
                 relative_transform = np.eye(4)
                 final_fitness = 0.0
                 final_rmse = -1.0 # Indicate failure
                 init_f = 0.0
                 init_e = 0.0
            else:
                # --- RANSAC for Initial Alignment ---
                print("Running RANSAC for initial alignment...")
                # Note: RANSAC estimates transform from source (current) to target (prev)
                initial_transform_ransac = get_initial_transform_from_ransac(
                    current_pcd_processed, prev_pcd_processed, # Use processed clouds
                    ransac_voxel_size,
                    with_normals=True)

                # Evaluate RANSAC result (optional logging)
                eval_ransac = o3d.pipelines.registration.evaluate_registration(
                    current_pcd_processed, prev_pcd_processed, icp_threshold, initial_transform_ransac) # Use ICP threshold for comparable eval
                print(f"RANSAC Initial Guess - Fitness: {eval_ransac.fitness:.4f}, RMSE: {eval_ransac.inlier_rmse:.4f}")
                if eval_ransac.fitness < 0.01: # If RANSAC found very few correspondences
                    print("Warning: RANSAC fitness very low. Result might be unreliable.")
                    # Optionally fallback to identity or random if RANSAC fails badly
                    # initial_transform_ransac = np.eye(4)


                # --- Point-to-Plane ICP Refinement ---
                print("Running Point-to-Plane ICP for refinement...")
                # ICP refines the transform from source (current) to target (prev)
                transformation_icp, init_f, init_e, final_fitness, final_rmse = run_icp(
                    current_pcd_processed, prev_pcd_processed,
                    initial_transform=initial_transform_ransac,  # Use RANSAC result!
                    max_iteration=icp_max_iteration,
                    threshold=icp_threshold,
                    use_point_to_plane=True, # Use Point-to-Plane
                    knn=knn,
                    voxel_size=ransac_voxel_size
                )

                # --- Calculate Relative Motion ---
                # We found T_curr_to_prev. Relative motion T_prev_to_curr is inv(T_curr_to_prev)
                # Check ICP result quality before inverting and applying
                if final_fitness < 0.1: # Adjust threshold as needed
                     print(f"Warning: Low ICP fitness ({final_fitness:.3f}) for pair {i-1}-{i}. Result might be inaccurate.")
                     # Decide on fallback: use RANSAC only? Use Identity?
                     # Using identity might stop drift but lose motion estimate.
                     # Using RANSAC only might be better than bad ICP.
                     # Let's try RANSAC result if ICP fails badly
                     print("Using RANSAC transform directly due to low ICP fitness.")
                     relative_transform = np.linalg.inv(initial_transform_ransac)
                     # Or fallback to identity if RANSAC was also bad?
                     # if eval_ransac.fitness < 0.01: relative_transform = np.eye(4)

                elif final_rmse > icp_threshold * 2 : # If RMSE is much larger than threshold
                    print(f"Warning: High ICP RMSE ({final_rmse:.3f}) for pair {i-1}-{i}. Result might be inaccurate.")
                    print("Using RANSAC transform directly due to high ICP RMSE.")
                    relative_transform = np.linalg.inv(initial_transform_ransac)

                else:
                    # ICP result seems reasonable, use its inverse for relative motion
                     relative_transform = np.linalg.inv(transformation_icp)


        # Update global transform and trajectory
        global_transform = global_transform @ relative_transform
        trajectory.append(copy.deepcopy(global_transform))

        # Transform the *original* high-resolution current point cloud to global frame
        if current_pcd is not None: # Only transform if loaded correctly
            transformed_pcd = copy.deepcopy(current_pcd)
            transformed_pcd.transform(global_transform)
            registered_pcds.append(transformed_pcd)
        else:
            # Append an empty cloud or handle missing data appropriately
            registered_pcds.append(o3d.geometry.PointCloud())


        # Store metrics for this pair
        registration_metrics.append({
            'pair': f"{i-1}-{i}",
            'source': pcd_files[i],
            'target': pcd_files[i-1],
            'ransac_fitness': eval_ransac.fitness if 'eval_ransac' in locals() else -1,
            'ransac_rmse': eval_ransac.inlier_rmse if 'eval_ransac' in locals() else -1,
            'initial_fitness_icp': init_f,
            'initial_rmse_icp': init_e,
            'final_fitness': final_fitness,
            'final_rmse': final_rmse,
        })
        del locals()['eval_ransac'] # Clear eval_ransac for next loop


        print(f"Registration {i}/{len(pcd_files)-1} - ICP Fitness: {final_fitness:.6f}, ICP RMSE: {final_rmse:.6f}")

        # Update the previous processed cloud for the next iteration's target
        if current_pcd_processed is not None:
             prev_pcd_processed = current_pcd_processed
        # else: prev_pcd_processed remains the same as the last successful one, which might cause issues.
        # A better strategy might be needed if preprocessing fails.

    metrics_df = pd.DataFrame(registration_metrics)
    return registered_pcds, trajectory, metrics_df

# === Visualization and Analysis Functions ===

def save_trajectory_to_csv(trajectory, output_path):
    """Save the trajectory (poses) to a CSV file."""
    positions = []
    for frame_idx, transform in enumerate(trajectory):
        x, y, z = transform[:3, 3]
        rotation = transform[:3, :3]

        # ZYX Euler angles (yaw, pitch, roll) - Handle potential gimbal lock issues if needed
        # Note: arctan2 handles quadrants correctly. Check angle ranges if specific conventions needed.
        # Check for near gimbal lock condition (when cos(pitch) is close to 0)
        pitch = np.arcsin(-rotation[2, 0])
        if np.isclose(np.cos(pitch), 0.0):
            # Gimbal lock: Set roll to 0 and calculate yaw relative to Z
            roll = 0.0
            yaw = np.arctan2(rotation[0, 1], rotation[1, 1]) # Use alternative formula
            print(f"Warning: Potential gimbal lock detected at frame {frame_idx}. Roll set to 0.")
        else:
            roll = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

        roll_deg, pitch_deg, yaw_deg = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

        positions.append({
            'x': x, 'y': y, 'z': z,
            'roll_deg': roll_deg, 'pitch_deg': pitch_deg, 'yaw_deg': yaw_deg
        })

    trajectory_df = pd.DataFrame(positions)
    try:
        trajectory_df.to_csv(output_path, index_label='frame')
        print(f"Trajectory saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving trajectory to CSV: {e}")

    return trajectory_df

def plot_trajectory_3d(trajectory_df, filename='turtlebot_trajectory_3d.png'):
    """Plot the estimated 3D trajectory."""
    if trajectory_df is None or trajectory_df.empty:
        print("Cannot plot empty trajectory.")
        return None, None
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(trajectory_df['x'], trajectory_df['y'], trajectory_df['z'], 'b-', linewidth=2, label='Robot Path')
    ax.scatter(trajectory_df['x'], trajectory_df['y'], trajectory_df['z'], c='red', s=30, label='Waypoints', alpha=0.7)
    ax.scatter(trajectory_df['x'].iloc[0], trajectory_df['y'].iloc[0], trajectory_df['z'].iloc[0],
               c='lime', s=100, label='Start', edgecolors='k')
    ax.scatter(trajectory_df['x'].iloc[-1], trajectory_df['y'].iloc[-1], trajectory_df['z'].iloc[-1],
               c='purple', s=100, label='End', edgecolors='k')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Estimated TurtleBot 3D Trajectory (RANSAC + P2Plane ICP)')
    
    # Make axes equal scale for better visualization
    try:
        all_coords = trajectory_df[['x', 'y', 'z']].values.flatten()
        max_range = np.array([trajectory_df['x'].max()-trajectory_df['x'].min(),
                              trajectory_df['y'].max()-trajectory_df['y'].min(),
                              trajectory_df['z'].max()-trajectory_df['z'].min()]).max() / 2.0

        mid_x = (trajectory_df['x'].max()+trajectory_df['x'].min()) * 0.5
        mid_y = (trajectory_df['y'].max()+trajectory_df['y'].min()) * 0.5
        mid_z = (trajectory_df['z'].max()+trajectory_df['z'].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    except Exception as e:
        print(f"Could not set equal axes: {e}")


    ax.legend()
    ax.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"3D trajectory plot saved to '{filename}'")
    plt.close(fig) # Close plot to free memory
    return fig, ax

def plot_trajectory_2d(trajectory_df, filename='turtlebot_trajectory_2d.png'):
    """Plot the estimated 2D (X-Y) trajectory projection."""
    if trajectory_df is None or trajectory_df.empty:
        print("Cannot plot empty trajectory.")
        return
        
    plt.figure(figsize=(10, 10))

    plt.plot(trajectory_df['x'], trajectory_df['y'], 'b-', linewidth=2, label='Robot Path')
    plt.scatter(trajectory_df['x'], trajectory_df['y'], c='red', s=30, label='Waypoints', alpha=0.7)
    plt.scatter(trajectory_df['x'].iloc[0], trajectory_df['y'].iloc[0], c='lime', s=100, label='Start', edgecolors='k')
    plt.scatter(trajectory_df['x'].iloc[-1], trajectory_df['y'].iloc[-1], c='purple', s=100, label='End', edgecolors='k')

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Estimated TurtleBot 2D Trajectory (Top-Down View, RANSAC + P2Plane ICP)')
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"2D trajectory plot saved to '{filename}'")
    plt.close() # Close plot

def visualize_combined_point_cloud(registered_pcds, trajectory=None, downsample_voxel_size=0.05, save_path="global_registered_pointcloud_ransac_p2plane.pcd"):
    """Visualize the combined registered point cloud map and optionally the trajectory."""
    if not registered_pcds:
        print("No registered point clouds to visualize.")
        return

    map_vis_elements = []
    combined_pcd_for_saving = o3d.geometry.PointCloud()

    # Add registered point clouds (colored uniquely)
    num_clouds = len(registered_pcds)
    for i, pcd in enumerate(registered_pcds):
        if pcd is None or not pcd.has_points():
            continue
            
        # Downsample for visualization performance
        pcd_down = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
        if not pcd_down.has_points():
            continue

        pcd_colored = copy.deepcopy(pcd_down)

        # Assign distinct color using HSV space
        hue = i / float(num_clouds) if num_clouds > 0 else 0
        color = plt.cm.hsv(hue)[:3] # Get RGB from HSV colormap
        pcd_colored.paint_uniform_color(color)
        map_vis_elements.append(pcd_colored)

        # Add to combined cloud for saving (use downsampled version to keep file size manageable)
        combined_pcd_for_saving += pcd_down

    # Add trajectory visualization if provided
    if trajectory is not None and len(trajectory) > 1:
        line_set = o3d.geometry.LineSet()
        points = np.array([T[:3, 3] for T in trajectory])
        lines = [[i, i + 1] for i in range(len(points) - 1)]
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))]) # Red trajectory
        map_vis_elements.append(line_set)

        # Add coordinate frames at key poses
        step = max(1, len(trajectory) // 10) # Show ~10 frames
        for i in range(0, len(trajectory), step):
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            coord_frame.transform(trajectory[i])
            map_vis_elements.append(coord_frame)

    # Visualize
    if map_vis_elements:
        print("Visualizing combined map and trajectory...")
        o3d.visualization.draw_geometries(map_vis_elements)
    else:
        print("No elements to visualize.")

    # Save the combined point cloud
    if combined_pcd_for_saving.has_points():
        try:
            o3d.io.write_point_cloud(save_path, combined_pcd_for_saving)
            print(f"Saved global registered point cloud to '{save_path}'")
        except Exception as e:
            print(f"Error saving combined point cloud: {e}")
    else:
        print("No points in combined cloud to save.")


# === Main Execution Block ===

def main():
    # --- Configuration ---
    dataset_dir = "C:/Users/aarya/Github_Projects/Striver_A2Z/cv_bonus/selected_pcds" # IMPORTANT: Update this path
    output_dir = "." # Directory to save outputs (CSV, PNG, PCD)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Registration Parameters
    ransac_voxel_size = 0.05    # Voxel size for preprocessing before RANSAC/ICP
    icp_threshold = 0.05        # Max correspondence distance for ICP refinement
    icp_max_iteration = 200     # Max iterations for ICP refinement
    knn_normals = 30            # K-neighbors for normal estimation

    # Visualization/Saving Parameters
    map_vis_voxel_size = 0.05   # Voxel size for downsampling the final map visualization/saving
    trajectory_csv_file = os.path.join(output_dir, "turtlebot_trajectory_ransac_p2plane.csv")
    metrics_csv_file = os.path.join(output_dir, "registration_metrics_ransac_p2plane.csv")
    plot_3d_file = os.path.join(output_dir, "turtlebot_trajectory_3d_ransac_p2plane.png")
    plot_2d_file = os.path.join(output_dir, "turtlebot_trajectory_2d_ransac_p2plane.png")
    global_map_file = os.path.join(output_dir, "global_registered_pointcloud_ransac_p2plane.pcd")
    summary_report_file = os.path.join(output_dir, "registration_summary_ransac_p2plane.txt")
    # --- End Configuration ---


    # Get sorted list of point cloud files
    try:
        all_files = os.listdir(dataset_dir)
    except FileNotFoundError:
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
        
    pcd_files = sorted([f for f in all_files if f.startswith("pointcloud_") and f.endswith(".pcd")])

    if not pcd_files:
        print(f"Error: No PCD files found in '{dataset_dir}'. Check the path and file naming.")
        return
        
    print(f"Found {len(pcd_files)} point cloud files.")

    # --- Run Registration ---
    print("\nRegistering all point clouds using RANSAC + Point-to-Plane ICP:")
    print(f"- RANSAC/ICP Voxel Size: {ransac_voxel_size}")
    print(f"- ICP Distance Threshold: {icp_threshold}")
    print(f"- ICP Max Iterations: {icp_max_iteration}")
    print(f"- KNN for Normals: {knn_normals}")

    registered_pcds, trajectory, metrics_df = register_all_point_clouds_with_ransac(
        dataset_dir, pcd_files,
        ransac_voxel_size=ransac_voxel_size,
        icp_threshold=icp_threshold,
        icp_max_iteration=icp_max_iteration,
        knn=knn_normals
    )

    if registered_pcds is None or trajectory is None or metrics_df is None:
        print("\nRegistration failed.")
        return

    print("\nRegistration complete!")
    print(f"Registered {len(registered_pcds)} point clouds.")
    print(f"Trajectory has {len(trajectory)} poses.")

    # --- Save and Analyze Results ---
    try:
        metrics_df.to_csv(metrics_csv_file, index=False)
        print(f"Saved registration metrics to '{metrics_csv_file}'")
    except Exception as e:
        print(f"Error saving metrics CSV: {e}")

    trajectory_df = save_trajectory_to_csv(trajectory, trajectory_csv_file)

    if trajectory_df is not None and not trajectory_df.empty:
        plot_trajectory_3d(trajectory_df, filename=plot_3d_file)
        plot_trajectory_2d(trajectory_df, filename=plot_2d_file)

        # --- Print Summary ---
        print("\n--- Summary of Registration Results ---")
        avg_fitness = metrics_df['final_fitness'].mean()
        avg_rmse = metrics_df['final_rmse'][metrics_df['final_rmse'] >= 0].mean() # Exclude -1 errors
        print(f"Average final ICP fitness: {avg_fitness:.4f}")
        print(f"Average final ICP RMSE (valid pairs): {avg_rmse:.4f}")

        # Calculate total distance
        diffs = np.diff(trajectory_df[['x', 'y', 'z']].values, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        total_distance = np.sum(segment_lengths)
        print(f"Total estimated trajectory distance: {total_distance:.4f} meters")

        # Save summary report
        try:
            with open(summary_report_file, "w") as f:
                f.write("Point Cloud Registration Summary (RANSAC + Point-to-Plane ICP)\n")
                f.write("============================================================\n\n")
                f.write(f"Number of point clouds processed: {len(pcd_files)}\n")
                f.write(f"Parameters Used:\n")
                f.write(f"- RANSAC/ICP Voxel Size: {ransac_voxel_size}\n")
                f.write(f"- ICP Distance Threshold: {icp_threshold}\n")
                f.write(f"- ICP Max Iterations: {icp_max_iteration}\n")
                f.write(f"- KNN for Normals: {knn_normals}\n\n")
                f.write(f"Registration Results (Averages):\n")
                f.write(f"- Average Final ICP Fitness: {avg_fitness:.4f}\n")
                f.write(f"- Average Final ICP RMSE (valid pairs): {avg_rmse:.4f}\n\n")
                # Include RANSAC stats if desired
                # avg_ransac_f = metrics_df['ransac_fitness'][metrics_df['ransac_fitness'] >= 0].mean()
                # avg_ransac_e = metrics_df['ransac_rmse'][metrics_df['ransac_rmse'] >= 0].mean()
                # f.write(f"- Average RANSAC Fitness: {avg_ransac_f:.4f}\n")
                # f.write(f"- Average RANSAC RMSE: {avg_ransac_e:.4f}\n\n")

                f.write(f"Trajectory Information:\n")
                f.write(f"- Total distance traveled: {total_distance:.4f} meters\n")
                f.write(f"- Bounding box (X): {trajectory_df['x'].min():.2f} to {trajectory_df['x'].max():.2f} meters\n")
                f.write(f"- Bounding box (Y): {trajectory_df['y'].min():.2f} to {trajectory_df['y'].max():.2f} meters\n")
                f.write(f"- Bounding box (Z): {trajectory_df['z'].min():.2f} to {trajectory_df['z'].max():.2f} meters\n")
            print(f"\nSaved summary report to '{summary_report_file}'")
        except Exception as e:
            print(f"Error writing summary report: {e}")

    else:
        print("Trajectory data is empty, cannot plot or calculate distance.")


    # --- Visualize Final Map ---
    visualize_combined_point_cloud(registered_pcds, trajectory,
                                   downsample_voxel_size=map_vis_voxel_size,
                                   save_path=global_map_file)

    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()