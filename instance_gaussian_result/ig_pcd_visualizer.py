import numpy as np
import open3d as o3d
from plyfile import PlyData

def load_gaussian_ply(filename):
    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex'].data

    xyz = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)

    # instance gaussian stores features in 6 variables: 'ins_feat_r', 'ins_feat_g', 'ins_feat_b' and 'ins_feat_r2', 'ins_feat_g2', 'ins_feat_b2'
    # to visualize the other 3 instance features please add a "2" in to the attributes! (in line 13, 15, 16 and 17)
    if all(attr in vertex_data.dtype.names for attr in ['ins_feat_r', 'ins_feat_g', 'ins_feat_b']):
        colors = np.stack([
            vertex_data['ins_feat_r'],
            vertex_data['ins_feat_g'],
            vertex_data['ins_feat_b']
        ], axis=-1)
        colors = np.clip(colors, 0.0, 1.0)
    else:
        colors = None

    opacity = vertex_data['opacity'] if 'opacity' in vertex_data.dtype.names else None
    return xyz, colors, opacity

def visualize_point_cloud(xyz, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = 1.7

    vis.run()
    vis.destroy_window()


# small script to visualize learned instance features for the instance gaussian pointclouds
if __name__ == "__main__":
    filename = "ramen.ply"
    #filename = "ramen/point_cloud.ply"
    #filename = "waldo_kitchen/point_cloud/iteration_30000/point_cloud.ply"
    #filename = "teatime/point_cloud/iteration_30000/point_cloud.ply"

    xyz, colors, opacity = load_gaussian_ply(filename)
    visualize_point_cloud(xyz, colors)
