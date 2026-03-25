import open3d as o3d

m = o3d.io.read_triangle_mesh(r'outputs\wheel_mesh\wheel_mesh_smooth.ply')
m.compute_vertex_normals()
print(m)
o3d.visualization.draw_geometries(
    [m],
    mesh_show_back_face=True,
    window_name='wheel_mesh_smooth'
)