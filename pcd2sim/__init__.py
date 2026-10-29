"""pcd2sim: map (PCD / OctoMap) -> MuJoCo-ready scene directory.

SCENE_FORMAT is the version written to meta.json["format_version"] and checked by quadsim.sim on load.
  2: ground.npz slabs + NaN zmean under slabs
  3: contact.npy (bool per box: boxes a ground robot can touch, see voxel.contact_mask) and a per-voxel
     `contact` mask in voxels.npz; boxes are merged separately inside and outside the contact band so a
     wall box never spans both
"""
SCENE_FORMAT = 3
