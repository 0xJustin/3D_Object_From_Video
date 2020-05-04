# 3D_Object_From_Video
Creating 3D objects for printing based on videos in the wild

# Main file
Full pipeline.ipynb

# Dependencies
1. Add pre-trained model to root directory. <br/>
...Dense Depth Pre-trained Model: https://s3-eu-west-1.amazonaws.com/densedepth/nyu.h5 <br/>
2. Run: <code>python3 3d_reconstruction.py</code>


Open3d library installation:
pip: pip install open3d==0.8.0.0
conda: conda install -c open3d-admin open3d==0.8.0.0
