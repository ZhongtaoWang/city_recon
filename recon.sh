input_img_path=$1

# Define the working directory
work_dir="work_dir"

# Remove the directory if it exists, then create it
if [ -d "$work_dir" ]; then
    rm -rf "$work_dir"
fi
mkdir -p "$work_dir"

# Create subdirectories
mkdir -p "$work_dir/images"
mkdir -p "$work_dir/label"
mkdir -p "$work_dir/depth"
mkdir -p "$work_dir/ply"
mkdir -p "$work_dir/visualize"

# Copy input_img_path to workdir/images
cp "$input_img_path" "$work_dir/images/"
img_path="$work_dir/images/$(basename $input_img_path)"

python third_party/AerialFormer/tools/test.py config/aerialformer.py weights/loveda_iter_135000.pth --out work_dir/label/labels.pkl
rm -r work_dirs

cd third_party/Depth-Anything/ 
python myrun.py --img-path ../../work_dir/images/ --outdir ../../work_dir/depth --encoder vitl --weight-dir ../../weights
cd ../..

python build_model.py
chmod -R 777 $work_dir
echo "done"