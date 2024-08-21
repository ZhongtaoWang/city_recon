import shutil
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor ,SamAutomaticMaskGenerator
import os
import pickle
from PIL import Image

import matplotlib.pyplot as plt
from skimage import morphology, io, color

import numpy as np
#how to use pip to install sklearn 
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
from scipy.ndimage import mean ,maximum
import scipy


#hyperparameters
building_scale = 0.12 # scale factor for building height


def save_ply_color(filename, vertices, colors, faces):
    with open(filename, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write(f"element face {len(faces)}\n")
        file.write("property list uchar int vertex_index\n")
        file.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            file.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")
        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def create_buildings_from_labels_and_image(label_array, image_array, ply_path):
    vertices = []
    colors = []
    faces = []
    current_vertex_index = 0

    # Add colored ground plane
    for i in range(label_array.shape[0]):
        for j in range(label_array.shape[1]):
            # Add a small square at each (j, i) with the color from the image
            base_idx = current_vertex_index
            corners = [
                (j, i, 0),
                (j+1, i, 0),
                (j+1, i+1, 0),
                (j, i+1, 0)
            ]
            vertices.extend(corners)
            color = tuple(image_array[i, j].tolist())  # Get RGB values from image array
            ground_colors = [color] * 4  # Same color for all vertices of this ground square
            colors.extend(ground_colors)
            ground_face = [base_idx, base_idx + 1, base_idx + 2, base_idx + 3]
            
            faces.append([ground_face[0], ground_face[1], ground_face[2]])
            faces.append([ground_face[2], ground_face[3], ground_face[0]])
            current_vertex_index += 4

    # Add buildings
    for i in range(label_array.shape[0]):
        for j in range(label_array.shape[1]):
            if label_array[i, j] != 0:  # Assuming non-zero values are building areas
                base_idx = current_vertex_index
                height = - label_array[i, j] * building_scale * np.max(label_array.shape) 
                corners = [
                    (j, i, 0),
                    (j+1, i, 0),
                    (j+1, i+1, 0),
                    (j, i+1, 0),
                    (j, i, height),
                    (j+1, i, height),
                    (j+1, i+1, height),
                    (j, i+1, height)
                ]
                vertices.extend(corners)
                color = tuple(image_array[i, j].tolist())  # Get RGB values from image array
                building_colors = [color] * 8  # Same color for all vertices of this building
                colors.extend(building_colors)
                
                cube_faces = [
                    [base_idx + 4, base_idx + 5, base_idx + 6, base_idx + 7],  # top
                    [base_idx, base_idx + 1, base_idx + 5, base_idx + 4],  # sides
                    [base_idx + 1, base_idx + 2, base_idx + 6, base_idx + 5],
                    [base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6],
                    [base_idx + 3, base_idx + 0, base_idx + 4, base_idx + 7]
                ]
                for face in cube_faces:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[2], face[3], face[0]])

                current_vertex_index += 8  # 8 vertices were added per box

    save_ply_color(ply_path, vertices, colors, faces)

# Example: create_buildings_from_labels_and_image(label, image, 'output.ply')


def replace_with_mean(data):
    """
    Replaces each non-zero connected component in a 2D numpy array with the mean value
    of that connected component.

    Parameters:
    - data: A 2D numpy array.

    Returns:
    - A 2D numpy array where each non-zero connected component has been replaced by its mean value.
    """
    # 创建一个与原数组形状相同的数组，用于存储最终结果
    result = np.zeros_like(data)
    
    # 用label函数标记连通区域，结构元素定义为对角连接也算连通
    structure = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]])
    labeled_array, num_features = scipy.ndimage.label(data != 0, structure=structure)
    
    # 对每一个连通区域应用均值计算
    for i in range(1, num_features + 1):
        component_slice = (labeled_array == i)
        # 计算当前连通块的最大值
        component_mean = mean(data, labels=labeled_array, index=i)
        # 更新结果数组
        result[component_slice] = component_mean
    
    return result

def fit_plane_to_ground_ransac(depth, label):
    """
    Uses RANSAC to robustly fit a plane to the ground pixels in a depth map and normalizes
    the depth of all pixels relative to this plane.

    Parameters:
    - depth: numpy array of shape (H, W), depth values for each pixel.
    - label: numpy array of shape (H, W), binary mask with 0s for ground and 1s for buildings.

    Returns:
    - normalized_depth: numpy array of shape (H, W), depth adjusted relative to the RANSAC-fitted ground plane.
    """
    # Extract indices where ground is present
    ground_indices = np.where(label == 0)
    
    # Prepare input features [x, y] for each ground pixel
    x_coords = ground_indices[1]
    y_coords = ground_indices[0]
    X = np.vstack([x_coords, y_coords]).T
    
    # Vector B with depth values of ground
    B = depth[ground_indices]
    
    # Define the RANSAC regressor
    ransac = RANSACRegressor(#base_estimator=LinearRegression(),
                             min_samples=0.9,  # at least 10% of data
                             residual_threshold=None,  # distance threshold to consider as inlier
                             max_trials=100)  # number of iterations
    
    # Fit RANSAC model
    ransac.fit(X, B)
    
    # Use the inlier mask to determine the plane from inliers
    inlier_mask = ransac.inlier_mask_
    ransac.fit(X[inlier_mask], B[inlier_mask])
    
    # Use the model coefficients to calculate the fitted plane across the whole depth map
    X_full, Y_full = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    predicted_plane = ransac.predict(np.column_stack((X_full.ravel(), Y_full.ravel()))).reshape(depth.shape)
    
    # Normalize the depth map
    normalized_depth = depth - predicted_plane
    
    return ( normalized_depth - np.min(normalized_depth) ) / (np.max(normalized_depth) - np.min(normalized_depth))

# Example usage:
# Assuming depth and label are your depth and mask arrays
# depth_normalized = fit_plane_to_ground_ransac(depth, label)



def save_ply(filename, vertices, faces):
    with open(filename, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(vertices)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write(f"element face {len(faces)}\n")
        file.write("property list uchar int vertex_index\n")
        file.write("end_header\n")
        for vertex in vertices:
            file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def create_buildings_from_labels(label_array, ply_path):
    vertices = []
    faces = []
    current_vertex_index = 0

    # Add ground plane
    ground_corners = [
        (0, 0, 0),
        (label_array.shape[1], 0, 0),
        (label_array.shape[1], label_array.shape[0], 0),
        (0, label_array.shape[0], 0)
    ]
    vertices.extend(ground_corners)
    ground_faces = [
        [current_vertex_index, current_vertex_index + 1, current_vertex_index + 2, current_vertex_index + 3]
    ]
    for face in ground_faces:
        faces.append([face[0], face[1], face[2]])
        faces.append([face[2], face[3], face[0]])
    current_vertex_index += 4

    # Add buildings
    for i in range(label_array.shape[0]):
        for j in range(label_array.shape[1]):
            if label_array[i, j] != 0:  # Assuming non-zero values are building areas
                base_idx = current_vertex_index
                height = - label_array[i, j] * building_scale * np.max(label_array.shape) 
                corners = [
                    (j, i, 0),
                    (j+1, i, 0),
                    (j+1, i+1, 0),
                    (j, i+1, 0),
                    (j, i, height),
                    (j+1, i, height),
                    (j+1, i+1, height),
                    (j, i+1, height)
                ]
                vertices.extend(corners)
                
                cube_faces = [
                    [base_idx + 4, base_idx + 5, base_idx + 6, base_idx + 7],  # top
                    [base_idx, base_idx + 1, base_idx + 5, base_idx + 4],  # sides
                    [base_idx + 1, base_idx + 2, base_idx + 6, base_idx + 5],
                    [base_idx + 2, base_idx + 3, base_idx + 7, base_idx + 6],
                    [base_idx + 3, base_idx + 0, base_idx + 4, base_idx + 7]
                ]
                for face in cube_faces:
                    faces.append([face[0], face[1], face[2]])
                    faces.append([face[2], face[3], face[0]])

                current_vertex_index += 8  # 8 vertices were added per box

    save_ply(ply_path, vertices, faces)



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=330):
    pos_points = coords[labels==1]  
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            #show_points(input_point, input_label, plt.gca())
            pass
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        #plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.show()
        plt.close()


def show_logit(logits, scores, input_point, input_label, input_box, filename, image):
    for i, (logit, score) in enumerate(zip(logits, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(logit)
        plt.axis('off')
        #plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.show()
        plt.close()


def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    #plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    #show image
    plt.show()
    plt.close()


def image_to_one_hot(image_npy,N):
    image_logit = np.zeros((N,image_npy.shape[0], image_npy.shape[1]))
    for i in range(N):
        image_logit[i,:,:] = (image_npy == i)
    return image_logit


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    mask_return = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']

        #if ann['area'] < 300: continue
        #print(m.shape)
        #print(sampled_points)
        #print(sampled_points[:,::-1])
        #if np.sum(m[sampled_points[:,1],sampled_points[:,0]]) == 0:
        #if np.sum(m[sampled_points[:,1],sampled_points[:,0]]) <= 3 * np.sum(m[background_points[:,1],background_points[:,0]]):
        if np.sum(m * label ) <  np.sum(m * background_label * 2) :
            continue
        #print('selected')
        color_mask = np.concatenate([np.random.random(3), [0.7]])
        #color_mask = np.concatenate([np.array([1,0,0]), [0.35]])
        img[m] = color_mask
        mask_return[m] = 1
    #ax.imshow(img)
    return mask_return

#mkdir 'work_dir'
work_dir = 'work_dir'
input_img_path = os.listdir('work_dir/images')[0]
img_path = work_dir + '/images/' + os.path.basename(input_img_path)

sam_checkpoint = "weights/sam_hq_vit_h.pth"
#sam_checkpoint = "/data4/wangzhongtao/seg/segment-anything/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
image_name = img_path
image = cv2.imread(image_name)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

image_label_path = 'work_dir/label/labels.pkl'
with open(image_label_path, 'rb') as f:
    labels = pickle.load(f)
    f.close()
image_label = np.array(labels[0])


image_logit = image_to_one_hot(image_label,np.max(image_label)+1)

tau = 0.5
hq_token_only = False 
num_points = 10000


logit = image_logit[1]
label = logit
#normalize labelimage_logit[5].copy()
label = (label - np.min(label))/(np.max(label) - np.min(label))
label[label<tau] = 0
label[label>=tau] = 1

label_save = Image.fromarray((label*255).astype(np.uint8))
label_save.save('work_dir/visualize/building_label.png')

background_label = logit
#normalize label
background_label = 1 - (background_label - np.min(background_label))/(np.max(background_label) - np.min(background_label))
background_label[background_label<1-tau] = 0


mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
mask = show_anns(masks)

#save mask with PIL
mask_save = Image.fromarray((mask*255).astype(np.uint8))
mask_save.save('work_dir/visualize/building_mask.png')


selem = morphology.rectangle(10, 10)
opened_mask = mask
for i in range(3):
    opened_mask = morphology.binary_opening(opened_mask, footprint=selem)

#save opened mask with PIL
opened_mask_save = Image.fromarray((opened_mask*255).astype(np.uint8))
opened_mask_save.save('work_dir/visualize/building_refined_mask.png')

image_depth_path = work_dir + '/depth/' + os.path.basename(input_img_path)
image_label = opened_mask
image_depth = np.array(Image.open(image_depth_path))

depth_normalized = fit_plane_to_ground_ransac(image_depth, image_label)

#save depth_normalized * image_label with PIL
depth_normalized_save = Image.fromarray((depth_normalized*255).astype(np.uint8))
depth_normalized_save.save('work_dir/visualize/depth_normalized.png')

final_depth = replace_with_mean(depth_normalized * image_label)

#save final_depth with PIL
final_depth_save = Image.fromarray((final_depth*255).astype(np.uint8))
final_depth_save.save('work_dir/visualize/final_depth.png')

create_buildings_from_labels(final_depth, f'work_dir/ply/{os.path.basename(input_img_path).split(".")[0]}.ply')
create_buildings_from_labels_and_image(final_depth, image, f'work_dir/ply/{os.path.basename(input_img_path).split(".")[0]}_color.ply')
