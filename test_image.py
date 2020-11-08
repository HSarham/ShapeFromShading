import model
import tensorflow as tf
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_scaled_image(in_image):
    width = in_image.width
    height = in_image.height

    scaled_width = 128
    scaled_height = 128

    if width > height:
        scaled_height = int(128 / width * height)
    else:
        scaled_width = int(128 / height * width)

    return in_image.resize((scaled_width, scaled_height))


def resize_depth(in_depth, width, height):
    pil_image = Image.fromarray(np.uint16(in_depth * 1000))
    pil_image = pil_image.resize((width, height), Image.NEAREST)
    output = np.array(pil_image, dtype=np.float32) / 1000
    return output


def create_mesh(in_depth):
    rows, cols = in_depth.shape

    X = np.empty([rows, cols, 1])
    Y = np.empty([rows, cols, 1])
    Z = np.empty([rows, cols, 1])

    x, y = np.meshgrid(np.arange(0, 1, 1 / cols), np.arange(0, 1, 1 / rows))

    X[:, :, 0] = x
    Y[:, :, 0] = y
    Z[:, :, 0] = in_depth

    vertices = np.concatenate((X, Y, Z), axis=2)
    # print(vertices[0,:2,:])
    vertices = vertices.reshape((rows * cols, 3))
    # print(vertices[0:2,:])
    faces = np.empty([0, 3])
    for i in range(rows - 1):
        for j in range(cols - 1):
            vertex_index = i * 128 + j
            if (abs(in_depth[i, j] - in_depth[i + 1, j + 1]) < 0.05):
                if (abs(in_depth[i, j] - in_depth[i + 1, j]) < 0.05):
                    faces = np.append(faces, [[vertex_index, vertex_index + cols, vertex_index + cols + 1]], axis=0)
                if (abs(in_depth[i, j] - in_depth[i, j + 1]) < 0.05):
                    faces = np.append(faces, [[vertex_index, vertex_index + cols + 1, vertex_index + 1]], axis=0)

    return vertices, faces


def write_mesh_output(weights_path, image_path, out_mesh_path):
    Ws = {}
    w_file = h5py.File(weights_path, 'r')
    for name in model.weight_names:
        Ws[name] = tf.constant(w_file[name])

    image = Image.open(image_path)
    input_image = np.empty([1, 128, 128, 3])
    input_image[0] = np.array(image.resize((128, 128)), dtype=np.float32) / 255

    X = tf.constant(input_image, dtype='float32')
    ff = model.feed_forward(X, Ws)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    depth = np.array(ff.eval(session=sess))
    depth_image = depth[0, :, :, 0]

    scaled_image = get_scaled_image(image)
    scaled_width = scaled_image.width
    scaled_height = scaled_image.height
    resized_depth = resize_depth(depth_image, scaled_width, scaled_height)

    plt.imshow(resized_depth, cmap=plt.get_cmap('gray'))
    plt.show()

    vertices, faces = create_mesh(resized_depth)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.reshape(np.array(scaled_image, dtype=np.float32) / 255, (scaled_width * scaled_height, 3)))

    o3d.io.write_triangle_mesh(out_mesh_path, mesh)


image_p = sys.argv[1]
weights_p = sys.argv[2]
out_mesh_p = "out_mesh.ply"
write_mesh_output(weights_p, image_p, out_mesh_p)
