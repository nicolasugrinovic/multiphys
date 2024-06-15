import numpy as np
from PIL import Image
import cv2
import trimesh
import pickle
import torch
import json
import os
# import open3d as o3d
import scipy.io
import scipy.io as scio
import matplotlib.pyplot as plt
import io
from pathlib import Path
import yaml

# pcd = o3d.geometry.PointCloud()
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def xml_str_to_file(xml_str, filename):
    out_path = Path(filename).parent
    out_path.mkdir(exist_ok=True, parents=True)
    with open(filename, "w") as f:
        f.write(xml_str)

def read_yaml(path):
    with open(path, 'rb') as f:
        conf = yaml.safe_load(f.read())
    return conf

def cuda2numpy(tensor):
    return tensor.clone().detach().cpu().numpy()


def plot_from_j3d(gt_p3d_ours, scale=1, offset=0):
    black = np.ones([300, 300, 3], dtype=np.uint8)
    one_gt = gt_p3d_ours[..., :2]
    one_gt_ = one_gt - one_gt.min()
    one_gt_n = 0.5 * one_gt_ / one_gt_.max() + 0.25
    one_gt_n = scale * 300 * one_gt_n + offset
    plot_joints_cv2(black, one_gt_n, with_text=True)


def getNumbers(list):
    num_list = []
    for this in list:
        nums = ''
        for char in this:
            if char.isdigit():
                nums += char
        # print(nums)
        num_list.append(int(nums))
    num_list = np.array(num_list)
    return num_list

def getNumberFromString(this):
    nums = ''
    for char in this:
        if char.isdigit():
            nums += char
    return int(nums)

def get_heigths(j3d_height):
    jts0 = j3d_height[:, :-1, :]
    jts1 = j3d_height[:, 1:, :]
    p_diff = jts0 - jts1
    p_sqr = p_diff * p_diff
    p_sqr_sum = p_sqr.sum(2)
    norm = np.sqrt(p_sqr_sum)
    heights = norm.sum(1)
    return heights


def get_distance(points):
    p1, p2 = points
    p_diff = p1 - p2
    p_sqr = p_diff * p_diff
    p_sqr_sum = p_sqr.sum()
    norm = np.sqrt(p_sqr_sum)
    return norm


def draw_lsp_14kp__bone(img_pil, pts):
    bones = [
        [0, 1, 255, 0, 0],
        [1, 2, 255, 0, 0],
        [2, 12, 255, 0, 0],
        [3, 12, 0, 0, 255],
        [3, 4, 0, 0, 255],
        [4, 5, 0, 0, 255],
        [12, 9, 0, 0, 255],
        [9, 10, 0, 0, 255],
        [10, 11, 0, 0, 255],
        [12, 8, 255, 0, 0],
        [8, 7, 255, 0, 0],
        [7, 6, 255, 0, 0],
        [12, 13, 0, 255, 0]
    ]
    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    for pt in pts:
        if pt[2] > 0.2:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    for line in bones:
        pa = pts[line[0]]
        pb = pts[line[1]]
        xa, ya, xb, yb = int(pa[0]), int(pa[1]), int(pb[0]), int(pb[1])
        if pa[2] > 0.2 and pb[2] > 0.2:
            cv2.line(img, (xa, ya), (xb, yb), (line[2], line[3], line[4]), 2)

    # plot(img)
    return img



def vectorize_distance(a, b):
    """
    Calculate euclidean distance on each row of a and b
    :param a: Nx... np.array
    :param b: Mx... np.array
    :return: MxN np.array representing correspond distance
    """
    N = a.shape[0]
    a = a.reshape(N, -1)
    M = b.shape[0]
    b = b.reshape(M, -1)
    a2 = np.tile(np.sum(a ** 2, axis=1).reshape(-1, 1), (1, M))
    b2 = np.tile(np.sum(b ** 2, axis=1), (N, 1))
    dist = a2 + b2 - 2 * (a @ b.T)
    dist[np.where(dist<-0.0)] = 0
    return np.sqrt(dist)

# def save_points(rxz, name='points.ply'):
#     if isinstance(name, str):
#         name = Path(name)
#     name.parent.mkdir(parents=True, exist_ok=True)
#     pc = trimesh.PointCloud(rxz)
#     pcd.points = o3d.utility.Vector3dVector(pc.vertices)
#     o3d.io.write_point_cloud(str(name), pcd)

def check_pathlib(path):
    if isinstance(path, str):
        path = Path(path)
    return path

def create_dir(out_path):
    out_path = check_pathlib(out_path)
    out_path.mkdir(exist_ok=True, parents=True)

def mask_joints_w_vis(j2d):
    vis = j2d[0, :, 2].astype(bool)
    j2d_masked = j2d[:, vis]
    return j2d_masked

def joints_delete_zeros(j2d):
    vis = (j2d.sum(2) > 0).astype(bool)[0]
    j2d_masked = j2d[:, vis]
    return j2d_masked

def joints_delete_zeros_v1(j2d):
    vis = (j2d.sum(2) > 0).astype(bool)
    vis = np.all(vis, axis=0)
    j2d_masked = j2d[:, vis]
    return j2d_masked

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(file, data):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def read_pickle(f):
    with open(f, 'rb') as data:
        x = pickle.load(data)
    return x

def read_torch(f):
    return torch.load(f)


def read_mat(f):
    mat = scipy.io.loadmat(f)
    return mat

def read_txt(f):
    with open(f, 'r') as f:
        instances = f.read().split('\n')
        instances = instances[:-1]
    return instances

def read_str_txt(path):
    with open(path, 'r') as file:
        data = file.read()
    return data

def write_txt(dst, img_names_rand):
    check_pathlib(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(dst, img_names_rand, fmt="%s", delimiter=',')

def write_str_txt(print_str, f='filename.txt'):
    check_pathlib(f).parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w") as text_file:
        text_file.write(print_str)

def read_npy(f):
    return np.load(f)

def write_npy(f, data):
    check_pathlib(f).parent.mkdir(parents=True, exist_ok=True)
    np.save(f, data)

def write_mat(f, data):
    check_pathlib(f).parent.mkdir(parents=True, exist_ok=True)
    scio.savemat(f, data)


def read_pickle_compatible(f):
    with open(f, 'rb') as data:
        u = pickle._Unpickler(data)
        u.encoding = 'latin1'
        p = u.load()
    return p

def write_pickle(data, f):
    out_path = check_pathlib(f)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(f, 'wb') as f:
        pickle.dump(data, f)


def read_image(im_path):
    import matplotlib.pyplot as plt
    img = plt.imread(im_path)
    return img

def save_image(img, img_path):
    import matplotlib.pyplot as plt
    plt.imsave(img_path, img)
    # cv2.imwrite(img_path, img)


def read_image_PIL(im_path):
    from PIL import Image
    img = np.asarray(Image.open(im_path))
    # img = np.array(Image.open(im_path))
    return img

def save_img(im_path, img_pil_or_np):
    """
    save image, it can be pil or numpy
    """
    from PIL import Image
    is_pil = Image.isImageType(img_pil_or_np)

    if is_pil:
        img_pil_or_np.save(im_path)
    else:
        plt.imsave(im_path, img_pil_or_np)



def plot(img):
    import matplotlib.pyplot as plt
    # from matplotlib import transforms
    # tr = transforms.Affine2D().rotate_deg(rotation_in_degrees)
    plt.figure()
    # plt.imshow(img, transform=tr)
    plt.imshow(img)
    plt.show()

def plot_w_tittle(img, title, show=False):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.axis('off')
    buf = io.BytesIO()
    fig.canvas.draw()
    ax.imshow(img)
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    img = np.asarray(image)
    if show:
        plt.show()
    plt.close('all')
    return img

def string2number(string):
    '''
    solo funciona para un solo numero junto, especialmente para numeros de los imgs paths
    '''
    return int(''.join([s for s in string if s.isdigit()]))


def delete_nans(x):
    good_idxs = ~np.isnan(x)
    x = x[good_idxs]
    return x

def filter_nans(x):
    filter(lambda v: v == v, x)
    return x


def plot_to_img(plt, fig):
    # Convert plot to image.
    dpi_fig = fig.dpi
    buf = io.BytesIO()
    dpi = dpi_fig
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    # plt.show()
    plt.close(fig)
    return img


def plot_boxes_cv2(img_pil, boxes, do_return=False):
    """
    this is for boxes format xyxy
    img can be pil or ndarray
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)


    for (xmin, ymin, xmax, ymax), c in zip(boxes.tolist(), COLORS * 100):
        color = (np.array(c) * 255).astype(int).tolist()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)


    if do_return:
        return img
    else:
        plot(img)



def plot_boxes_w_text_cv2(img_pil, boxes, number_list, do_plot=True, fontScale=2):
    """
    this is for boxes format xyxy
    img can be pil or ndarray
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    for (xmin, ymin, xmax, ymax), c, number in zip(boxes.tolist(), COLORS * 100, number_list):
        color = (np.array(c) * 255).astype(int).tolist()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        ymid = int((ymin + ymax) / 2)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        text = '%.2f' % number
        img = cv2.putText(img, text, (xmin, ymin+50), font, fontScale=fontScale, color=(255, 0, 0), thickness=2)

    if do_plot:
        plot(img)
    else:
        return img

def plot_boxes_w_persID_cv2(img_pil, boxes, number_list=None, do_return=False, fontScale=4):
    """
    this is for boxes format xyxy
    img can be pil or ndarray
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    if number_list is None:
        number_list = list(range(1, len(boxes)+1))

    for (xmin, ymin, xmax, ymax), c, number in zip(boxes.tolist(), COLORS * 100, number_list):
        color = (np.array(c) * 255).astype(int).tolist()
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        ymid = int((ymin + ymax) / 2)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)

        text = '%d' % number
        img = cv2.putText(img, text, (xmin, ymid), font, fontScale=fontScale, color=(255, 0, 0), thickness=6)

    if do_return:
        return img
    else:
        plot(img)

def add_txt_to_img(img_pil, text, x=0,y=100, fontSize=2, thickness=4, color=(255, 0, 0), alpha=1.0, bg_color=None, offset=-10):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    original = img_pil.copy()
    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        if img_pil.max() < 2:
            img_pil = (255 * img_pil).astype(np.uint8)
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    # img = cv2.rectangle(img, (x, y+20), (x + 2000, y+10-100), (0, 255, 0), -1)
    if bg_color is not None:
        text_size, _ = cv2.getTextSize(text, font, fontSize, thickness)
        text_w, text_h = text_size
        # text_color_bg = (0, 0, 0)
        text_h = int(0.9*text_h)
        # offset = -10
        img = cv2.rectangle(img, (x, y-text_h + offset), (x + text_w, y + text_h + offset), bg_color, -1)
        # plot(img)
        # plot(result)

    img = cv2.putText(img, text, (x, y), font, fontScale=fontSize,
                      color=color, thickness=thickness)

    result = cv2.addWeighted(img, alpha, original, 1 - alpha, 0)
    return result


def add_txt_to_img_w_pad(img_pil, text, x=0,y=100, fontSize=2, thickness=4):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX

    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    # x = 0
    # y = 100
    h, w, c = img.shape
    new_h = int(h * 1.1)
    zero_img = np.zeros([new_h, w, c], dtype=np.uint8)
    offs = new_h - h
    zero_img[offs:, ...] = img
    # plot(zero_img)
    img = zero_img
    img = cv2.rectangle(img, (x, y+20), (x + 2000, y+10-100), (0, 255, 0), -1)
    img = cv2.putText(img, text, (x, y), font, fontScale=fontSize,
                      color=(255, 0, 0), thickness=thickness)
    # plot(img)


    return img


def plot_joints_cv2(img_pil, gtkps, show=True, with_text=False, sc=None):
    '''
    :param img_pil:
    :param gtkps: should be of dims n x K x 3 or 2 -> [n,K,3], joint locations should be integer
    :param do_plot:
    :return:
    '''
    font = cv2.FONT_HERSHEY_PLAIN
    is_pil = Image.isImageType(img_pil)
    if is_pil:
        img = np.asarray(img_pil)
    else:
        img_ = Image.fromarray(img_pil)
        img = np.asarray(img_)

    h, w, _ = img.shape
    max_s = max(h, w)

    if sc is None:
        if max_s > 500:
            sc = int(max_s / 500)
        else:
            sc = 1
    # sc = 2
    # convert to int for cv2 compat
    if isinstance(gtkps, np.ndarray):
        gtkps = gtkps.astype(np.int)
    elif isinstance(gtkps, torch.Tensor):
        gtkps = gtkps.int().numpy()
    else:
        print('Unknown type!!')

    for kpts in gtkps:
        for i, (x, y) in enumerate(kpts[..., :2]):
            img = cv2.circle(img, (x, y), radius=2*sc, color=(255, 255, 0), thickness=2*sc)
            if with_text:
                text = '%d' % i
                img = cv2.putText(img, text, (x, y), font, fontScale=1.0*sc, color=(255, 0, 0), thickness=1*sc)

    if show:
        plot(img)
    return img

def plot_skel_cv2(img, j2d_multi_person, type='OP', alpha = .6, all_yellow=False):
    """
    j2d: (N, 3)
    """

    ################################ This is for OpenPose 25 joints ################################
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
              [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0]]
    # for OP 25 joints
    limbSeq = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [11, 24],
               [11, 22], [22, 23], [8, 12], [12, 13], [13, 14], [14, 21], [14, 19], [19, 20], [0, 15], [15, 17],
               [0, 16], [16, 18]]

    colors_emb = [
        [255, 255, 0], [255, 255, 0], [255, 255, 0],[255, 255, 0],[255, 255, 0],[255, 255, 0],[255, 255, 0],
        [255, 255, 0],[255, 255, 0],[255, 255, 0],[255, 255, 0],[255, 255, 0],[255, 255, 0],[255, 255, 0],
        [255, 255, 0],[255, 255, 0],[255, 255, 0], [255, 255, 0],[255, 255, 0],
                  ]
    # for embPose joints 12 format (look for openpose_subindex)
    limbSeq_emb = [[0, 1], [0, 4],[1, 2],[2, 3],[4, 5],[5, 6],[0, 7],[7, 8],[8, 9],[7, 10],[10, 11]]

    if type == 'OP':
        pass
    elif type == 'emb':
        limbSeq = limbSeq_emb
        colors = colors_emb

    if all_yellow:
        colors = colors_emb

    ###############################################################################################
    original = img.copy()
    H, W, _ = img.shape

    dim_n = len(j2d_multi_person.shape)
    # assert dim_n == 3, 'j2d_multi_person should be of dims N x K x 3'
    if dim_n == 2:
        j2d_multi_person = j2d_multi_person[None]

    if isinstance(j2d_multi_person, torch.Tensor):
        j2d_multi_person = j2d_multi_person.numpy()

    for j2d in j2d_multi_person:
        x_s = j2d[:, 0]
        y_s = j2d[:, 1]
        try:
            conf_s = j2d[:, 2]
        except:
            conf_s = np.ones_like(x_s)

        for i, limb in enumerate(limbSeq):
            l1 = limb[0];
            l2 = limb[1];
            conf = conf_s[1];
            if conf > 0.4 and not ((x_s[l1] == W and y_s[l1] == H) or (x_s[l2] == W and y_s[l2] == H)
                                   or (x_s[l1] == 0 and y_s[l1] == 0) or (x_s[l2] == 0 and y_s[l2] == 0)):
                sc = 2
                p1 = (int(x_s[l1]), int(y_s[l1]))
                p2 = (int(x_s[l2]), int(y_s[l2]))
                img = cv2.line(img, p1, p2, color=[j for j in colors[i % 19]], thickness=2 * sc)
                # plot(img)

        # plot kp
        for i in range(len(x_s)):
            conf = conf_s[1];
            if conf > 0.4 and not ((x_s[i] == W and y_s[i] == H) or (x_s[i] == W and y_s[i] == H)
                                   or (x_s[i] == 0 and y_s[i] == 0)):
                # ax.plot(x_s[i], y_s[i], 'o', markersize=10, color=[j / 255 for j in colors[i % 19]], alpha=alpha)
                x = int(x_s[i])
                y = int(y_s[i])
                img = cv2.circle(img, (x, y), radius=2 * sc, color=[j for j in colors[i % 19]], thickness=2 * sc)

        # Perform weighted addition of the input image and the overlay
        result = cv2.addWeighted(img, alpha, original, 1 - alpha, 0)
    # plot(result)
    return result


def save_trimesh(vertices, faces, out_path, color=None):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.clone().detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.clone().detach().cpu().numpy()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if color is not None:
        if len(color)==3:
            color = color[0]
        mesh.visual.vertex_colors = color
    mesh.export(out_path);


def save_mesh(vertices, faces=None, out_path='scene.obj'):
    dim_n = 1
    is_list = isinstance(vertices, list)
    is_array = isinstance(vertices, np.ndarray)
    is_tensor = isinstance(vertices, torch.Tensor)
    if not is_list:
        dim_n = len(vertices.shape)
    assert is_list or ((is_array or is_tensor) and dim_n==3), 'vertices should be a list or an array or tensor of Bx N x 3'

    if faces is None:
        # smpl_model_path = './models/smpl_faces.npy'
        smpl_model_path = 'data/smplx_faces.pkl'
        faces = np.load(smpl_model_path)
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.clone().detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.clone().detach().cpu().numpy()
    tri_verts = []
    for v in vertices:
        triv = trimesh.Trimesh(v, faces)
        tri_verts.append(triv)
    scene = tri_verts[0].scene()
    for v in tri_verts[1:]:
        scene.add_geometry(v)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))

def save_scene(meshes, out_path='scene.obj'):
    if not isinstance(meshes, list):
        raise ValueError('meshes should be a list of trimesh.Trimesh')
    scene = meshes[0].scene()
    for v in meshes[1:]:
        scene.add_geometry(v)
    scene.export(str(out_path))

    
def save_pointcloud(j3d, path):
    if isinstance(j3d, torch.Tensor):
        j3d = j3d.clone().detach().cpu().numpy()
    if isinstance(path, str):
        path = Path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pc = trimesh.PointCloud(j3d)
    pc.export(path)


def read_mesh(mesh):
    return trimesh.load_mesh(mesh)

def save_meshes_seq(verts_batch_ds, faces, out_path, name, ply=False):
    out_path = check_pathlib(out_path) / Path(name)
    out_path.mkdir(exist_ok=True, parents=True)
    for n, verts in enumerate(verts_batch_ds):
        if ply:
            out_path_ = out_path / Path(f'{name}_{n:03d}.ply')
        else:
            out_path_ = out_path / Path(f'{name}_{n:03d}.obj')
        save_trimesh(verts, faces, out_path_)
    print('Meshes saved at')
    print(out_path)


def check_nan(verts_hat):
    if isinstance(verts_hat, torch.Tensor):
        verts_hat_np = verts_hat.detach().cpu().numpy()
    is_nan = np.isnan(verts_hat_np)
    while len(is_nan.shape) > 1:
        is_nan = is_nan.sum(-1)
    is_nan_s = (is_nan > 0).astype(np.int)
    print(f"is_nan_s_={is_nan_s}")


def get_minmax(verts_hat):
    verts_hat_np = verts_hat.detach().cpu().numpy()
    vmin_hat = verts_hat_np.min()
    vmax_hat = verts_hat_np.max()
    print(f"vmin_hat={vmin_hat}, vmax_hat={vmax_hat}")



def get_smplx_faces():
    faces_path = '/home/nugrinov/code/HumanPoseGeneration/data/smplx_faces.pkl'
    faces = read_pickle(faces_path)
    return faces