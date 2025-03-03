#!/usr/bin/env python
# encoding: utf-8
# Software ExPI_toolbox
# Copyright Inria
# Year 2021
# Contact : wen.guo@inria.fr

import csv
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import xml.dom.minidom as xmldom
import math

order_orig = ['m-fhead','m-lhead','m-rhead', 'm-back','m-lshoulder','m-rshoulder',\
        'm-lelbow','m-relbow','m-lwrist','m-rwrist','m-lhip','m-rhip','m-lknee','m-rknee','m-lheel', 'm-rheel','m-ltoes','m-rtoes',\
        'f-fhead','f-lhead','f-rhead', 'f-back','f-lshoulder','f-rshoulder',\
        'f-lelbow','f-relbow','f-lwrist','f-rwrist','f-lhip','f-rhip','f-lknee','f-rknee','f-lheel', 'f-rheel','f-ltoes','f-rtoes']
p1_order0 = [0,0,0, 3,4,6, 3,5,7,   3,10,12,14,   3,11,13,15]
p1_order1 = [1,2,3, 4,6,8, 5,7,9, 10,12,14,16, 11,13,15,17]
p2_order0 = (np.array(p1_order0)+18).tolist()
p2_order1 = (np.array(p1_order1)+18).tolist()
nb_kpts = 36
def vis_acro_2d_nan(cvimg, pose_list, save_path=None):
    ## pose_list: 36*2
    ## 2 person: m, f (36*2)
    ## 18*2 kpts, 17*2 limbs
    nb_kpts = 36
    p = np.array(pose_list, dtype=np.float32).reshape((nb_kpts, 2))
    p_y, p_x = p[:, 1], p[:, 0]

    for i in range(nb_kpts):
        if not math.isnan(p_x[i]):
            cv2.circle(cvimg, (p_x[i],p_y[i]), radius=3, color=[0,0,0], thickness=-1, lineType=cv2.LINE_AA)
            #nb_kpt_vis += 1
    for i in range(len(p1_order0)):
        if not math.isnan(p_x[p1_order0[i]]) and not math.isnan(p_x[p1_order1[i]]):
            cv2.line(cvimg, (p_x[p1_order0[i]], p_y[p1_order0[i]]), (p_x[p1_order1[i]], p_y[p1_order1[i]]),\
                color=[0,0,255], thickness=2, lineType=cv2.LINE_AA)
        if not math.isnan(p_x[p2_order0[i]]) and not math.isnan(p_x[p2_order1[i]]):
            cv2.line(cvimg, (p_x[p2_order0[i]], p_y[p2_order0[i]]), (p_x[p2_order1[i]], p_y[p2_order1[i]]),\
                color=[255,0,0], thickness=2, lineType=cv2.LINE_AA)
    if save_path:
        cv2.imwrite(save_path,cvimg)
    else:
        return cvimg

def vis_acro_2d(cvimg, pose_list, save_path=None):
    ## pose_list: 36*2
    ## 2 person: m, f (36*2)
    ## 18*2 kpts, 17*2 limbs

    #nb_kpts = 36
    p = np.array(pose_list, dtype=np.float32).reshape((nb_kpts, 2))

    p_y, p_x = p[:, 1], p[:, 0]
    #print("x:",p_x,"y:",p_y)

    #nb_kpt_vis = 0
    for i in range(nb_kpts):
        if not math.isnan(p_x[i]):
            cv2.circle(cvimg, (p_x[i],p_y[i]), radius=3, color=[0,0,0], thickness=-1, lineType=cv2.LINE_AA)
            #nb_kpt_vis += 1
    for i in range(len(p1_order0)):
        if not math.isnan(p_x[p1_order0[i]]) and not math.isnan(p_x[p1_order1[i]]):
            cv2.line(cvimg, (p_x[p1_order0[i]], p_y[p1_order0[i]]), (p_x[p1_order1[i]], p_y[p1_order1[i]]),\
                color=[0,0,255], thickness=2, lineType=cv2.LINE_AA)
        if not math.isnan(p_x[p2_order0[i]]) and not math.isnan(p_x[p2_order1[i]]):
            cv2.line(cvimg, (p_x[p2_order0[i]], p_y[p2_order0[i]]), (p_x[p2_order1[i]], p_y[p2_order1[i]]),\
                color=[255,0,0], thickness=2, lineType=cv2.LINE_AA)
    if save_path:
        #if nb_kpt_vis < 36:
        cv2.imwrite(save_path,cvimg)
    else:
        return cvimg

def vis_acro_3d(pose_list, save_path):
    ## pose_list: 36*3
    ## 2 person: m, f (36*3)
    ## 18*2 kpts, 17*2 limbs

    #nb_kpts = 36
    p = np.array(pose_list, dtype=np.float32).reshape((nb_kpts, 3))

    #color =  ['darkgreen','seagreen','black', 'dimgray', 'dimgrey','skyblue','royalblue','navy','darkcyan',\
    #        'darkgreen','gray','darkgray','darkgrey','c','dodgerblue','navy']

    p_y, p_x, p_z = p[:, 1], p[:, 0], p[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(nb_kpts):
        ax.scatter(p_x[i],p_y[i],p_z[i],c='black')
    for i in range(len(p1_order0)):
        ax.plot([p_x[p1_order0[i]], p_x[p1_order1[i]]], [p_y[p1_order0[i]], p_y[p1_order1[i]]],[p_z[p1_order0[i]], p_z[p1_order1[i]]],\
                c= 'r')
        ax.plot([p_x[p2_order0[i]], p_x[p2_order1[i]]], [p_y[p2_order0[i]], p_y[p2_order1[i]]],[p_z[p2_order0[i]], p_z[p2_order1[i]]],\
                c= 'b')
    ax.set_xlabel('x')
    ax.set_ylabel('depth')
    ax.set_zlabel('y')
    ax.set_zlim(0,1800)#(0,1800)
    plt.xlim(-2000, 2000)
    plt.ylim(-2000, 2000)
    ax.view_init(30, -120)
    plt.savefig(save_path)

def read_calib(xml_path, camera_name):
    ## read orig camara calib file
    xml_file = xmldom.parse(xml_path)
    eles = xml_file.documentElement
    cam_calib = eles.getElementsByTagName("Camera")[camera_name - 1]

    P_s = cam_calib.getElementsByTagName("P")[0].firstChild.data # string
    P = [float(s) for s in P_s.split(' ')[:-1]] #dim 12
    P = np.array(P).reshape((3,4))#.tolist()

    K = np.array([float(s) for s in cam_calib.getElementsByTagName("K")[0].firstChild.data.split(' ')[:-1]]).reshape((3,3))
    R = np.array([float(s) for s in cam_calib.getElementsByTagName("R")[0].firstChild.data.split(' ')[:-1]]).reshape((3,3))
    T = np.array([float(s) for s in cam_calib.getElementsByTagName("T")[0].firstChild.data.split(' ')[:-1]]).reshape((3,1))

    Distor = cam_calib.getElementsByTagName("Distortion")[0]
    xc, yc, K1, K2 = float(Distor.getAttribute("Cx")), float(Distor.getAttribute("Cy")),\
                     float(Distor.getAttribute("K1")), float(Distor.getAttribute("K2"))

    out_dict = { 'P':P, 'K':K, 'R':R, 'T':T, 'xc':xc, 'yc':yc, 'K1':K1, 'K2':K2}

    return out_dict

def read_gt_clean(root_path):
    # read data with img name
    # return gt dict img_name:numpy array(nb_kpts*3) in order_orig
    ## read mesh.start
    align_file = open(root_path+'talign.csv')
    read_align = csv.reader(align_file, delimiter=",")
    t = 1
    for row in read_align:
        if t == 1:
            off = [r for r in row]
            if off[0] == 'mesh.start':
                offset = int(off[1])
            else:
                print('ERROR: error in getting mesh.start')
        t += 1
    ## read gt
    tsv_file = open(root_path+'mocap_cleaned.tsv')
    read_tsv = csv.reader(tsv_file, delimiter=",")
    gt = {}
    t = 1
    for row in read_tsv:
        #if t == 1:
        #    order = [str(o) for o in row][1:]
        if t >= 2:
            img_id = t-2+offset
            img_name = 'img-' + str(img_id).zfill(6) + '.jpg'
            gt[img_name] = np.array([float(g) for g in row]).reshape((nb_kpts, 3))
        t += 1
    tsv_file.close()
    return gt

def read_gt_old(root_path):
    # return gt dict img_name:numpy array(nb_kpts*3) in order_orig
    ##read offset
    align_file = open(root_path+'talign.csv')
    read_align = csv.reader(align_file, delimiter=",")
    t = 1
    for row in read_align:
        if t == 7:
            off = [r for r in row]
            if off[0] == 'offset':
                offset = int(off[1])
            else:
                print('ERROR: error in getting OFFSET')
        t += 1
    ## read mocap gt file
    gt = {}
    t = 1
    tsv_file = open(root_path+'mocap.tsv')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        if t == 10:
            order = [str(o) for o in row][1:]#[1:-1]#[1:]
        if (t - offset - 11)%2 == 0 and t >= 11:
            img_id = int((t - offset - 11)/2)
            img_name = 'img-' + str(img_id).zfill(6) + '.jpg'
            #print(img_name)
            gt[img_name] = [float(g) for g in row]#[0:-3]
            #from IPython import embed
            #embed()
            #exit()
        t += 1
    tsv_file.close()
    ## change order
    for img_name in gt:
        kpts = gt[img_name]
        kpts = np.array(kpts).reshape(-1,3)

        kpts_dict = {order[i]:kpts[i] for i in range(nb_kpts)}
        ## for the 2 seq with missed 1 joint
        #kpts_dict = {order[i]:kpts[i] for i in range(35)}
        #kpts_dict['m-rwrist'] = np.array([float('nan')]*3)  # acro1 around-the-back1
        #kpts_dict['f-rknee'] = np.array([float('nan')]*3)  # acro2b around-the-back7

        kpts_3d = [kpts_dict[o] for o in order_orig]
        kpts_3d = np.array(kpts_3d, dtype=np.float32).reshape((nb_kpts, 3))
        gt[img_name] = kpts_3d
    return gt

def world2img(kpts_3d, P,K,R,T, xc, yc, K1, K2):
    # kpts_3d: [X,Y,Z] <array>36*3
    # P: <array>3*4
    # distortion: (xc, yc, K1, K2): float
    # return kpts_2d: [w,h] <array>36*2
    #kpts_3d = np.concatenate((kpts_3d, np.ones((36,1))),axis=1).reshape((36,4,1)) #X,Y,Z,1 #world
    #kpts_2d_homo = np.matmul(P.reshape((1,3,4)), kpts_3d).reshape((36,3)) #36*3, xhd, yhd, zhd #img

    kpts_2d = []
    for world in kpts_3d:#for img in kpts_2d_homo:

        #imgcv,_ = cv2.projectPoints(world / 1000, R, T, K, None)
        xw, yw, zw = world[0] / 1000, world[1] / 1000, world[2] / 1000
        img = np.dot(np.array(P),np.array([xw, yw, zw, 1]))
        xd, yd = img[0]/img[2],img[1]/img[2]#img[0]/zw, img[1]/zw

        # distortion
        #w,h = (xd-xc)*F + xc, (yd-yc)*F + yc
        dx, dy = (xd-xc)/1000, (yd-yc)/1000 # dx
        r2 = dx * dx + dy * dy
        F = 1 - K1 * r2 - K2 * r2 * r2
        w = xc + 1000 * dx * F
        h = yc + 1000 * dy * F

        kpts_2d.append([w,h])
        #print("P:",P,K1,K2,xc,yc)
        #print("K:",K,"R:",R,"T:",T)
        #print('w:',world,'cvimg:',imgcv,'img_before:',img,'img_after:',xd,yd,'img_f:',w,h)
    kpts_2d = np.array(kpts_2d, dtype=np.float32)
    #exit()
    return kpts_2d


def img2world(kpts2d_list, P_list):
    # input kpts2d_list: n*2
    # P_list : n* 3*4 (n=2 for 2 cameras)
    # out : 3
    d, point = [], []
    for i in range(len(kpts2d_list)):
        P = P_list[i]
        w1,h1 = kpts2d_list[i][0],kpts2d_list[i][1]
        a1,b1,c1,r1,a2,b2,c2,r2 = P[0][0]-w1*P[2][0], P[0][1]-w1*P[2][1], P[0][2]-w1*P[2][2], w1*P[2][3]-P[0][3],\
                                P[1][0]-h1*P[2][0], P[1][1]-h1*P[2][1], P[1][2]-h1*P[2][2], h1*P[2][3]-P[1][3]
        #print("line:",a1,b1,c1,r1,a2,b2,c2,r2)
        l1 = find_standform_3dline(a1,b1,c1,r1,a2,b2,c2,r2)
        d1 = find_norm_3dline(l1)
        point1 = find_point_3dline(l1)
        #print(d1,point1)
        d.append(d1)
        point.append(point1)
    world = nearest_intersection(np.array(point).reshape((-1,3)), np.array(d).reshape((-1,3)))*1000
    world = world.reshape((3))
    return world

def check_img2world(kpts_3d,root_path):#(kpts_2d, kpts_2d_, P, P_):#(kpts_2d, P,K,R,T, xc, yc, K1, K2):
    #world -> img
    world_orig = kpts_3d[0] #np.array([100,100,100])#kpts_3d[0]
    d,point = [],[]
    for i in [12,20]:#,30,38,67]:#range(2):
        P,K,R,T,xc,yc,K1,K2 = read_calib(root_path + 'calib-new.xml', i+1)

        ## 3d -> 2d(to be labled by hand on the img)
        # world->img
        print(">>>world3d orig:",world_orig)
        kpts_2d,_ = cv2.projectPoints(world_orig / 1000, R, T, K, None)
        print(">>>img2d distor/1000:",kpts_2d)
        ## distortion (w,h = (xd-xc)*F + xc, (yd-yc)*F + yc
        xd,yd=kpts_2d[0][0][0],kpts_2d[0][0][1]
        dx, dy = (xd-xc)/1000, (yd-yc)/1000 # dx
        r2 = dx * dx + dy * dy
        F = 1 - K1 * r2 - K2 * r2 * r2
        w = xc + 1000 * dx * F
        h = yc + 1000 * dy * F
        print(">>>img2d undistor:",w,h)

        ##undistor

        #print(">>>img2d distor res:",kpts_2d_)
        #exit()
        ## img - > world
        w1,h1=w,h
        #w1,h1 = kpts_2d_[0][0][0], kpts_2d_[0][0][1]
        a1,b1,c1,r1,a2,b2,c2,r2 = P[0][0]-w1*P[2][0], P[0][1]-w1*P[2][1], P[0][2]-w1*P[2][2], w1*P[2][3]-P[0][3],\
                                P[1][0]-h1*P[2][0], P[1][1]-h1*P[2][1], P[1][2]-h1*P[2][2], h1*P[2][3]-P[1][3]
        #print("line:",a1,b1,c1,r1,a2,b2,c2,r2)
        l1 = find_standform_3dline(a1,b1,c1,r1,a2,b2,c2,r2)
        d1 = find_norm_3dline(l1)
        point1 = find_point_3dline(l1)
        #print(d1,point1)
        d.append(d1)
        point.append(point1)
    world = nearest_intersection(np.array(point).reshape((-1,3)), np.array(d).reshape((-1,3)))*1000
    #world = find_intersection(point[0],point[1],point[2],point[3])
    print(">>>world 3d res:",world)
    exit()
    return kpts_3d

def find_intersection(pt1,pt2,pt3,pt4):
    # same with nearest_intersection(), not used here.
    import scipy.optimize
    #takes in two lines, the line formed by pt1 and pt2, and the line formed by pt3 and pt4, and finds their intersection or closest point
    #least squares method
    def errFunc(estimates):
        s, t = estimates
        x = pt1 + s * (pt2 - pt1) - (pt3 + t * (pt4 - pt3))
        return x

    estimates = [1, 1]

    sols = scipy.optimize.least_squares(errFunc, estimates)
    s,t = sols.x

    x1 =  pt1[0] + s * (pt2[0] - pt1[0])
    x2 =  pt3[0] + t * (pt4[0] - pt3[0])
    y1 =  pt1[1] + s * (pt2[1] - pt1[1])
    y2 =  pt3[1] + t * (pt4[1] - pt3[1])
    z1 =  pt1[2] + s * (pt2[2] - pt1[2])
    z2 = pt3[2] + t * (pt4[2] - pt3[2])

    x = (x1 + x2) / 2  #halfway point if they don't match
    y = (y1 + y2) / 2  # halfway point if they don't match
    z = (z1 + z2) / 2  # halfway point if they don't match
    print(x,y,z)
    return (x,y,z)


def find_standform_3dline(a1,b1,c1,r1,a2,b2,c2,r2):
    #input: a1x + b1y + c1z = r1
    #       a2x + b2y + c2z = r2
    #output:list of line=[x1,y1,z1,s1,s2,s3]
    #       (x-x1)/s1 = (y-y1)/s2 = (z-z1)/s3

    x1 = (b2*r1-b1*r2)/(a1*b2-a2*b1)
    y1 = (a1*r2-a2*r1)/(a1*b2-a2*b1)
    z1 = 0
    x2 = 0
    z2 = (r1*b2-r2*b1)/(b2*c1-b1*c2)
    y2 = (r1*c2-r2*c1)/(b1*c2-b2*c1)
    '''
    x1 = (b2*(r1-100*c1)-b1*(r2-100*c2))/(a1*b2-a2*b1)
    y1 = (a1*(r2-100*c2)-a2*(r1-100*c1))/(a1*b2-a2*b1)
    z1 = 100
    x2 = 1000
    y2 = ((r1-1000*a1)*b2-(r2-1000*a2)*b1)/(b2*c1-b1*c2)
    z2 = ((r1-1000*a1)*c2-(r2-1000*a2)*c1)/(b1*c2-b2*c1)
    '''
    s1,s2,s3 = x2-x1, y2-y1, z2-z1
    #print("P1:",x1,y1,z1,"P2:",x2,y2,z2,"N:",s1,s2,s3)
    return [x1,y1,z1,s1,s2,s3]
    #return [x1,y1,z1,x2,y2,z2,s1,s2,s3]
def find_norm_3dline(line):
    #input: [x1,y1,z1,s1,s2,s3]
    #out:   (3,) array of unit direction vertor
    x1,y1,z1,s1,s2,s3 = line
    d = np.array([s1,s2,s3])
    d_hat = d/np.linalg.norm(d)
    return d_hat
def find_point_3dline(line):
    #input: [x1,y1,z1,s1,s2,s3]
    #out: (3,) array of a point on the line
    x1,y1,z1,s1,s2,s3 = line
    return np.array([x1, y1, z1])

def nearest_intersection(points, dirs):
    # param points: (N, 3) array of points on the lines
    # param dirs: (N, 3) array of unit direction vectors
    # returns: (3,) array of intersection point
    dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
    points_mat = points[:, :, np.newaxis]
    I = np.eye(3)
    return np.linalg.lstsq( (I - dirs_mat).sum(axis=0),
                            ((I - dirs_mat) @ points_mat).sum(axis=0),
                            rcond=None)[0]

