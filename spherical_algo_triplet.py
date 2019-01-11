import numpy as np
from plyfile import PlyData, PlyElement

def filter_radius(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w,n):
    p3d_1_new=[]
    p3d_2_new=[]
    p3d_3_new=[]
    sv_u_new=[]
    sv_v_new=[]
    sv_w_new=[]
    sv_u_mean=np.mean(sv_u)
    sv_v_mean=np.mean(sv_v)
    sv_w_mean=np.mean(sv_w)

    sv_u_std = np.std(sv_u)
    sv_v_std = np.std(sv_v)
    sv_w_std = np.std(sv_w)

    sv_u_range = sv_u_std * n
    sv_v_range = sv_v_std * n
    sv_w_range = sv_w_std * n

    for i in range(len(sv_u)):

        if ( abs( sv_u[i] - sv_u_mean ) < sv_u_range ):

            if ( abs( sv_v[i] - sv_v_mean ) < sv_v_range ):

                if ( abs( sv_w[i] - sv_w_mean ) < sv_w_range ):

                    p3d_1_new.append(p3d_1[i])
                    p3d_2_new.append(p3d_2[i])
                    p3d_3_new.append(p3d_3[i])
                    sv_u_new.append(sv_u[i])
                    sv_v_new.append(sv_v[i])
                    sv_w_new.append(sv_w[i])

    return p3d_1_new,p3d_2_new,p3d_3_new,sv_u_new,sv_v_new,sv_w_new

def pose_estimation(p3d_1,p3d_2,p3d_3,error_max):
    if len(p3d_1)==len(p3d_2)==len(p3d_3):
        longueur=len(p3d_1)
    sv_u=np.ones(longueur)
    sv_v=np.ones(longueur)
    sv_w=np.ones(longueur)
    sv_e_old=0
    sv_e=1
    count=0
    while abs(sv_e-sv_e_old)>error_max:
        sv_e_old=sv_e
        sv_r,sv_t=estimation_rot_trans(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w)
        sv_u,sv_v,sv_w,sv_e=estimation_rayons(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w,sv_r,sv_t)
        p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w=filter_radius(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w,3)
        count+=1
        sv_t_12=sv_t[0,0:3]
        sv_t_23=sv_t[1,3:6]
        sv_t_31=sv_t[2,6:9]
        sv_e_norm=2.0*sv_e/(np.linalg.norm(sv_t_12)+np.linalg.norm(sv_t_23))
        sv_r_12=sv_r[0:3,0:3]
        sv_r_23=sv_r[3:6,3:6]
        sv_r_31=sv_r[6:9,6:9]
        for rotation_test in [sv_r_12,sv_r_23,sv_r_31]:
            the_det=round(np.linalg.det(rotation_test),4)
            if the_det!=1.0:
                print('attention rotation de déterminant non unitaire => '+str(the_det))
                print('-'*30)
        print(count,sv_e_norm)
    print('translations')
    print(sv_t_12)
    print(sv_t_23)
    print('rotations')
    print(sv_r_12)
    print(sv_r_23)
    sv_scene,positions=pose_scene(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w,sv_r,sv_t)
    return [positions,sv_r,sv_scene]

def svd_rotation(v,u):
    vu=np.dot(v,u)
    det=round(np.linalg.det(vu),4)
    m=np.identity(3)
    m[2,2]=det
    vm=np.dot(v,m)
    vmu=np.dot(vm,u)
    return vmu

def block_diag(a,b,c):
    if a.shape==b.shape==c.shape:
        s=a.shape
        x1=[a,np.zeros(s),np.zeros(s)]
        x2=[np.zeros(s),b,np.zeros(s)]
        x3=[np.zeros(s),np.zeros(s),c]
        x=np.block([x1,x2,x3])
        return x
    else:
        print('error: shape')

def estimation_rot_trans(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w):
    if len(p3d_1)==len(p3d_2)==len(p3d_3)==len(sv_u)==len(sv_v)==len(sv_w):
        longueur=len(p3d_1)
    p3d_1_exp=[]
    p3d_2_exp=[]
    p3d_3_exp=[]
    for i in range(longueur):
        p3d_1_exp.append(np.multiply(p3d_1[i],sv_u[i]))
        p3d_2_exp.append(np.multiply(p3d_2[i],sv_v[i]))
        p3d_3_exp.append(np.multiply(p3d_3[i],sv_w[i]))
    sv_corr_12=np.zeros((3,3))
    sv_corr_23=np.zeros((3,3))
    sv_corr_31=np.zeros((3,3))
    sv_cent_1=np.zeros(3)
    sv_cent_2=np.zeros(3)
    sv_cent_3=np.zeros(3)
    sv_diff_1=np.zeros(3)
    sv_diff_2=np.zeros(3)
    sv_diff_3=np.zeros(3)
    for i in range(longueur):
        sv_cent_1+=p3d_1_exp[i]
        sv_cent_2+=p3d_2_exp[i]
        sv_cent_3+=p3d_3_exp[i]
    sv_cent_1/=longueur
    sv_cent_2/=longueur
    sv_cent_3/=longueur
    for i in range(longueur):
        sv_diff_1=p3d_1_exp[i]-sv_cent_1
        sv_diff_2=p3d_2_exp[i]-sv_cent_2
        sv_diff_3=p3d_3_exp[i]-sv_cent_3
        sv_corr_12+=np.outer(sv_diff_1,sv_diff_2)
        sv_corr_23+=np.outer(sv_diff_2,sv_diff_3)
        sv_corr_31+=np.outer(sv_diff_3,sv_diff_1)
    svd_U_12,svd_s_12,svd_Vt_12=np.linalg.svd(sv_corr_12)
    sv_r_12=svd_rotation(svd_Vt_12.transpose(),svd_U_12.transpose())
    svd_U_23,svd_s_23,svd_Vt_23=np.linalg.svd(sv_corr_23)
    sv_r_23=svd_rotation(svd_Vt_23.transpose(),svd_U_23.transpose())
    svd_U_31,svd_s_31,svd_Vt_31=np.linalg.svd(sv_corr_31)
    sv_r_31=svd_rotation(svd_Vt_31.transpose(),svd_U_31.transpose())
    sv_t_12=sv_cent_2-np.dot(sv_r_12,sv_cent_1)
    sv_t_23=sv_cent_3-np.dot(sv_r_23,sv_cent_2)
    sv_t_31=sv_cent_1-np.dot(sv_r_31,sv_cent_3)
    sv_r=block_diag(sv_r_12,sv_r_23,sv_r_31)
    sv_t=block_diag(sv_t_12,sv_t_23,sv_t_31)
    return sv_r,sv_t

def centers_determination(sv_r,sv_t):
    sv_r_12=sv_r[0:3,0:3]
    sv_r_23=sv_r[3:6,3:6]
    sv_r_31=sv_r[6:9,6:9]
    sv_t_12=sv_t[0,0:3]
    sv_t_23=sv_t[1,3:6]
    sv_t_31=sv_t[2,6:9]
    c1=np.zeros(3)
    c2=np.dot(sv_r_12.transpose(),-sv_t_12)
    c3=np.dot(sv_r_12.transpose(),-sv_t_12+np.dot(sv_r_23.transpose(),-sv_t_23))
    c3bis=np.dot(sv_r_31,sv_t_31)
    diff=c3-c3bis
    diff_norm=np.linalg.norm(diff)
    if diff_norm > 10**-3:
        print('attention c3!=c3_bis')
        print(str(c3)+' => c3')
        print(str(c3bis)+' => c3bis')
        print(str(diff)+' => diff')
        print('-'*30)
    return c1,c2,c3

def azims_determination(a1,a2,a3,sv_r,sv_t):
    sv_r_12=sv_r[0:3,0:3]
    sv_r_23=sv_r[3:6,3:6]
    sv_r_31=sv_r[6:9,6:9]
    sv_t_12=sv_t[0,0:3]
    sv_t_23=sv_t[1,3:6]
    sv_t_31=sv_t[2,6:9]
    a2=np.dot(sv_r_12.transpose(),a2)
    a3=np.dot(sv_r_12.transpose(),np.dot(sv_r_23.transpose(),a3))
    return a1,a2,a3

def intersection(liste_p,liste_azim):
    if len(liste_p)==len(liste_azim):
        longueur=len(liste_p)
    sum_v=np.zeros((3,3))
    sum_vp=np.zeros((3,1))
    for i in range(longueur):
        azim=np.matrix(liste_azim[i])
        p=np.matrix(liste_p[i])
        v=np.identity(3)-np.dot(azim.transpose(),azim)
        vp=np.dot(v,p.transpose())
        sum_v+=v
        sum_vp+=vp
    inter=np.dot(np.linalg.inv(sum_v),sum_vp)
    inter=np.squeeze(np.asarray(inter))
    rayons=[]
    for i in range(longueur):
        centre=liste_p[i]
        azim=liste_azim[i]
        inter_proj=azim*np.dot(inter-centre,azim)/np.dot(azim,azim)
        direction=np.dot(inter_proj,azim)
        if direction<0:
            rayons.append(-np.linalg.norm(inter_proj))
        else:
            rayons.append(+np.linalg.norm(inter_proj))
    return rayons

def estimation_rayons(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w,sv_r,sv_t):
    if len(p3d_1)==len(p3d_2)==len(p3d_3)==len(sv_u)==len(sv_v)==len(sv_w):
        longueur=len(p3d_1)
    sv_r_12=sv_r[0:3,0:3]
    sv_r_23=sv_r[3:6,3:6]
    sv_r_31=sv_r[6:9,6:9]
    c1,c2,c3=centers_determination(sv_r,sv_t)
    sv_u_new=[]
    sv_v_new=[]
    sv_w_new=[]
    sv_e=0.0
    for i in range(longueur):
        a1,a2,a3=azims_determination(p3d_1[i],p3d_2[i],p3d_3[i],sv_r,sv_t)
        try:
            rayons=intersection([c1,c2,c3],[a1,a2,a3])
        except:
            rayons=[sv_u[i],sv_v[i],sv_w[i]]
        sv_u_ind=rayons[0]
        sv_v_ind=rayons[1]
        sv_w_ind=rayons[2]
        inter_u=c1+a1*sv_u_ind
        inter_v=c2+a2*sv_v_ind
        inter_w=c3+a3*sv_w_ind
        sv_e=max(sv_e,np.linalg.norm(inter_u-inter_v))
        sv_e=max(sv_e,np.linalg.norm(inter_v-inter_w))
        sv_e=max(sv_e,np.linalg.norm(inter_w-inter_u))
        sv_u_new.append(sv_u_ind)
        sv_v_new.append(sv_v_ind)
        sv_w_new.append(sv_w_ind)
    return sv_u_new,sv_v_new,sv_w_new,sv_e

def pose_scene(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w,sv_r,sv_t):
    if len(p3d_1)==len(p3d_2)==len(p3d_3)==len(sv_u)==len(sv_v)==len(sv_w):
        longueur=len(p3d_1)
    sv_r_12=sv_r[0:3,0:3]
    sv_r_23=sv_r[3:6,3:6]
    sv_r_31=sv_r[6:9,6:9]
    c1,c2,c3=centers_determination(sv_r,sv_t)
    sv_scene=[]
    for i in range(longueur):
        a1,a2,a3=azims_determination(p3d_1[i],p3d_2[i],p3d_3[i],sv_r,sv_t)
        try:
            rayons=intersection([c1,c2,c3],[a1,a2,a3])
        except:
            rayons=[sv_u[i],sv_v[i],sv_w[i]]
            print("attention l'intersection "+str(i)+"n'a pas pu être calculé (parrallelisme)")
        sv_u_ind=rayons[0]
        sv_v_ind=rayons[1]
        sv_w_ind=rayons[2]
        inter=(1.0/3)*(c1+a1*sv_u_ind+c2+a2*sv_v_ind+c3+a3*sv_w_ind)
        sv_scene.append(inter)
    positions=[c1,c2,c3]
    return [sv_scene,positions]

def read_matches(nom_fichier):
    fichier=open(nom_fichier,'r')
    text=fichier.read()
    fichier.close()
    text=text.split('\n')
    elements=[]
    for line in text:
        line_split=line.split(' ')
        for elem in line_split:
            if elem!='':
                elements.append(float(elem))
    nb_matches=int(len(elements)/3)
    #print(len(elements)/3,nb_matches)
    matches=[[],[],[]]
    ind_1=0
    ind_2=0
    for i in range(len(elements)):
        if ind_2==0:
            matches[ind_1].append((elements[i],elements[i+1]))
            ind_1=(ind_1+1)%3
            ind_2=1
        else:
            ind_2=0
    return matches

def pix_to_sph(matches_px):
    dim_x=6016
    dim_y=3008
    matches_sphere=[]
    for i in range(len(matches_px)):
        matches_temp=[]
        for j in range(len(matches_px[i])):
            x_px=matches_px[i][j][0]
            y_px=matches_px[i][j][1]
            theta=0.5*np.pi-np.pi*y_px/dim_y
            phi=2*np.pi*x_px/dim_x
            x=np.cos(theta)*np.cos(phi)
            y=np.cos(theta)*np.sin(phi)
            z=np.sin(theta)
            matches_temp.append(np.array([x,y,z]))
        matches_sphere.append(matches_temp)
    return matches_sphere

def normalize_points(points):
    new_points=[]
    somme=0.0
    for point in points:
        somme+=np.linalg.norm(point)
    lamb=len(points)/somme
    for point in points:
        new_points.append(lamb*point)
    return new_points

def save_ply(scene,name):
    scene_ply=[]
    for elem in scene:
        scene_ply.append(tuple(elem))
    scene_ply=np.array(scene_ply,dtype=[('x','f4'),('y','f4'),('z','f4')])
    el=PlyElement.describe(scene_ply,'vertex',comments=[name])
    PlyData([el],text=True).write(name+'.ply')

matches=read_matches('triplet_matches')
spheres=pix_to_sph(matches)
positions,rotations,scene=pose_estimation(spheres[0],spheres[1],spheres[2],10**-8)
#positions=normalize_points(positions)
#scene=normalize_points(scene)
save_ply(positions,'positions_triplet_test')
save_ply(scene,'scene_triplet_test')
