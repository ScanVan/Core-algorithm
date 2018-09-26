import numpy as np

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
        count+=1
        sv_t_12=sv_t[0,0:3]
        sv_t_23=sv_t[1,3:6]
        sv_t_31=sv_t[2,6:9]
        sv_e_norm=2.0*sv_e/(np.linalg.norm(sv_t_12)+np.linalg.norm(sv_t_23))
        print(count,sv_e_norm)
    sv_scene,positions=pose_scene(p3d_1,p3d_2,p3d_3,sv_u,sv_v,sv_w,sv_r,sv_t)
    return [positions,sv_scene]

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
        sv_u_ind=rayons[0]
        sv_v_ind=rayons[1]
        sv_w_ind=rayons[2]
        inter=(1.0/3)*(c1+a1*sv_u_ind+c2+a2*sv_v_ind+c3+a3*sv_w_ind)
        sv_scene.append(inter)
    positions=[c1,c2,c3]
    return [sv_scene,positions]
