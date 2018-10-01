import numpy as np

def pose_estimation(p3d_liste,error_max):
    nb_sph=len(p3d_liste)
    nb_pts=len(p3d_liste[0])
    sv_u_liste=[]
    for i in range(nb_sph):
        sv_u_liste.append(np.ones(nb_pts))
    sv_e_old=0
    sv_e_norm=1
    count=0
    while abs(sv_e_norm-sv_e_old)>error_max:
        sv_e_old=sv_e_norm
        sv_r_liste,sv_t_liste=estimation_rot_trans(p3d_liste,sv_u_liste)
        sv_u_liste,sv_e_liste=estimation_rayons(p3d_liste,sv_u_liste,sv_r_liste,sv_t_liste)
        count+=1
        sv_t_norm=0.0
        for i in range(len(sv_t_liste)):
            sv_t_norm+=np.linalg.norm(sv_t_liste[i])
        sv_e_norm=len(sv_e_liste)*max(sv_e_liste)/sv_t_norm
        #print(count,sv_e_norm)
    sv_scene,positions=pose_scene(p3d_liste,sv_u_liste,sv_r_liste,sv_t_liste)
    return [positions,sv_scene]

def svd_rotation(v,u):
    vu=np.dot(v,u)
    det=round(np.linalg.det(vu),4)
    m=np.identity(3)
    m[2,2]=det
    vm=np.dot(v,m)
    vmu=np.dot(vm,u)
    return vmu

def estimation_rot_trans(p3d_liste,sv_u_liste):
    nb_sph=len(p3d_liste)
    nb_pts=len(p3d_liste[0])
    p3d_liste_exp=[]
    for i in range(nb_sph):
        p3d_exp=[]
        for j in range(nb_pts):
            p3d_exp.append(np.multiply(p3d_liste[i][j],sv_u_liste[i][j]))
        p3d_liste_exp.append(p3d_exp)
    sv_corr_liste=[]  
    for i in range(nb_sph-1):
        sv_corr_liste.append(np.zeros((3,3)))
    sv_cent_liste=[]
    for i in range(nb_sph):
        sv_cent=np.zeros(3)
        for j in range(nb_pts):
            sv_cent+=p3d_liste_exp[i][j]
        sv_cent_liste.append(sv_cent/nb_pts)
    for i in range(nb_sph-1):
        for j in range(nb_pts):
            sv_corr_liste[i]+=np.outer(p3d_liste_exp[i][j]-sv_cent_liste[i],p3d_liste_exp[i+1][j]-sv_cent_liste[i+1])
    sv_r_liste=[]
    sv_t_liste=[]
    for i in range(nb_sph-1):
        svd_U,svd_s,svd_Vt=np.linalg.svd(sv_corr_liste[i])
        sv_r=svd_rotation(svd_Vt.transpose(),svd_U.transpose())
        sv_t=sv_cent_liste[i+1]-np.dot(sv_r,sv_cent_liste[i])
        sv_r_liste.append(sv_r)
        sv_t_liste.append(sv_t)
    return sv_r_liste,sv_t_liste

def centers_determination(sv_r_liste,sv_t_liste):
    nb_sph=len(sv_r_liste)+1
    center_liste=[]
    for i in range(nb_sph):
        center=np.zeros(3)
        for j in range(i):
            k=i-1-j
            center=np.dot(sv_r_liste[k].transpose(),center-sv_t_liste[k])
        center_liste.append(center)
    return center_liste

def azims_determination(azim_liste,sv_r_liste,sv_t_liste):
    nb_sph=len(azim_liste)
    for i in range(nb_sph):
        for j in range(i):
            k=i-1-j
            azim_liste[i]=np.dot(sv_r_liste[k].transpose(),azim_liste[i])
    return azim_liste

def intersection(liste_p,liste_azim):
    nb_pts=len(liste_p)
    sum_v=np.zeros((3,3))
    sum_vp=np.zeros((3,1))
    for i in range(nb_pts):
        azim=np.matrix(liste_azim[i])
        p=np.matrix(liste_p[i])
        v=np.identity(3)-np.dot(azim.transpose(),azim)
        vp=np.dot(v,p.transpose())
        sum_v+=v
        sum_vp+=vp
    inter=np.dot(np.linalg.inv(sum_v),sum_vp)
    inter=np.squeeze(np.asarray(inter))
    rayons=[]
    for i in range(nb_pts):
        centre=liste_p[i]
        azim=liste_azim[i]
        inter_proj=azim*np.dot(inter-centre,azim)/np.dot(azim,azim)
        direction=np.dot(inter_proj,azim)
        if direction<0:
            rayons.append(-np.linalg.norm(inter_proj))
        else:
            rayons.append(+np.linalg.norm(inter_proj))
    return rayons

def intersection_bis(liste_p,liste_azim):
    nb_pts=len(liste_p)
    sum_v=np.zeros((3,3))
    sum_vp=np.zeros((3,1))
    for i in range(nb_pts):
        v11=1.0-liste_azim[i][0]**2
        v22=1.0-liste_azim[i][1]**2
        v33=1.0-liste_azim[i][2]**2
        v12=-liste_azim[i][0]*liste_azim[i][1]
        v13=-liste_azim[i][0]*liste_azim[i][2]
        v23=-liste_azim[i][1]*liste_azim[i][2]
        sum_v[0,0]+=v11
        sum_v[0,1]+=v12
        sum_v[0,2]+=v13
        sum_v[1,0]+=v12
        sum_v[1,1]+=v22
        sum_v[1,2]+=v23
        sum_v[2,0]+=v13
        sum_v[2,1]+=v23
        sum_v[2,2]+=v33
        p1=liste_p[i][0]
        p2=liste_p[i][1]
        p3=liste_p[i][2]
        sum_vp[0,0]+=p1*v11+p2*v12+p3*v13
        sum_vp[1,0]+=p1*v12+p2*v22+p3*v23
        sum_vp[2,0]+=p1*v13+p2*v23+p3*v33
    inter=np.dot(np.linalg.inv(sum_v),sum_vp)
    inter=np.squeeze(np.asarray(inter))
    rayons=[]
    for i in range(nb_pts):
        centre=liste_p[i]
        azim=liste_azim[i]
        inter_proj=azim*np.dot(inter-centre,azim)/np.dot(azim,azim)
        direction=np.dot(inter_proj,azim)
        if direction<0:
            rayons.append(-np.linalg.norm(inter_proj))
        else:
            rayons.append(+np.linalg.norm(inter_proj))
    return rayons

def estimation_rayons(p3d_liste,sv_u_liste,sv_r_liste,sv_t_liste):
    nb_sph=len(p3d_liste)
    nb_pts=len(p3d_liste[0])
    center_liste=centers_determination(sv_r_liste,sv_t_liste)
    sv_e_liste=[]
    for i in range(nb_sph-1):
        sv_e_liste.append(0.0)
    for j in range(nb_pts):
        azim_liste=[]
        for i in range(nb_sph):
            azim_liste.append(p3d_liste[i][j])
        azim_liste=azims_determination(azim_liste,sv_r_liste,sv_t_liste)
        try:
            rayons=intersection_bis(center_liste,azim_liste)
            for i in range(nb_sph):
                sv_u_liste[i][j]=rayons[i]
        except:
            rayons=[]
            for i in range(nb_sph):
                rayons.append(sv_u_liste[i][j])
        inter_liste=[]
        for i in range(nb_sph):
            inter_liste.append(center_liste[i]+azim_liste[i]*rayons[i])
        for i in range(nb_sph-1):
            sv_e_liste[i]=max(sv_e_liste[i],np.linalg.norm(inter_liste[i]-inter_liste[i+1]))
    return sv_u_liste,sv_e_liste

def pose_scene(p3d_liste,sv_u_liste,sv_r_liste,sv_t_liste):
    nb_sph=len(p3d_liste)
    nb_pts=len(p3d_liste[0])
    center_liste=centers_determination(sv_r_liste,sv_t_liste)
    sv_scene=[]
    for j in range(nb_pts):
        azim_liste=[]
        for i in range(nb_sph):
            azim_liste.append(p3d_liste[i][j])
        azim_liste=azims_determination(azim_liste,sv_r_liste,sv_t_liste)
        try:
            rayons=intersection_bis(center_liste,azim_liste)
        except:
            rayons=[]
            for i in range(nb_sph):
                rayons.append(sv_u_liste[i][j])
        inter_liste=[]
        for i in range(nb_sph):
            inter_liste.append(center_liste[i]+azim_liste[i]*rayons[i])
        inter=sum(inter_liste)/len(inter_liste)
        sv_scene.append(inter)
    return [sv_scene,center_liste]
