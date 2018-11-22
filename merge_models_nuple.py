def fusion(model_1,model_2,nb_com):
    pos_1=model_1[0]
    rot_1=model_1[1]
    sce_1=model_1[2]
    pos_2=model_2[0]
    rot_2=model_2[1]
    sce_2=model_2[2]
    len_1=len(pos_1)
    len_2=len(pos_2)
    if len_2>nb_com:
        nb_rot=len_1-nb_com
        scale_factor=np.linalg.norm(pos_1[nb_rot+1]-pos_1[nb_rot])/np.linalg.norm(pos_2[1]-pos_2[0])
        for i in range(len(pos_2)):
            pos_2[i]*=scale_factor
        for i in range(len(sce_2)):
            sce_2[i]*=scale_factor
        translation=(pos_1[nb_rot]-pos_2[0])
        rotation=np.identity(3)
        for k in range(nb_rot-1,-1,-1):
            rotation=np.dot(rot_1[k].transpose(),rotation)
        new_positions=[]
        for i in range(len_2):
            new_pos=np.squeeze(np.asarray(translation+np.dot(rotation,pos_2[i])))
            new_positions.append(new_pos)
        positions=pos_1[:nb_rot]+new_positions
        rotations=rot_1[:nb_rot]+rot_2
        scene=[]
        for i in range(len(sce_1)):
            point=np.squeeze(np.asarray(sce_1[i]))
            scene.append(point)
        for i in range(len(sce_2)):
            point=np.squeeze(np.asarray(translation+np.dot(rotation,sce_2[i])))
            scene.append(point)
        model=[positions,rotations,scene]
        return model
    else:
        return 'error'

def fusion_totale(x):
    model=x[0]
    for i in range(1,len(x)):
        model=fusion(model,x[i],2)
    return model
