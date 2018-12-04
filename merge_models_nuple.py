#  Scanvan
#
#      Vincent Buntinx - vbuntinx@shogaku.ch
#      Copyright (c) 2016-2018 DHLAB, EPFL
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
    #! \file   merge_models_nuple.py
    #  \author Vincent Buntinx <vbuntinx@shogaku.ch>
    #
    #  Scanvan - https://github.com/ScanVan

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
