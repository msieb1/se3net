import torch
import torch.nn.functional as f
from ipdb import set_trace as st

def axisAngleToRotationMatrix(v):
    theta = torch.norm(v, p=2)
    v /= theta

    r11 = 1 + (-v[2]**2 - v[1]**2)*(1-torch.cos(theta)) + 0*torch.sin(theta) 
    r12 =  (v[0] * v[1])*(1-torch.cos(theta)) - v[2] * torch.sin(theta) 
    r13 =  (v[0] * v[2])*(1-torch.cos(theta)) + v[1] * torch.sin(theta)
    r21 =  (v[0] * v[1])*(1-torch.cos(theta)) + v[2] * torch.sin(theta)
    r22 = 1 + (-v[2]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r23 =  (v[1] * v[2])*(1-torch.cos(theta))  - v[0] * torch.sin(theta)
    r31 =  (v[0] * v[2])*(1-torch.cos(theta))  - v[1] * torch.sin(theta)
    r32 =  (v[1] * v[2])*(1-torch.cos(theta)) + v[0] * torch.sin(theta)
    r33 = 1 + (-v[1]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r1 = torch.cat([r11[None],r12[None],r13[None]])
    r2 = torch.cat([r21[None],r22[None],r23[None]])
    r3 = torch.cat([r31[None],r32[None],r33[None]])
    R = torch.stack((r1, r2, r3), dim=0)
    return R

def axisAngleToRotationMatrix_batched(v):
    theta = torch.norm(v, p=2, dim=1)
    v = f.normalize(v, p=2, dim=1)

    r11 = 1 + (-v[:, 2]**2 - v[:, 1]**2)*(1-torch.cos(theta)) + 0*torch.sin(theta) 
    r12 =  (v[:, 0] * v[:, 1])*(1-torch.cos(theta)) - v[:, 2] * torch.sin(theta) 
    r13 =  (v[:, 0] * v[:, 2])*(1-torch.cos(theta)) + v[:, 1] * torch.sin(theta)
    r21 =  (v[:, 0] * v[:, 1])*(1-torch.cos(theta)) + v[:, 2] * torch.sin(theta)
    r22 = 1 + (-v[:, 2]**2 - v[:, 0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r23 =  (v[:, 1] * v[:, 2])*(1-torch.cos(theta))  - v[:, 0] * torch.sin(theta)
    r31 =  (v[:, 0] * v[:, 2])*(1-torch.cos(theta))  - v[:, 1] * torch.sin(theta)
    r32 =  (v[:, 1] * v[:, 2])*(1-torch.cos(theta)) + v[:, 0] * torch.sin(theta)
    r33 = 1 + (-v[:, 1]**2 - v[:, 0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r1 = torch.stack((r11,r12,r13), dim=1).unsqueeze(1)
    r2 = torch.stack((r21,r22,r23), dim=1).unsqueeze(1)
    r3 = torch.stack((r31,r32,r33), dim=1).unsqueeze(1)
    R = torch.cat((r1, r2, r3), dim=1)
    return R

# def axisAngleToRotationMatrix(a):
#     v = a[:-1]
#     theta = a[-1]
#     v /= torch.norm(v, p=2)
#     r11 = 1 + (-v[2]**2 - v[1]**2)*(1-torch.cos(theta)) + 0*torch.sin(theta) 
#     r12 =  (v[0] * v[1])*(1-torch.cos(theta)) - v[2] * torch.sin(theta) 
#     r13 =  (v[0] * v[2])*(1-torch.cos(theta)) + v[1] * torch.sin(theta)
#     r21 =  (v[0] * v[1])*(1-torch.cos(theta)) + v[2] * torch.sin(theta)
#     r22 = 1 + (-v[2]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
#     r23 =  (v[1] * v[2])*(1-torch.cos(theta))  - v[0] * torch.sin(theta)
#     r31 =  (v[0] * v[2])*(1-torch.cos(theta))  - v[1] * torch.sin(theta)
#     r32 =  (v[1] * v[2])*(1-torch.cos(theta)) + v[0] * torch.sin(theta)
#     r33 = 1 + (-v[1]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
#     r1 = torch.cat([r11[None],r12[None],r13[None]])
#     r2 = torch.cat([r21[None],r22[None],r23[None]])
#     r3 = torch.cat([r31[None],r32[None],r33[None]])
#     R = torch.stack((r1, r2, r3), dim=0)
#     return R

def rotationMatrixToAxisAngle(R):
    r11 = R[0, 0]
    r12 = R[0, 1]
    r13 = R[0, 2]
    r21 = R[1, 0]
    r22 = R[1, 1]
    r23 = R[1, 2]
    r31 = R[2, 0]
    r32 = R[2, 1]
    r33 = R[2, 2]
    
    angle = torch.acos((r11 + r22 + r33 - 1)/ 2.0)
    x = (r32 - r23) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    y = (r13 - r31) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    z = (r21 - r12) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    # v = torch.stack((x, y, z, angle), dim=0)
    v =  torch.stack((x, y, z), dim=0)
    v *= angle
    return v

def rotationMatrixToAxisAngle_batched(R):
    r11 = R[:, 0, 0]
    r12 = R[:, 0, 1]
    r13 = R[:, 0, 2]
    r21 = R[:, 1, 0]
    r22 = R[:, 1, 1]
    r23 = R[:, 1, 2]
    r31 = R[:, 2, 0]
    r32 = R[:, 2, 1]
    r33 = R[:, 2, 2]
    
    theta = torch.acos((r11 + r22 + r33 - 1)/ 2.0)
    x = (r32 - r23) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    y = (r13 - r31) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    z = (r21 - r12) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    # v = torch.stack((x, y, z, theta), dim=0)
    v =  torch.stack((x, y, z), dim=1)
    v = torch.mul(theta.unsqueeze(1).repeat(1,3), v)

    return v

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
# Return X-Y-Z (roll pitch yaw)
def rotationMatrixToEulerAngles(R) :
    
    if not type(R) == 'torch.cuda.FloatTensor':
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
         
        singular = sy < 1e-6
     
        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])
    else:
        sy = torch.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
     
        if  not singular :
            x = torch.atan2(R[2,1] , R[2,2])
            y = torch.atan2(-R[2,0], sy)
            z = torch.atan2(R[1,0], R[0,0])
        else :
            x = torch.atan2(-R[1,2], R[1,1])
            y = torch.atan2(-R[2,0], sy)
            z = 0
        return torch.stack((x, y, z))


def sincos2rotm(a_pred, tensor=True):
    # XYZ - roll pitch yaw
    # copy of matlab                                                                                        
    # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx                                                       
    #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx                                                       
    #        -sy            cy*sx             cy*cx]
    if tensor:                                                        
        sinx, cosx = norm_sincos(a_pred[0], a_pred[1]) 
        siny, cosy = norm_sincos(a_pred[2], a_pred[3]) 
        sinz, cosz = norm_sincos(a_pred[4], a_pred[5]) 
        r11 = cosy*cosz
        r12 = sinx*siny*cosz - cosx*sinz
        r13 = cosx*siny*cosz + sinx*sinz
        r21 = cosy*sinz
        r22 = sinx*siny*sinz + cosx*cosz
        r23 = cosx*siny*sinz - sinx*cosz
        r31 = -siny
        r32 = sinx*cosy
        r33 = cosx*cosy
        r1 = torch.cat([r11[None],r12[None],r13[None]])
        r2 = torch.cat([r21[None],r22[None],r23[None]])
        r3 = torch.cat([r31[None],r32[None],r33[None]])
        R = torch.stack((r1, r2, r3), dim=0)
    else:
        sinx, cosx = norm_sincos(a_pred[0], a_pred[1], tensor=False) 
        siny, cosy = norm_sincos(a_pred[2], a_pred[3], tensor=False) 
        sinz, cosz = norm_sincos(a_pred[4], a_pred[5], tensor=False) 
        r11 = cosy*cosz
        r12 = sinx*siny*cosz - cosx*sinz
        r13 = cosx*siny*cosz + sinx*sinz
        r21 = cosy*sinz
        r22 = sinx*siny*sinz + cosx*cosz
        r23 = cosx*siny*sinz - sinx*cosz
        r31 = -siny
        r32 = sinx*cosy
        r33 = cosx*cosy
        r1 = np.concatenate([r11[None],r12[None],r13[None]])
        r2 = np.concatenate([r21[None],r22[None],r23[None]])
        r3 = np.concatenate([r31[None],r32[None],r33[None]])
        R = np.stack((r1, r2, r3), axis=0)       
    return R 

def axisAngletoRotationMatrix(a):
    v = a[:-1]
    theta = a[-1]
    v /= torch.norm(v, p=2)
    r11 = 1 + (-v[2]**2 - v[1]**2)*(1-torch.cos(theta)) + 0*torch.sin(theta) 
    r12 =  (v[0] * v[1])*(1-torch.cos(theta)) - v[2] * torch.sin(theta) 
    r13 =  (v[0] * v[2])*(1-torch.cos(theta)) + v[1] * torch.sin(theta)
    r21 =  (v[0] * v[1])*(1-torch.cos(theta)) + v[2] * torch.sin(theta)
    r22 = 1 + (-v[2]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r23 =  (v[1] * v[2])*(1-torch.cos(theta))  - v[0] * torch.sin(theta)
    r31 =  (v[0] * v[2])*(1-torch.cos(theta))  - v[1] * torch.sin(theta)
    r32 =  (v[1] * v[2])*(1-torch.cos(theta)) + v[0] * torch.sin(theta)
    r33 = 1 + (-v[1]**2 - v[0]**2)*(1-torch.cos(theta)) + 0 * torch.sin(theta)
    r1 = torch.cat([r11[None],r12[None],r13[None]])
    r2 = torch.cat([r21[None],r22[None],r23[None]])
    r3 = torch.cat([r31[None],r32[None],r33[None]])
    R = torch.stack((r1, r2, r3), dim=0)
    return R

def rotationMatrixtoAxisAngle(R):
    r11 = R[0, 0]
    r12 = R[0, 1]
    r13 = R[0, 2]
    r21 = R[1, 0]
    r22 = R[1, 1]
    r23 = R[1, 2]
    r31 = R[2, 0]
    r32 = R[2, 1]
    r33 = R[2, 2]
    
    angle = torch.acos((r11 + r22 + r33 - 1)/ 2.0)
    x = (r32 - r23) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    y = (r13 - r31) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    z = (r21 - r12) / torch.sqrt((r32 - r23)**2 + (r13 - r31)**2 + (r21 - r12)**2)
    v = torch.stack((x, y, z, angle), dim=0)
    return v
