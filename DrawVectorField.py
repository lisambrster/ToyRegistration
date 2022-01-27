# input is GT or predicted vector field

# output is image with sparse display of vector flow
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

image_path = '/mnt/home/lbrown/MyUnet/regnet/images/'

def MakeVectorFieldImage(x_disp,y_disp,fname):
    #x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    
    #u = 1
    #v = -1
    h,w = x_disp.shape
    skip = 16
    npts_x = int(w/16)
    npts_y = int(h/16)
    print(npts_x,npts_y)
    xx = np.linspace(0,255,npts_x) # min, max, npts
    yy = np.linspace(255,0,npts_y,-1)
    x, y = np.meshgrid(xx,yy)
    #grid = np.indices((256, 256))  # this is now 2x256x256
    
    # scale 1 should be absolute units - but its so small ??
    # rotating 10 degrees - distance is in pixel units
    plt.quiver(x, y, -x_disp[::skip,::skip], y_disp[::skip,::skip],scale_units='xy', scale=1)
    #plt.show()
    plt.savefig(image_path + fname) #'VectorFieldSmooth.jpg')

# compute distance between 2d vector in array disp_vector
# with its neighbor at every location
def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def dist(i,j,disp_x,disp_y):
    # crop image according to i,j
    squared_dist = (shift_image(disp_x,i,j) - disp_x)*(shift_image(disp_x,i,j) - disp_x) + (shift_image(disp_y,i,j) - disp_y)*(shift_image(disp_y,i,j) - disp_y)
    return squared_dist

# regularization measure
# given a displacement vector field (x,y)
# at each point - get mean square distance between each neighboring displacement vector
def Regularization(x_disp,y_disp):
    # vector (x,y)
    # for each of 8 neighbors
    h,w = x_disp.shape
    disp_vector = np.zeros([h,w],dtype=np.float32)


    for i in range(-1,2):
        for j in range(-1,2):
            if (i == 0) and (j == 0):
                continue
            # compute square distance between vector
            sqd = dist(i, j, x_disp, y_disp)
            disp_vector = disp_vector + sqd
    disp_vector = disp_vector / 8
    disp_vector = disp_vector[2:254, 2:254]
    average_MSD = np.mean(disp_vector) # mean over flattened array
    return average_MSD
    
#########################################################################
# main
def main():

    # compute regularization penalty from displacements

    # make vector field from ground truth displacements or predictions
    ground_truth = False
    if ground_truth:
        x_disp = tiff.imread(image_path + 'simreg-x.tif')
        y_disp = tiff.imread(image_path + 'simreg-y.tif')

    else:
        x_disp = tiff.imread(image_path + 'disp-x-pred.tif')
        y_disp = tiff.imread(image_path + 'disp-y-pred.tif')

    x_disp = x_disp[2:258,2:258]
    y_disp = y_disp[2:258,2:258]
    print(np.mean(x_disp),np.mean(y_disp))

    # compute regularization penalty from displacements
    average_MSD = Regularization(x_disp,y_disp)
    print(average_MSD)


    # min 0, max .6
    # min 0, max -.6
    print(x_disp.shape,y_disp.shape,np.max(x_disp),np.min(x_disp),np.max(y_disp),np.min(y_disp))
    fname = 'VectorFieldSmooth.jpg'
    MakeVectorFieldImage(x_disp,y_disp,fname)

if __name__ == "__main__":
    main()