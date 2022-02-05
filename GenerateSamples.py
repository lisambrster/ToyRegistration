# generate synthetic samples of 2D images, label masks, and class dicts
import numpy as np

from scipy.ndimage.morphology import distance_transform_edt
from skimage.segmentation import watershed
from scipy import ndimage as ndi


# n is the image size (nxn)
# blob size - this inversely related to size (.8 is big, 5 is small)
def generate_sample_3D(n=128,nBlobs=6, blob_size = 0.8, gaussian_noise_sd=5):
    center = tuple(np.random.randint(0, n, (3, nBlobs))) # random centers for each of m blobs
    #print('center y ',center[0][:])
    #print('center x ', center[1][:])
    markers = np.zeros((n, n, n), np.float32)
    markers[center] = np.random.permutation(np.arange(1, nBlobs + 1))
    # arange is just numpy's range - so for each marker 1 to m inclusive
    # give the image pixel in the center of the blob one of the labels in (1,m) ? are these the seeds ?
    dist = distance_transform_edt(1 - 1 * (markers > 0)) #+ 4 * ndi.zoom(np.random.uniform(0, 1, (n // 16, n // 16)),
    dist *= blob_size # bigger the smaller ..

    goodmask = np.exp(-.1 * dist) > .1
    y = watershed(dist, markers, mask=goodmask)

    # gaussian noise (last number bigger is sd ?
    x = ndi.filters.gaussian_filter((y > 0).astype(np.float32), gaussian_noise_sd)
    x = x*255
    x = x.astype(np.uint8)
    x = np.stack((x,) * 3, axis=-1)
    y = y.astype(np.uint8)

    return x, y

# n is the image size (nxn)
# blob size - this inversely related to size (.8 is big, 5 is small)

def generate_sample_reg():
    #print('generate sample reg')
    x,y = generate_sample()
    # translation - random (0,1) -> +/- 40
    trans = 5
    x_displacement = 0
    y_displacement = 0
    #x_displacement = np.random.random_sample() * trans - trans/2
    #y_displacement = np.random.random_sample() * trans - trans/2
    #print(x_displacement,y_displacement)
    x2, y2 = translate_sample(x, y, x_displacement, y_displacement)
    # x is two channels (time1 and time2)
    x = np.stack((x,) * 2, axis=-1)
    #print(x.shape)
    x[:,:,1] = x2
    #print(x.shape)
    # y is four channels (labels time1, labels time2, x displacement, y displacement)
    #y.astype(np.float32)
    #y = np.stack((y,) * 4, axis=-1)
    newy = np.zeros([256,256,4],dtype=np.float32)
    ind = np.where(y > 0)
    y[ind] = 1.0
    ind = np.where(y2 > 0)
    y2[ind] = 1.0
    newy[:, :, 0] = y
    newy[:, :, 1] = y2
    newy[:, :, 2] = x_displacement
    newy[:, :, 3] = y_displacement
    #
    #print('y shape, min,max, type ',newy.shape,np.min(newy[:,:,2]),np.max(newy[:,:,2]),type(newy[0,0,2]) )
    return x,newy

def generate_sample_rot():
    #print('generate sample reg')
    x,y = generate_sample()
    # translation - random (0,1) -> +/- 40
    trans = 0
    x_displacement = np.random.random_sample() * trans - trans/2
    y_displacement = np.random.random_sample() * trans - trans/2
    #x_displacement = 2
    #y_displacement = -4
    #print(x_displacement,y_displacement)
    x2, y2, M = transform_sample(x, y, x_displacement, y_displacement)
    # x is two channels (time1 and time2)
    x = np.stack((x,) * 2, axis=-1)
    #print(x.shape)
    x[:,:,1] = x2
    #print(x.shape)
    # y is four channels (labels time1, labels time2, x displacement, y displacement)
    #y.astype(np.float32)
    #y = np.stack((y,) * 4, axis=-1)
    newy = np.zeros([256,256,4],dtype=np.float32)
    ind = np.where(y > 0)
    y[ind] = 1.0
    ind = np.where(y2 > 0)
    y2[ind] = 1.0
    newy[:, :, 0] = y
    newy[:, :, 1] = y2
    # need to make displacement image using transform matrix M
    grid = np.indices((256, 256)) # this is now 2x256x256
    Tones = np.ones([1,256,256])
    Hgrid = np.concatenate((grid,Tones))
    # M is 2x3  ((x,y) X (r1,r2,t)), Hgrid is 3x(256x256) (x,y,1) X (256x256)
    H3 = np.float32([Hgrid[0].flatten(),Hgrid[1].flatten(),Hgrid[2].flatten()])
    new_pts_x = np.matmul( M[0], H3 )
    new_pts_y = np.matmul(M[1], H3)
    #print('gridx min max ', np.max(grid[0]), np.min(grid[0])) # 255, 0
    #print('rotation matrix ',M[0]) # [1 -0.002 0]
    #print('orig pts x min max',np.max(H3), np.min(H3)) # 255, 0
    #print('new pts x min max ',np.max(new_pts_x), np.min(new_pts_x))
    newy[:,:,2]  =  grid[0] - new_pts_x.reshape([256,256]) #
    newy[:,:,3]  =  grid[1] - new_pts_y.reshape([256,256]) #
    #print('disp x min max ',np.max(newy[:,:,2]), np.min(newy[:,:,2]))
    '''
    for i in range(256):
        for j in range(256):
            # M is (2 x 3)
            new_pt = np.matmul( M , np.transpose([i,j,1]) )
            new_i = new_pt[0]
            new_j = new_pt[1]
            disp_x = i - new_i
            disp_y = j - new_j
            newy[i,j,2] = disp_x
            newy[i,j,3] = disp_y
    '''

    #newy[:, :, 2] = x_displacement
    #newy[:, :, 3] = y_displacement

    #
    #print('y shape, min,max, type ',newy.shape,np.min(newy[:,:,2]),np.max(newy[:,:,2]),type(newy[0,0,2]) )
    return x,newy

# generate two time points that differ by small 'jiggle' of each blob
def generate_sample_jiggle(n=256,nBlobs=6, blob_size = 0.8, gaussian_noise_sd=5):
    np.random.seed(100) # 0 for train, 100 for val 
    # don't put blob centers within border_pad
    border_pad = 30
    # 2 is number of dimensions, for 3 blobs returns: (array([32, 77, 56]), array([125, 148, 219]))
    center = tuple(np.random.randint(border_pad, n-border_pad, (2, nBlobs)))  # random centers for each of m blobs
    #print('center y ',center[0][:])
    #print('center x ', center[1][:])
    markers = np.zeros((n, n), np.float32)
    markers[center] = np.arange(1, nBlobs + 1)
    #markers[center] = np.random.permutation(np.arange(1, nBlobs + 1))
    # arange is just numpy's range - so for each marker 1 to m inclusive
    # give the image pixel in the center of the blob one of the labels in (1,m) ? are these the seeds ?
    dist = distance_transform_edt(1 - 1 * (markers > 0)) #+ 4 * ndi.zoom(np.random.uniform(0, 1, (n // 16, n // 16)),
    dist *= blob_size # bigger the smaller ..

    goodmask = np.exp(-.1 * dist) > .1
    y = watershed(dist, markers, mask=goodmask)

    # gaussian noise (last number bigger is sd ?
    x = ndi.filters.gaussian_filter((y > 0).astype(np.float32), gaussian_noise_sd)
    x = x*255
    x = x.astype(np.uint8)
    y = y.astype(np.uint8)

    # now make 'jiggle' of each center
    # translation - random (0,1) -> +/- 40
    trans = border_pad # NOTE -- this should be no bigger than border_pad size otherwise negative or greater than 255 indexing..
    # add 'jiggle' (this random displacement) to each marker
    # for each marker
    d_x = np.zeros((nBlobs),dtype=np.float32)
    d_y = np.zeros((nBlobs),dtype=np.float32)
    c_x = np.zeros((nBlobs),dtype=np.uint8)
    c_y = np.zeros((nBlobs),dtype=np.uint8)
    for imarker in range(nBlobs):
        x_displacement = np.round(np.random.random_sample() * trans - trans / 2)
        y_displacement = np.round(np.random.random_sample() * trans - trans / 2)
        d_x[imarker] =  -x_displacement
        d_y[imarker] =  -y_displacement
        c_x[imarker] = int(center[0][imarker] + x_displacement)
        c_y[imarker] = int(center[1][imarker] + y_displacement)
    new_center = (c_x,c_y)
    #print('new center ',new_center)
    #print('center ',center)
    #print('disp x ',d_x)
    #print('disp y ', d_y)
    new_markers = np.zeros((n, n), np.float32)
    # same label as previous center..
    new_markers[new_center] = markers[center]
    # arange is just numpy's range - so for each marker 1 to m inclusive
    # give the image pixel in the center of the blob one of the labels in (1,m) ? are these the seeds ?
    dist = distance_transform_edt(1 - 1 * (new_markers > 0)) #+ 4 * ndi.zoom(np.random.uniform(0, 1, (n // 16, n // 16)),
    dist *= blob_size # bigger the smaller ..
    goodmask = np.exp(-.1 * dist) > .1
    new_y = watershed(dist, new_markers, mask=goodmask)

    # gaussian noise (last number bigger is sd ?
    new_x = ndi.filters.gaussian_filter((new_y > 0).astype(np.float32), gaussian_noise_sd)
    new_x = new_x * 255
    new_x = new_x.astype(np.uint8)
    new_y = new_y.astype(np.uint8)

    # two frame image
    # combine x and new_x
    x = np.stack((x,) * 2, axis=-1)
    # print(x.shape)
    x[:, :, 1] = new_x
    # print(x.shape)

    # this is GT output
    groundtruth_output = np.zeros([n,n,4],dtype=np.float32)


    # need DVF -- use center to new_center as initialization then interpolate
    # start with center displacements
    #dvf = np.zeros((n, n, 2), np.float32)
    #for imarker in range(nBlobs):
        #print(center[0][imarker],center[1][imarker]) # 182, 113
        #print(d_x[imarker]) # array of zeros..
        #print(d_y[imarker]) # array or list of zeros
        #dvf[center[0][imarker],center[1][imarker],0] = d_x[imarker]
        #dvf[center[0][imarker], center[1][imarker], 1] = d_y[imarker]

    # apply to whole 'blob' (all points with associated label y)
    # should have same displacement as center point
    nlabels = nBlobs
    for ilabel in range(1,nlabels+1):
        ind = np.where(y == ilabel)
        # NOTE 0 axis is Y
        groundtruth_output[ind[0][:],  ind[1][:], 2] = d_y[ilabel-1] #dvf[center[0][imarker],center[1][imarker],0]
        groundtruth_output[ind[0][:], ind[1][:], 3] = d_x[ilabel-1] #dvf[center[0][imarker], center[1][imarker], 1]

    # make label images binary and add to 4-channel ground truth output
    ind = np.where(y > 0)
    y[ind] = 1.0
    ind = np.where(new_y > 0)
    new_y[ind] = 1.0
    groundtruth_output[:, :, 0] = y
    groundtruth_output[:, :, 1] = new_y

    # then smooth ? apply gaussian to all locations with zero values (that way we leave original local
    # displacements alone..)

    
    return x, groundtruth_output



def generate_sample(n=256,nBlobs=6, blob_size = 0.8, gaussian_noise_sd=5):

    #center = tuple(np.random.randint(0, n, (2, nBlobs))) # random centers for each of m blobs
    center = tuple(np.random.randint(10, n-10, (2, nBlobs)))  # random centers for each of m blobs
    #print('center y ',center[0][:])
    #print('center x ', center[1][:])
    markers = np.zeros((n, n), np.float32)
    markers[center] = np.random.permutation(np.arange(1, nBlobs + 1))
    # arange is just numpy's range - so for each marker 1 to m inclusive
    # give the image pixel in the center of the blob one of the labels in (1,m) ? are these the seeds ?
    dist = distance_transform_edt(1 - 1 * (markers > 0)) #+ 4 * ndi.zoom(np.random.uniform(0, 1, (n // 16, n // 16)),
    dist *= blob_size # bigger the smaller ..

    goodmask = np.exp(-.1 * dist) > .1
    y = watershed(dist, markers, mask=goodmask)

    # gaussian noise (last number bigger is sd ?
    x = ndi.filters.gaussian_filter((y > 0).astype(np.float32), gaussian_noise_sd)
    x = x*255
    x = x.astype(np.uint8)
    #x = np.stack((x,) * 3, axis=-1)
    y = y.astype(np.uint8)
    return x, y

'''
def translate_sample(x,y,translation_size_x,translation_size_y):
    y = y.astype(np.uint8)
    # start with random translation
    tx = translation_size_x #np.random.uniform(-translation_size,translation_size_x)
    ty = translation_size_y #np.random.uniform(-translation_size,translation_size_y)
    theta = 0 #(45)*np.pi/180
    # need to rotate around center
    M = np.float32([[np.cos(theta), -np.sin(theta), tx], [np.sin(theta), np.cos(theta), ty]])
    #print('transform ',M)
    x2 = cv2.warpAffine(x, M, (x.shape[1], x.shape[0]))
    y2 = cv2.warpAffine(y, M, (y.shape[1], y.shape[0]))
    return x2,y2
'''

def transform_sample(x,y,translation_size_x,translation_size_y):
    y = y.astype(np.uint8)
    # start with random translation
    tx = translation_size_x #np.random.uniform(-translation_size,translation_size_x)
    ty = translation_size_y #np.random.uniform(-translation_size,translation_size_y)
    rot = 5
    theta = np.random.uniform(-rot,rot)*np.pi/180
    #theta =  0*np.pi/180
    # need to rotate around center
    M = np.float32([[np.cos(theta), -np.sin(theta), tx], [np.sin(theta), np.cos(theta), ty]])
    #print('transform ',M)
    x2 = x #cv2.warpAffine(x, M, (x.shape[1], x.shape[0]))
    y2 = y #cv2.warpAffine(y, M, (y.shape[1], y.shape[0]))
    return x2,y2,M


# UNFINISHED
# add to current DVF (x,y) a small local deformation
# the local deformation has a random value between +/- size
# place value every shift along x (and y) 
# then interpolate to get all the other points
def AddSmallLocalDeformation(x,y,size,shift):
    n = x.shape[0]
    LocalDeformationX = np.zeros((n, n), np.float32)
    LocalDeformationY = np.zeros((n, n), np.float32)
    for i in range(n,shift):
        for j in range(n,shift):
            LocalDeformation[i,j] = np.random.uniform(-size,size)
            LocalDeformation[i, j] = np.random.uniform(-size, size)
    # now interpolate the points in between

    





