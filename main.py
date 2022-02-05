import numpy as np
import tifffile as tiff
import os
import sys

import DrawVectorField
import GenerateSamples

#########################################################################
# main
def main():
    args = sys.argv[1:]
    
    image_path = ("./output-images/")
    if not os.path.exists(image_path):
        print("Directory for output images does not exist. Creating it at: " + image_path)
        os.makedirs(image_path)

    np.random.seed(0)

    # test 3D
    if len(args) > 0 and args[0] == '3D':
        x,y = GenerateSamples.generate_sample_3D()
        print(x.shape,y.shape)
        tiff.imsave(image_path + 'raw1.tif',x)
        tiff.imsave(image_path + 'label1.tif',y)
    elif len(args) > 0 and args[0] == 'jiggle':
        x,dvf = GenerateSamples.generate_sample_jiggle()
        print(x.shape,dvf.shape)

        #fig, axs = plt.subplots(2,2)
        #fig.suptitle('Vertically stacked subplots')
        #axs[0,0].imshow(x,cmap='gray')
        #axs[0,1].imshow(y,cmap='flag')
        print(type(x[0,0,0]))
        print(type(dvf[0,0,0]))

        tiff.imsave(image_path + 'raw1.tif',x[:,:,0])
        tiff.imsave(image_path + 'raw2.tif',x[:,:,1])

        # translation - random +/-
        #x2,y2 = transform_sample(x,y, 40)
        # save label image y
        tiff.imsave(image_path + 'label1.tif',dvf[:,:,0])
        # save raw image x
        tiff.imsave(image_path + 'label2.tif',dvf[:,:,1])
        print(dvf.shape)
        tiff.imsave(image_path + 'dvf_x.tif', dvf[:, :, 3])
        tiff.imsave(image_path + 'dvf_y.tif', dvf[:, :, 2])
        # note vector field has y axis inverted (in FIJI flip label images to match)
        DrawVectorField.MakeVectorFieldImage(dvf[:,:,2], dvf[:,:,3],'jiggle_dvf.jpg',image_path)
        #axs[1,0].imshow(x2,cmap='gray')
        #axs[1,1].imshow(y2,cmap='flag')
        #plt.show()
    else:
        x,y = GenerateSamples.generate_sample()
        tiff.imwrite(image_path + 'raw1.tif', x)
        tiff.imwrite(image_path + 'raw2.tif', y)

if __name__ == "__main__":
    main()
