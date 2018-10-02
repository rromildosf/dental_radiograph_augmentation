import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import loader
import cv2 as cv

def apply_aug( img_coords, seqs ):
    images = []
    for i in range( 4 ):
        try :
            aug = make_augumentation( img_coords, seqs[0] )
            images.append( aug )
        except ValueError as merror:
            print( merror )
            continue
    return images

def augument( img_coords, seq ):
    # images = [ np.copy(img_coords) for i in range(5) ]
    # seq.augment_images( images )
    image = cv.imread( '{}/{}'.format(IMGS_PATH, img_coords[0]) )
    images = []
    for i in range(5):
        images.append( image )
    aug = seq.augment_images( images )
    return images
    # print( images[0].shape )
    # print( aug )

def apply_aug2( img_coords, seqs ):
    images = []
    for i in range(5):
        images.append( make_augumentation( img_coords, seqs ) )
    return images

def make_augumentation( image_coords, seq=None ):
    image = cv.imread( '{}/{}'.format(IMGS_PATH, image_coords[0]) )
    points = []
    
    for i in image_coords[1]:
        points.append( ia.BoundingBox( i[0], i[1], i[2], i[3] ) )

    bbs = ia.BoundingBoxesOnImage( points, shape=image.shape )
    seq_det = seq.to_deterministic()
    image_aug = seq_det.augment_images([image])[0]
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

    # print coordinates before/after augmentation (see below)
    # use after.x_int and after.y_int to get rounded integer coordinates
    points_aug = []

    w, h = image.shape[:2]
    # print( '{}  {}'.format( w, h ) )
    for i in range(len(bbs.bounding_boxes)):
        after = bbs_aug.bounding_boxes[i]

        bb = [after.x1_int, after.y1_int, after.x2_int, after.y2_int]
        
        if (  after.x1_int < 0 or  after.x1_int > w or after.x2_int < 0 or after.x2_int > w or
            after.y1_int < 0 or  after.y1_int > h or after.y2_int < 0 or after.y2_int > h ):
            # print(' ******* Has incorrect boundings.' )
            raise ValueError('Has incorrect boundings.')  
         
        points_aug.append( bb )
        
    image_after =  bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

    return image_aug, image_after, points_aug


   
def show_images( images ):
    import random
    index = random.randint( 0, 10 )
    window = cv.namedWindow('image')
    key = None
    
    for image in images[:index]:
        res = apply_aug( image, make_augs() )
        # images = [res[0], res[1]]

        for r in res:   
            cv.imshow( 'image', r[0] )
            # cv.imshow( 'image', r )
        
            if cv.waitKey(10000) & 0xFF == ord('q'):
                key = 1
                break
        if key != None:
            break
        
    cv.destroyAllWindows()


def heavy_aug():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq

def make_augs():
    seqs = []

    # seqs.append(
    #     iaa.Sequential([
    #         # iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
    #         # iaa.Affine( rotate=10, scale=(1.2, 1.2), shear=15 ), # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    #         iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
    #     ]))
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25) # move pixels locally around (with random strengths)
    #     ]))
    
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.PiecewiseAffine(scale=(0.01, 0.05)), # sometimes move parts of the image around
    #     ]))
    
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
    #         iaa.Affine( rotate=10, scale=(1.2, 1.2), shear=15 ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    #     ]))

    # seqs.append(
    #     iaa.Sequential([
    #         iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
    #         iaa.Affine( rotate=-10, scale=(1.1, 0.8), shear=-15 ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
    #     ]))
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.Multiply((0.7, 0.9)), # change brightness, doesn't affect keypoints
    #         iaa.Affine( rotate=-10 )
    #     ]))
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.Multiply((0.3, 0.5)), # change brightness, doesn't affect keypoints
    #         iaa.Fliplr(0.1), iaa.GaussianBlur((0, 3.0))
    #     ]))
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.GaussianBlur((0, 3.0)), iaa.Affine(translate_px={"x": (-40, 40)})
    #     ]))
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))
    #     ])) 
    # seqs.append(
    #     iaa.Sequential([
    #         iaa.OneOf([
    #             iaa.EdgeDetect(alpha=(0.5, 1.0)),
    #             iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
    #         ]),
    #     ])) 
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seqs.append(
        iaa.Sequential([
            iaa.SomeOf((0, 5),[
                iaa.Affine( rotate=10, scale=(1.2, 1.2), shear=15 ),
                iaa.Affine( rotate=-35, scale=(1.2, 1.2), shear=-15 ),
                iaa.Affine( rotate=-10, scale=(0.8, 1.2), shear=35 ),
                # iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
            ], random_order=True ),
            iaa.SomeOf((0, 3),[
                iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
                iaa.Affine( rotate=-10, scale=(1.1, 0.8), shear=-15 ), # rotate by exactly 10deg and scale to 50-70%, affects keypoints
                iaa.GaussianBlur((0, 3.0))
            ], random_order=True ),
            sometimes(
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-270, 180), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast,
            iaa.GaussianBlur((0, 3.0))
        ], random_order=True)) 

    return seqs

def save_ann( path, ann ):
    text = str(len(ann)) +'\n'
    for a in ann:
        print(a)
        text += ' '.join( [ str(i) for i in a] ) + '\n'
    f = open(path, 'w')
    f.write( text )
    f.close()    

def save_imgaug( images, imgs_path=None, ann_path=None, count_from=0 ):
    counter = count_from
    print("Saving images...")
    for data in images:
        for i in data:
            counter += 1
            print("Saving image #" + str(counter), end=' ', flush=True)

            fname = '{}/img_aug_{}.{}'
            cv.imwrite( fname.format( imgs_path, counter, 'jpeg' ), i[0] )
            save_ann( fname.format( ann_path, counter, 'txt' ), i[2] )
            # print( 'Saved!' )
    # print('Done!')

def init( img_coords, save_path_imgs, save_path_anns ):
    images = []
    print("Applying augumentation")
    counter = 723
    for image in img_coords:
        # try:
        print('Augumentating image:' + image[0] + ' ' + str(counter) )
        res = apply_aug( image, make_augs() )
        # images.append( res )
        counter+=1

        counter += len(res)
        save_imgaug( [res], save_path_imgs, save_path_anns, count_from=counter )

        # except MemoryError as merror:
        #     print('Error on augument image: ' + image[0] + ', skipping.')
        #     print( merror )
       
    # save_imgaug( images, IMGS_PATH, ANNS_PATH )

"""
Constants for path of the images and annotations
"""
LOAD_IMGS_PATH = '../caries/imgs'
LOAD_ANNS_PATH = '../caries/annotations'

SAVE_IMGS_PATH = './imgs'
SAVE_ANNS_PATH = './annotations'

ia.seed(21)
coords = loader.get_coords( LOAD_IMGS_PATH, LOAD_ANNS_PATH )
init( coords, SAVE_IMGS_PATH, SAVE_ANNS_PATH )

# show_images( coords )
