import os
import numpy as np

imgs = []

def get_names( array, ext ):
    newarr = []
    for i in [i.split('.') for i in array]:
        if( i[1].lower() == ext.lower() ):
            newarr.append( i[0] )
    return newarr

def get_coords( imgs_path, annotations_path, img_ext='JPEG', ann_ext='txt' ):
    imgs = get_names( os.listdir( imgs_path ), img_ext )
    ants  = get_names( os.listdir( annotations_path ), ann_ext )

    pairs = []
    for a in ants:
        for i in imgs:
            if( a == i ):
                ann = load_file( '{}/{}.{}'.format(annotations_path, a, ann_ext ) )
                pairs.append( (i +'.'+img_ext, ann) )
    # print( pairs )
    return pairs

def load_file( path ):
    coords = []
    with open( path, 'r' ) as f:
        l = 'l'
        while l != '':
            l = f.readline().strip()
            if( l == '' ): break

            l = l.split(' ')
            if( len(l) > 1 ): # cordenada válida
                coords.append( [ int(i) for i in l] )
    return coords


# coords = get_coords( './caries/imgs/', './caries/annotations/' )
# # print( load_file( './caries/annotations/1.txt') )
# for i in coords:
#     print( i )