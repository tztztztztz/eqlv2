import os
import csv
import json
import utils
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert Open Images annotations into MS Coco format')
    parser.add_argument('-p', '--path', dest='path',
                        help='path to openimages data', 
                        type=str)
    parser.add_argument('--version',
                        default='v6',
                        choices=['v4', 'v5', 'v6', 'challenge_2019'],
                        type=str,
                        help='Open Images Version')
    parser.add_argument('--subsets',
                        type=str,
                        nargs='+',
                        default=['val', 'train'],
                        choices=['train', 'val', 'test'],
                        help='subsets to convert')
    parser.add_argument('--task',
                        type=str,
                        default='bbox',
                        choices=['bbox', 'panoptic'],
                        help='type of annotations')
    args = parser.parse_args()
    return args

args = parse_args()
base_dir = args.path
if not isinstance(args.subsets, list):
    args.subsets = [args.subsets]

for subset in args.subsets:
    # Convert annotations
    print('converting {} data'.format(subset))

    # Select correct source files for each subset        
    if subset == 'train' and args.version != 'challenge_2019':
        category_sourcefile = 'class-descriptions-boxable.csv'
        image_sourcefile = 'train-images-boxable-with-rotation.csv'
        if args.version == 'v6':
            annotation_sourcefile = 'oidv6-train-annotations-bbox.csv'
        else:
            annotation_sourcefile = 'train-annotations-bbox.csv'
        image_label_sourcefile = 'train-annotations-human-imagelabels-boxable.csv'
        image_size_sourcefile = 'train_sizes-00000-of-00001.csv'
        segmentation_sourcefile = 'validation-annotations-object-segmentation.csv'
        segmentation_folder = 'annotations/validation_masks/'

    elif subset == 'val' and args.version != 'challenge_2019':
        category_sourcefile = 'class-descriptions-boxable.csv'
        image_sourcefile = 'validation-images-with-rotation.csv'
        annotation_sourcefile = 'validation-annotations-bbox.csv'
        image_label_sourcefile = 'validation-annotations-human-imagelabels-boxable.csv'
        image_size_sourcefile = 'validation_sizes-00000-of-00001.csv'
        segmentation_sourcefile = 'validation-annotations-object-segmentation.csv'
        segmentation_folder = 'annotations/validation_masks/'

    elif subset == 'test' and args.version != 'challenge_2019':
        category_sourcefile = 'class-descriptions-boxable.csv'
        image_sourcefile = 'test-images-with-rotation.csv'
        annotation_sourcefile = 'test-annotations-bbox.csv'
        image_label_sourcefile = 'test-annotations-human-imagelabels-boxable.csv'
        image_size_sourcefile = None

    elif subset == 'train' and args.version == 'challenge_2019':
        category_sourcefile = 'challenge-2019-classes-description-500.csv'
        image_sourcefile = 'train-images-boxable-with-rotation.csv'
        annotation_sourcefile = 'challenge-2019-train-detection-bbox.csv'
        image_label_sourcefile = 'challenge-2019-train-detection-human-imagelabels.csv'
        image_size_sourcefile = 'train_sizes-00000-of-00001.csv'
        segmentation_sourcefile = 'challenge-2019-train-segmentation-masks.csv'
        segmentation_folder = 'annotations/challenge_2019_train_masks/'

    elif subset == 'val' and args.version == 'challenge_2019':
        category_sourcefile = 'challenge-2019-classes-description-500.csv'
        image_sourcefile = 'validation-images-with-rotation.csv'
        annotation_sourcefile = 'challenge-2019-validation-detection-bbox.csv'
        image_label_sourcefile = 'challenge-2019-validation-detection-human-imagelabels.csv'
        image_size_sourcefile = 'validation_sizes-00000-of-00001.csv'
        segmentation_sourcefile = 'challenge-2019-validation-segmentation-masks.csv'
        segmentation_folder = 'annotations/challenge_2019_validation_masks/'

    # Load original annotations
    print('loading original annotations ...', end='\r')
    original_category_info = utils.csvread(os.path.join(base_dir, 'annotations', category_sourcefile))
    original_image_metadata = utils.csvread(os.path.join(base_dir, 'annotations', image_sourcefile))
    original_image_annotations = utils.csvread(os.path.join(base_dir, 'annotations', image_label_sourcefile))
    original_image_sizes = utils.csvread(os.path.join('data/', image_size_sourcefile))
    if args.task == 'bbox':
        original_annotations = utils.csvread(os.path.join(base_dir, 'annotations', annotation_sourcefile))
    elif args.task == 'panoptic':
        original_segmentations = utils.csvread(os.path.join(base_dir, 'annotations', segmentation_sourcefile))
        original_mask_dir = os.path.join(base_dir, segmentation_folder)
        segmentation_out_dir = os.path.join(base_dir, 'annotations/{}_{}_{}/'.format(args.task, subset, args.version))
        
    print('loading original annotations ... Done')

    oi = {}

    # Add basic dataset info
    print('adding basic dataset info')
    oi['info'] = {'contributos': 'Vittorio Ferrari, Tom Duerig, Victor Gomes, Ivan Krasin,\
                  David Cai, Neil Alldrin, Ivan Krasinm, Shahab Kamali, Zheyun Feng,\
                  Anurag Batra, Alok Gunjan, Hassan Rom, Alina Kuznetsova, Jasper Uijlings,\
                  Stefan Popov, Matteo Malloci, Sami Abu-El-Haija, Rodrigo Benenson,\
                  Jordi Pont-Tuset, Chen Sun, Kevin Murphy, Jake Walker, Andreas Veit,\
                  Serge Belongie, Abhinav Gupta, Dhyanesh Narayanan, Gal Chechik',
                  'description': 'Open Images Dataset {}'.format(args.version),
                  'url': 'https://storage.googleapis.com/openimages/web/index.html',
                  'version': '{}'.format(args.version),
                  'year': 2020}

    # Add license information
    print('adding basic license info')
    oi['licenses'] = [{'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/'},
                      {'id': 2, 'name': 'Attribution-NonCommercial License', 'url': 'http://creativecommons.org/licenses/by-nc/2.0/'},
                      {'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/'},
                      {'id': 4, 'name': 'Attribution License', 'url': 'http://creativecommons.org/licenses/by/2.0/'},
                      {'id': 5, 'name': 'Attribution-ShareAlike License', 'url': 'http://creativecommons.org/licenses/by-sa/2.0/'},
                      {'id': 6, 'name': 'Attribution-NoDerivs License', 'url': 'http://creativecommons.org/licenses/by-nd/2.0/'},
                      {'id': 7, 'name': 'No known copyright restrictions', 'url': 'http://flickr.com/commons/usage/'},
                      {'id': 8, 'name': 'United States Government Work', 'url': 'http://www.usa.gov/copyright.shtml'}]

    # Convert category information
    print('converting category info')
    oi['categories'] = utils.convert_category_annotations(original_category_info)

    # Convert image mnetadata
    print('converting image info ...')
    image_dir = os.path.join(base_dir, subset)
    oi['images'] = utils.convert_image_annotations(original_image_metadata, original_image_annotations, original_image_sizes, image_dir, oi['categories'], oi['licenses'])

    # Convert instance annotations
    print('converting annotations ...')
    # Convert annotations
    if args.task == 'bbox':
        oi['annotations'] = utils.convert_instance_annotations(original_annotations, oi['images'], oi['categories'], start_index=0)
    elif args.task == 'panoptic':
        oi['annotations'] = utils.convert_segmentation_annotations(original_segmentations, oi['images'], oi['categories'], original_mask_dir, segmentation_out_dir, start_index=0)
        oi['images'] = utils.filter_images(oi['images'], oi['annotations'])

    # Write annotations into .json file
    filename = os.path.join(base_dir, 'annotations/', 'openimages_{}_{}_{}.json'.format(args.version, subset, args.task))
    print('writing output to {}'.format(filename))
    json.dump(oi,  open(filename, "w"))
    print('Done')