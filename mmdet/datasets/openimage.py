import itertools
import logging
import os.path as osp
import tempfile

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class OpenimageDataset(CocoDataset):

    CLASSES = ('Infant bed', 'Rose', 'Flag', 'Flashlight', 'Sea turtle',
               'Camera', 'Animal', 'Glove', 'Crocodile', 'Cattle', 'House',
               'Guacamole', 'Penguin', 'Vehicle registration plate', 'Bench', 
               'Ladybug', 'Human nose', 'Watermelon', 'Flute', 
               'Butterfly', 'Washing machine', 'Raccoon', 'Segway', 'Taco', 
               'Jellyfish', 'Cake', 'Pen', 'Cannon', 'Bread', 'Tree', 
               'Shellfish', 'Bed', 'Hamster', 'Hat', 'Toaster', 'Sombrero', 
               'Tiara', 'Bowl', 'Dragonfly', 'Moths and butterflies', 
               'Antelope', 'Vegetable', 'Torch', 'Building', 
               'Power plugs and sockets', 'Blender', 'Billiard table', 'Cutting board', 
               'Bronze sculpture', 'Turtle', 'Broccoli', 'Tiger', 'Mirror', 
               'Bear', 'Zucchini', 'Dress', 'Volleyball', 'Guitar', 
               'Reptile', 'Golf cart', 'Tart', 'Fedora', 'Carnivore', 'Car', 
               'Lighthouse', 'Coffeemaker', 'Food processor', 'Truck', 'Bookcase',
               'Surfboard', 'Footwear', 'Bench', 'Necklace', 'Flower', 'Radish', 
               'Marine mammal', 'Frying pan', 'Tap', 'Peach', 'Knife', 'Handbag', 
               'Laptop', 'Tent', 'Ambulance', 'Christmas tree', 'Eagle', 
               'Limousine', 'Kitchen & dining room table', 'Polar bear', 
               'Tower', 'Football', 'Willow', 'Human head', 'Stop sign', 
               'Banana', 'Mixer', 'Binoculars', 'Dessert', 'Bee', 'Chair', 'Wood-burning stove', 'Flowerpot', 'Beaker', 'Oyster', 'Woodpecker', 
               'Harp', 'Bathtub', 'Wall clock', 'Sports uniform', 'Rhinoceros', 'Beehive', 'Cupboard', 'Chicken', 'Man', 'Blue jay', 'Cucumber', 'Balloon', 'Kite', 'Fireplace', 'Lantern', 'Missile', 'Book',
               'Spoon', 'Grapefruit', 'Squirrel', 'Orange', 'Coat', 'Punching bag', 'Zebra', 'Billboard', 'Bicycle', 'Door handle', 'Mechanical fan', 'Ring binder', 'Table', 'Parrot', 'Sock', 'Vase', 'Weapon', 
               'Shotgun', 'Glasses', 'Seahorse', 'Belt', 'Watercraft', 'Window',
               'Giraffe', 'Lion', 'Tire', 'Vehicle', 'Canoe', 'Tie', 'Shelf', 'Picture frame', 'Printer', 'Human leg', 'Boat', 'Slow cooker', 'Croissant', 'Candle', 'Pancake',
               'Pillow', 'Coin', 'Stretcher', 'Sandal', 'Woman', 'Stairs', 
               'Harpsichord', 'Stool', 'Bus', 'Suitcase', 'Human mouth', 'Juice', 'Skull', 'Door', 'Violin', 'Chopsticks', 'Digital clock', 'Sunflower', 'Leopard', 'Bell pepper', 'Harbor seal',
               'Snake', 'Sewing machine', 'Goose', 'Helicopter', 'Seat belt', 'Coffee cup', 'Microwave oven', 'Hot dog', 'Countertop', 'Serving tray', 'Dog bed', 'Beer', 'Sunglasses', 'Golf ball', 'Waffle',
               'Palm tree', 'Trumpet', 'Ruler', 'Helmet', 'Ladder', 'Office building', 'Tablet computer', 'Toilet paper', 'Pomegranate', 'Skirt', 'Gas stove', 'Cookie', 'Cart', 'Raven', 'Egg', 'Burrito', 'Goat', 'Kitchen knife', 'Skateboard', 'Salt and pepper shakers', 'Lynx', 'Boot', 'Platter', 'Ski', 'Swimwear', 'Swimming pool', 'Drinking straw', 'Wrench', 'Drum', 'Ant', 'Human ear', 'Headphones', 'Fountain', 'Bird', 'Jeans', 'Television', 'Crab',
               'Microphone', 'Home appliance', 'Snowplow', 'Beetle', 'Artichoke', 'Jet ski', 'Stationary bicycle', 'Human hair', 'Brown bear', 'Starfish', 'Fork', 'Lobster', 'Corded phone', 'Drink', 'Saucer', 'Carrot', 'Insect', 'Clock', 'Castle', 'Tennis racket', 'Ceiling fan', 'Asparagus', 'Jaguar', 'Musical instrument', 'Train', 'Cat', 'Rifle', 'Dumbbell', 'Mobile phone', 'Taxi', 'Shower', 'Pitcher', 'Lemon', 'Invertebrate', 'Turkey', 'High heels', 'Bust', 'Elephant', 'Scarf', 'Barrel', 'Trombone', 'Pumpkin', 'Box', 'Tomato', 'Frog', 'Bidet', 'Human face', 'Houseplant', 'Van', 'Shark', 'Ice cream', 'Swim cap', 'Falcon', 'Ostrich', 'Handgun', 'Whiteboard', 'Lizard', 'Pasta', 'Snowmobile', 'Light bulb', 'Window blind', 'Muffin', 'Pretzel', 'Computer monitor', 'Horn', 'Furniture', 'Sandwich', 'Fox', 'Convenience store', 'Fish', 'Fruit', 'Earrings', 'Curtain', 'Grape', 'Sofa bed', 'Horse', 'Luggage and bags', 'Desk', 'Crutch', 'Bicycle helmet', 'Tick', 'Airplane', 'Canary', 'Spatula', 'Watch', 'Lily', 'Kitchen appliance', 'Filing cabinet', 'Aircraft', 'Cake stand', 'Candy', 'Sink', 'Mouse', 'Wine', 'Wheelchair', 'Goldfish', 'Refrigerator', 'French fries', 'Drawer', 'Treadmill', 'Picnic basket', 'Dice', 'Cabbage', 'Football helmet', 'Pig', 'Person', 'Shorts', 'Gondola', 'Honeycomb', 'Doughnut', 'Chest of drawers', 'Land vehicle', 'Bat', 'Monkey', 'Dagger', 'Tableware', 'Human foot', 'Mug', 'Alarm clock', 'Pressure cooker', 'Human hand', 'Tortoise', 'Baseball glove', 'Sword', 'Pear', 'Miniskirt', 'Traffic sign', 'Girl', 'Roller skates', 'Dinosaur', 'Porch', 'Human beard', 'Submarine sandwich', 'Screwdriver', 'Strawberry', 'Wine glass', 'Seafood', 'Racket', 'Wheel', 'Sea lion', 'Toy', 'Tea', 'Tennis ball', 'Waste container', 'Mule', 'Cricket ball', 'Pineapple', 'Coconut', 'Doll', 'Coffee table', 'Snowman', 'Lavender', 'Shrimp', 'Maple', 'Cowboy hat', 'Goggles', 'Rugby ball', 'Caterpillar', 'Poster', 'Rocket', 'Organ', 'Saxophone', 'Traffic light', 'Cocktail', 'Plastic bag', 'Squash', 'Mushroom', 'Hamburger', 'Light switch', 'Parachute', 'Teddy bear', 'Winter melon', 'Deer', 'Musical keyboard', 'Plumbing fixture', 'Scoreboard', 'Baseball bat', 'Envelope', 'Adhesive tape', 'Briefcase', 'Paddle', 'Bow and arrow', 'Telephone', 'Sheep', 'Jacket', 'Boy', 'Pizza', 'Otter', 'Office supplies', 'Couch', 'Cello', 'Bull', 'Camel', 'Ball', 'Duck', 'Whale', 'Shirt', 'Tank', 'Motorcycle', 'Accordion', 'Owl', 'Porcupine', 'Sun hat', 'Nail', 'Scissors', 'Swan', 'Lamp', 'Crown', 'Piano', 'Sculpture', 'Cheetah', 'Oboe', 'Tin can', 'Mango', 'Tripod', 'Oven', 'Mouse', 'Barge', 'Coffee', 'Snowboard', 'Common fig', 'Salad', 'Marine invertebrates', 'Umbrella', 'Kangaroo', 'Human arm', 'Measuring cup', 'Snail', 'Loveseat', 'Suit', 'Teapot', 'Bottle', 'Alpaca', 'Kettle', 'Trousers', 'Popcorn', 'Centipede', 'Spider', 'Sparrow', 'Plate', 'Bagel', 'Personal care', 'Apple', 'Brassiere', 'Bathroom cabinet', 'studio couch', 'Computer keyboard', 'Table tennis racket', 'Sushi', 'Cabinetry', 'Street light', 'Towel', 'Nightstand', 'Rabbit', 'Dolphin', 'Dog', 'Jug', 'Wok', 'Fire hydrant', 'Human eye', 'Skyscraper', 'Backpack', 'Potato', 'Paper towel', 'Lifejacket', 'Bicycle wheel', 'Toilet')

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        try:
            from lvis import LVIS
        except ImportError:
            raise ImportError('Please follow config/lvis/README.md to '
                              'install open-mmlab forked lvis first.')
        self.coco = LVIS(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 groupwise=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.array([0.5])):
        """Evaluation in LVIS protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None):
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float]): IoU threshold used for evaluating
                recalls. If set to a list, the average recall of all IoUs will
                also be computed. Default: 0.5.

        Returns:
            dict[str, float]: LVIS style metrics.
        """

        try:
            from lvis import LVISResults
            from .openimage_eval import OpenimageEval
        except ImportError:
            raise ImportError('Please follow config/lvis/README.md to '
                              'install open-mmlab forked lvis first.')
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)

        eval_results = {}
        # get original api
        lvis_gt = self.coco
        for metric in metrics:
            msg = 'Evaluating {}...'.format(metric)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results['AR@{}'.format(num)] = ar[i]
                    log_msg.append('\nAR@{}\t{:.4f}'.format(num, ar[i]))
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError('{} is not in results'.format(metric))
            try:
                lvis_dt = LVISResults(lvis_gt, result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            lvis_eval = OpenimageEval(lvis_gt, lvis_dt, iou_type)
            lvis_eval.params.imgIds = self.img_ids
            lvis_eval.params.iouThrs = iou_thrs
            if metric == 'proposal':
                lvis_eval.params.useCats = 0
                lvis_eval.params.maxDets = list(proposal_nums)
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                for k, v in lvis_eval.get_results().items():
                    if k.startswith('AR'):
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[k] = val
            else:
                lvis_eval.evaluate()
                lvis_eval.accumulate()
                lvis_eval.summarize()
                lvis_results = lvis_eval.get_results()
                classwise = True
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = lvis_eval.eval['precision']
                    # precision: (iou, recall, cls, area range)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.load_cats([catId])[0]
                        precision = precisions[:, :, idx, 0]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                    with open(f"per-category-ap-{metric}.txt", 'w') as f:
                        f.write(table.table)

                for k, v in lvis_results.items():
                    if k.startswith('AP'):
                        key = '{}_{}'.format(metric, k)
                        val = float('{:.3f}'.format(float(v)))
                        eval_results[key] = val
                ap_summary = ' '.join([
                    '{}:{:.3f}'.format(k, float(v))
                    for k, v in lvis_results.items() if k.startswith('AP')
                ])
                eval_results['{}_mAP_copypaste'.format(metric)] = ap_summary
            lvis_eval.print_results()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
