import math
import os
import struct
import time
import urllib.parse
from functools import partial
from multiprocessing.pool import ThreadPool
from queue import Queue
from struct import unpack
from urllib.request import urlopen

import numpy as np
import torch
import tqdm

categories = ['aircraft carrier', 'airplane', 'alarm clock', 'ambulance', 'angel', 'animal migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball', 'baseball bat', 'basket', 'basketball',
              'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday cake', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge',
              'broccoli', 'broom', 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling fan',
              'cello', 'cell phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise ship', 'cup',
              'diamond', 'dishwasher', 'diving board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather',
              'fence', 'finger', 'fire hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip flops', 'floor lamp', 'flower', 'flying saucer', 'foot', 'fork', 'frog', 'frying pan', 'garden', 'garden hose', 'giraffe', 'goatee',
              'golf club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey puck', 'hockey stick', 'horse', 'hospital', 'hot air balloon', 'hot dog',
              'hot tub', 'hourglass', 'house', 'house plant', 'hurricane', 'ice cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light bulb', 'lighter', 'lighthouse', 'lightning',
              'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug',
              'mushroom', 'nail', 'necklace', 'nose', 'ocean', 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint can', 'palm tree', 'panda', 'pants', 'paper clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil',
              'penguin', 'piano', 'pickup truck', 'picture frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow',
              'rake', 'remote control', 'rhinoceros', 'rifle', 'river', 'roller coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school bus', 'scissors', 'scorpion', 'screwdriver', 'sea turtle', 'see saw', 'shark', 'sheep', 'shoe',
              'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping bag', 'smiley face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square',
              'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop sign', 'stove', 'strawberry', 'streetlight', 'string bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing set', 'sword', 'syringe',
              'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis racquet', 'tent', 'The Eiffel Tower', 'The Great Wall of China', 'The Mona Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
              'tractor', 'traffic light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill',
              'wine bottle', 'wine glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']

categories = ['cat']


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'country_code': country_code,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def encode_figure(fig):
    out = []
    cur_pos = 1
    xs, ys = zip(*fig)
    xmin = min([min(x) for x in xs])
    ymin = min([min(y) for y in ys])
    xmax = max([max(x) for x in xs])
    ymax = max([max(y) for y in ys])
    # +20 bc i don't like edges
    diag = int(math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)) + 20
    xmean = (xmax + xmin) // 2
    ymean = (ymax + ymin) // 2
    for x, y in zip(xs, ys):
        l = len(x)
        L = l * 2 + 1
        v = np.empty([L, 2], dtype=np.int16)
        x = [(a - xmean) * 255 // diag + 255 // 2 for a in x]
        y = [(a - ymean) * 255 // diag + 255 // 2 + 256 for a in y]
        v[0:-1:2, 0] = x
        v[1:-1:2, 0] = y

        p = np.arange(0, l, 1, dtype=np.int16) + cur_pos
        v[0:-1:2, 1] = p
        v[1:-1:2, 1] = p
        v[-1, 0] = 512
        cur_pos += l
        v[-1, 1] = cur_pos - 1
        out.append(v)

    out = np.concatenate(out, axis=0)
    start = np.array([[514, 0]], dtype=np.int16)
    end = np.array([[513, cur_pos]], dtype=np.int16)
    out = np.concatenate([start, out, end], axis=0)

    return out[..., 0][:], out[..., 1][:]


def data_stream(cat, Q, mode):
    while True:
        base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/binary'
        url = f'{base_url}/{urllib.parse.quote(cat)}.bin'
        fig_class = categories.index(cat)
        res = urlopen(url)
        c = 0
        while True:
            try:
                fig = unpack_drawing(res)
                c = c + 1
                if c <= 2 * 1024 and mode == 'test':
                    token, pos = encode_figure(fig['image'])
                    Q.put((fig_class, token, pos))
                elif c > 2 * 1024 and mode == 'train':
                    token, pos = encode_figure(fig['image'])
                    Q.put((fig_class, token, pos))
                elif c > 2 * 1024 and mode == 'test':
                    break
            except struct.error:
                break
        res.close()
        del res


class DrawQuick(torch.utils.data.IterableDataset):
    def __init__(self, mode='train'):
        super().__init__()
        self.dataQ = None
        self.pool = None
        self.mode = mode

    def __next__(self):
        return self.dataQ.get()

    def __repr__(self):
        return (
            'Draw, Quick! dataset\n'
            f'Classes: {" ".join(categories)}\n'
            f'Num. classes: {len(categories)}\n'
        )

    def prepare_pool(self):
        del self.dataQ
        del self.pool
        self.dataQ = Queue(maxsize=10 * 1024)
        f = partial(data_stream, Q=self.dataQ, mode=self.mode)
        self.pool = ThreadPool(len(categories)).map_async(f, categories)
        time.sleep(1)

    def __iter__(self):
        self.prepare_pool()
        return self


def dq_collate(batch):
    fig_class, token, pos = zip(*batch)
    B = len(fig_class)
    out = np.empty([B, 256, 3], dtype=np.int16)
    out[..., 0] = 513
    out[..., 1] = 255
    for i in range(B):
        l = min(len(token[i]), 256)
        out[i, :l, 0] = token[i][:l]
        out[i, :l, 1] = pos[i][:l]
        out[i, :, 2] = fig_class[i]

    return out


if __name__ == '__main__':
    dset = DrawQuick()
    train_dataloader = torch.utils.data.DataLoader(
        dset, batch_size=128, num_workers=1, collate_fn=dq_collate)

    for batch in tqdm.tqdm(train_dataloader):
        pass
