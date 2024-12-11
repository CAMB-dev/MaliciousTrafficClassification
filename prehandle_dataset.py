import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import numpy as np
import time

random_words_300 = ['watermelon', 'planet', 'cookie', 'jackal', 'giraffe', 'quilt', 'hotdog', 'otter', 'lantern', 'gannet', 'raccoon', 'x-ray', 'pencil', 'raft', 'ruler', 'karma', 'yield', 'hat', 'yarn', 'gym', 'cheetah', 'ivy', 'sloth', 'fern', 'nectar', 'peacock', 'mango', 'trunk', 'hammer', 'zip', 'effect', 'gorilla', 'bison', 'worm', 'truffle', 'acorn', 'xylophone', 'hippopotamus', 'highway', 'fox', 'plankton', 'eagle', 'ink', 'stone', 'wolf', 'tiger', 'starfish', 'tornado', 'balloon', 'garlic', 'anteater', 'whisk', 'whispery', 'jump', 'dragon', 'octagon', 'carob', 'dancer', 'wind', 'cat', 'pizza', 'elephant', 'microwave', 'squirrel', 'boulevard', 'jam', 'web', 'panda', 'zephyr', 'beach', 'snail', 'snake', 'nose', 'seal', 'gourd', 'cricket', 'javelin', 'lamp', 'vision', 'lock', 'perch', 'falcon', 'queen', 'umbrella', 'herb', 'yellow', 'telescope', 'feather', 'horizon', 'pig', 'clam', 'quill', 'opera', 'daisy', 'dove', 'kingfish', 'gashawk', 'whale', 'volcano', 'unity', 'zenith', 'dragonfly', 'twister', 'zebra', 'caterpillar', 'panther', 'tree', 'egg', 'bottle', 'dogfish', 'narwhal', 'quiver', 'shrimp', 'video', 'zinc', 'sled', 'mouse', 'dog', 'rose', 'treasure', 'parrot', 'wheel', 'vegetable', 'flute', 'jelly', 'mast', 'eel', 'hawk', 'chocolate', 'goat', 'treble', 'jab', 'monkey', 'nutmeg', 'dentist', 'kiwi', 'pear', 'dingo', 'orange', 'hill', 'aqua', 'neon', 'arrowband', 'grape', 'needle', 'rocket', 'kettle', 'uniform', 'summit', 'yo-yo',
                    'melon', 'news', 'vulture', 'iron', 'elm', 'turtle', 'exit', 'antelope', 'willow', 'island', 'tulip', 'banana', 'jaguar', 'book', 'lion', 'revamp', 'kite', 'elf', 'olive', 'bonfire', 'rhinoceros', 'deer', 'horn', 'zealot', 'yacht', 'guitar', 'jungle', 'unicorn', 'iguana', 'alleyboard', 'octopus', 'iceberg', 'herring', 'jade', 'quest', 'firefly', 'llama', 'notebook', 'vortex', 'frog', 'cloud', 'abacus', 'victory', 'king', 'envelope', 'yeti', 'spirit', 'glove', 'lime', 'nut', 'astronaut', 'water', 'hammock', 'dolphin', 'brick', 'jet', 'lemon', 'anchor', 'zest', 'coral', 'energy', 'flamingo', 'harp', 'blade', 'meadow', 'boat', 'cactus', 'rosebug', 'lemur', 'bicycle', 'hen', 'sprite', 'pony', 'diamond', 'yodel', 'igloo', 'xerox', 'sunbeam', 'goose', 'quail', 'vase', 'moose', 'kernel', 'wizard', 'army', 'kangaroo', 'owl', 'frost', 'napkin', 'glider', 'apple', 'dandelion', 'patch', 'jigsaw', 'rodent', 'pumpkin', 'trap', 'trampoline', 'helmet', 'fire', 'elbow', 'flame', 'hero', 'joker', 'vanilla', 'clover', 'armadillo', 'walnut', 'finch', 'urn', 'violin', 'compass', 'mare', 'ant', 'emerald', 'fence', 'venus', 'manor', 'ultraviolet', 'badge', 'sun', 'gecko', 'lizard', 'icecap', 'honey', 'pancake', 'mountain', 'echo', 'arrow', 'lighthouse', 'jellyfish', 'mustard', 'horse', 'velvet', 'ladder', 'airbrush', 'vine', 'xenon', 'ostrich', 'warp', 'albatross', 'rabbit', 'door', 'baker', 'ember', 'seafoam', 'koala', 'snaptrace', 'tumbleweed', 'cabbage']


def extract_payload(original_traffic: str) -> str:
    """Process payload from original traffic string."""
    payload = original_traffic.split(' -1 ', maxsplit=2)[2].strip()
    payload = payload.replace('-1', '[PAD]')

    datas = [int(x) for x in payload.replace(
        '[PAD]', '').split(' ') if x.strip() != '']

    datas = [random_words_300[x] for x in datas]
    payload2words = ' '.join(datas)

    padding_loc = payload.find('[PAD]')
    if padding_loc == -1:
        return payload2words
    else:
        return payload2words+payload[padding_loc:].replace(' ', '')


def process_chunk(chunk_data: Tuple[List, int, int]) -> List[List]:
    """Process a chunk of data and show progress."""
    data, chunk_id, total_chunks = chunk_data
    result = []
    chunk_size = len(data)

    for idx, row in enumerate(data):
        result.append([extract_payload(row[0]), row[1]])
        if idx % 1000 == 0:
            print(
                f'Chunk {chunk_id}/{total_chunks}: Processed {idx}/{chunk_size}')

    return result


def load_datas(path: str) -> pd.DataFrame:
    """Load and process data using multiprocessing."""
    df: pd.DataFrame = pd.read_csv(path)
    print(df.info())

    # 确定CPU核心数和数据分块
    num_cores = cpu_count()
    # 将数据分成比CPU核心数多一些的块，以便更好的负载均衡
    num_chunks = num_cores * 4

    # 将数据分块
    df_arrays = np.array_split(df.values, num_chunks)

    # 准备进程池的输入数据
    chunk_data = [(chunk, i+1, num_chunks)
                  for i, chunk in enumerate(df_arrays)]

    # 使用进程池处理数据
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_chunk, chunk_data)

    # 合并所有结果
    all_results = []
    for chunk_result in results:
        all_results.extend(chunk_result)

    return pd.DataFrame(all_results, columns=['payload', 'label'])


def prehandle_dataset(path: str, output_path: str = './datasets/output.csv') -> None:
    """Prehandle dataset and save to output path."""
    print(f"Processing {path}...")
    datas = load_datas(path)
    datas.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    prehandle_dataset('./datasets/test.csv', './datasets/test_word.csv')
    prehandle_dataset('./datasets/train.csv',
                      './datasets/train_word.csv')
