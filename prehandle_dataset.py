import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import numpy as np
import base64


def extract_payload(original_traffic: str) -> str:
    """Process payload from original traffic string."""
    payload = original_traffic.split(' -1 ', maxsplit=2)[2].strip()
    payload = payload.replace('-1', '[PAD]')

    datas = [int(x) for x in payload.replace(
        '[PAD]', '').split(' ') if x.strip() != '']
    if len(datas) != 0:
        datas = ''.join([hex(x).replace('0x', '').zfill(2) for x in datas])
        bytes_data = bytes.fromhex(datas)
        base64_str = base64.b64encode(bytes_data).decode('utf-8')
    else:
        base64_str = ''

    padding_loc = payload.find('[PAD]')
    if padding_loc == -1:
        return base64_str
    else:
        return (base64_str+payload[padding_loc:]).replace(' ','')


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
    prehandle_dataset('./datasets/test.csv', './datasets/test_prehandled.csv')
    prehandle_dataset('./datasets/train.csv',
                      './datasets/train_prehandled.csv')
