import os
from multiprocessing import cpu_count, Pool
from common_functions import parse_all_files_info
from paths import result_dataset_directory, spx_directory
from spectra_parser import parse_and_save_png_by_spx


def convert_file(item):
    old_filename = item['full_path']
    new_filename = os.path.join(result_dataset_directory, item['device'], item['element'],
                                item['file_name'].replace('.spx', '.png'))
    parse_and_save_png_by_spx(old_filename, new_filename)
    print(f'convert file form {old_filename} to {new_filename}')


if __name__ == '__main__':
    items = parse_all_files_info(spx_directory)
    directories = []
    os.makedirs(result_dataset_directory, exist_ok=True)
    for item in items:
        os.makedirs(os.path.join(result_dataset_directory, item['device'], item['element']), exist_ok=True)
    parallelism = cpu_count() - 1
    pool = Pool(processes=parallelism)
    pool.map(convert_file, items)
