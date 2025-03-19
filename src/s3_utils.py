import mimetypes
import os
from pathlib import Path, PosixPath
import traceback 
import boto3

def read_files_from_location(str_path="/home/adira/Downloads/s3"):
    file_paths = Path(str_path)
    return list([entry.name for entry in file_paths.iterdir() if entry.is_dir()])

def map_path_to_file(directories, str_path="/home/adira/Downloads/s3"):
    file_dicts = {}
    for d in directories:
        file_paths = Path("".join([str_path, f'/{d}'])).rglob("*")

        for file_path in file_paths:
            if file_path.is_file():
                full_file_name, file_name, file_extension = file_path.name, file_path.stem, file_path.suffix
                file_path, parent_directory = str(file_path.parent), file_path.parent.name
                content_type = mimetypes.guess_type(full_file_name)[0]
                file_size = os.path.getsize("".join([file_path, '/', full_file_name]))
                posix_path = PosixPath(file_path)
                relative_path = str(posix_path.relative_to(str_path))

                files_arr = []
                file_metadata_dict = {
                    'full_file_name': full_file_name,
                    'file_name': file_name,
                    'file_extension': file_extension,
                    'absolute_path': file_path,
                    'parent_directory': parent_directory,
                    'content_type': content_type,
                    'file_size': file_size,
                    'relative_path': relative_path
                }

                files_arr.append(file_metadata_dict)

                if d in file_dicts:
                    file_dicts[d].append(file_metadata_dict)
                else:
                    file_dicts[d] = files_arr

    return file_dicts


def upload_to_s3(file_dicts, bucket_name, additional_prefix_path=None):
    s3 = boto3.client('s3')

    for file_dict in file_dicts:
        for file in file_dicts.get(file_dict):
            file_name, relative_path, absolute_path = file['full_file_name'], file['relative_path'], \
                                                      file['absolute_path']
            source_path = ''.join([absolute_path, '/', file_name])
            s3_destination_path = ''.join([relative_path, '/', file_name])
            if additional_prefix_path is not None:
                s3_destination_path = f"{additional_prefix_path}/{s3_destination_path}"
            try:
                with open(source_path, 'rb') as f:
                    s3.upload_fileobj(f, bucket_name, s3_destination_path)
            except Exception as e:
                print(e)
                traceback.print_exc()

def upload_folder_to_s3(local_folder_path_str, bucket_name, additional_prefix_path=None):
    directory = read_files_from_location(local_folder_path_str)
    file_dicts_response = map_path_to_file(directory, local_folder_path_str)
    upload_to_s3(file_dicts_response, bucket_name, additional_prefix_path)