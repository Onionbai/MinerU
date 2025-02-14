import json
import pymupdf
import requests
import os
import zipfile
import numpy as np
from loguru import logger
from joblib import Parallel, delayed
import subprocess
import multiprocessing

def to_pdf(file_path):
    """支持多种文件类型转换PDF"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            with pymupdf.open(file_path) as doc:
                return doc.tobytes()
        elif ext in ('.jpg', '.jpeg', '.png'):
            with pymupdf.open(file_path) as img:
                return img.convert_to_pdf()
        elif ext in ('.doc', '.docx', '.ppt', '.pptx'):
            raise ValueError(f"Unsupported file type: {ext}")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        logger.error(f"Conversion failed for {file_path}: {e}")
        raise


def do_parse(file_path, url, output_dir, **kwargs):
    try:
        kwargs.setdefault('parse_method', 'auto')
        kwargs.setdefault('debug_able', False)

        # 获取原始文件名
        file_name = os.path.basename(file_path)
        
        response = requests.post(
            url,
            data={'kwargs': json.dumps(kwargs)},
            files={'file': (file_name, to_pdf(file_path))}  # 保持原始文件名
        )

        if response.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存并解压ZIP
            zip_path = os.path.join(output_dir, f'temp_{file_name}.zip')
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            os.remove(zip_path)
            return {
                'status': 'success',
                'output_dir': output_dir,
                'original_file': file_path,
                'extracted_files': zip_ref.namelist()
            }
        else:
            raise Exception(response.text)
    except Exception as e:
        logger.error(f'File: {file_path} - Error: {e}')
        return {'status': 'error', 'message': str(e)}


if __name__ == '__main__':
    # 配置输入目录和支持的文件类型
    url ='http://127.0.0.1:8999/predict'
    input_dir = './input'
    output_dir='./client_tmp'
    supported_ext = ['.pdf', '.jpg', '.jpeg', '.png', '.doc', '.docx', '.ppt', '.pptx']
    thread_num = 56
    # 收集需要处理的文件
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_ext:
                file_list.append(os.path.join(root, file))

    # 并行处理
    if file_list:
#        n_jobs = np.clip(len(file_list), 15, 36)
        print(np.clip(len(file_list)))
        n_jobs = min(len(file_list), max(1, thread_num))
        print("cpu",multiprocessing.cpu_count)
        print("n jobs",n_jobs)
        effective_threads = min(len(file_list), multiprocessing.cpu_count(), thread_num)
#        Parallel(n_jobs=effective_threads, prefer="processes")# 可选processes, threads
        results = Parallel(n_jobs, prefer='threads', verbose=10)(
            delayed(do_parse)(file_path, url, output_dir) for file_path in file_list
        )
#        print(results)
    else:
        print("No supported files found in input directory.")
        
