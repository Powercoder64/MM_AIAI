import torch
from omegaconf import OmegaConf
#import requests
import json
import sys
import os
from utils.utils import build_cfg_path, form_list_from_user_input, sanity_check
import time

def check_file_with_retries(filename, max_retries=5, delay_seconds=30):
    attempt = 0
    while attempt < max_retries:
        if os.path.exists(filename):
            print(f"File found: {filename}")
            return True
        else:
            print(f"File not found: {filename}. Retrying {attempt + 1}/{max_retries}...")
            time.sleep(delay_seconds)
            attempt += 1
    raise FileNotFoundError(f"File not found after {max_retries} retries: {filename}")

def parallel_feature_extraction(args):
    error  = 'no error'
    try:

        video_paths = form_list_from_user_input(args)
        print(video_paths)
        filename=args.filename
        #requesttype = args.requesttype
        #messageid = args.messageid

        from models.i3d.extract_i3d import ExtractI3D  # defined here to avoid import errors
        extractor = ExtractI3D(args)
 
        indices = torch.arange(len(video_paths))
        replicas = torch.nn.parallel.replicate(extractor, args.device_ids[:len(indices)])
        inputs = torch.nn.parallel.scatter(indices, args.device_ids[:len(indices)])
        torch.nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
        # closing the tqdm progress bar to avoid some unexpected errors due to multi-threading
        extractor.progress.close()

        try:
           file_exists = check_file_with_retries('./output/' + filename[0:-4] + '_rgb.npy', max_retries=5, delay_seconds=30)
        except FileNotFoundError as e:
           error = str(e)
           print(e)
           #send_status_update(messageid, filename, requesttype, 'error', error)
           raise e

        #send_status_update(messageid, filename, requesttype, 'preprocessing-completed', 'no error')

    except Exception as e:
        filename=args.filename
        #messageid = args.messageid
        #requesttype = args.requesttype
        error_message = str(e)

        #send_status_update(messageid, filename, requesttype,'error', error_message)
        raise e

# def send_status_update(messageid, filename, requesttype, response_type, comment=""):
#     url = "http://aiai-service-service.aiai-ml-curvex-dev.svc.cluster.local/aiai/api/model_run_status_update"
#     payload = json.dumps({
#         "messageid": messageid,
#         "filename": filename,
#         "requestType": requesttype,
#         "responseType": response_type,
#         "comment": comment
#     })
#     with open('home/project/' + str(filename) + '/' + str(messageid) + '/pre_processing_log.txt', 'w') as f:
#         f.write(str(messageid) + ' ' + str(payload))
#
#     headers = {'Content-Type': 'application/json'}
#     try:
#         response = requests.request("POST", url, headers=headers, data=payload)
#         print(response)
#     except Exception as e:
#         with open('home/project/' + str(filename) + '/' + str(messageid) + '/pre_processing_log.txt', 'w') as f:
#             f.write(str(e))
#         raise e

if __name__ == "__main__":
    try:
        cfg_cli = OmegaConf.from_cli()
        cfg_yml = OmegaConf.load(build_cfg_path('i3d'))
        # the latter arguments are prioritized
        cfg = OmegaConf.merge(cfg_yml, cfg_cli)
        # OmegaConf.set_readonly(cfg, True)
        print(OmegaConf.to_yaml(cfg))
        # some printing

        if cfg.on_extraction in ['save_numpy', 'save_pickle']:
            print(f'Saving features files(version March 26 2023) to {cfg.output_path}')
        if cfg.keep_tmp_files:
            print(f'Keeping temp files in {cfg.tmp_path}')
        os.makedirs('/home/viva/project/tmp/', exist_ok=True)
        try:
            #sanity_check(cfg)
            parallel_feature_extraction(cfg)
            #with open('home/project/' + str(cfg.filename) + '/' + str(cfg.messageid) + '/pre_processing_log.txt', 'w') as f:
                #messageid = cfg.messageid
                #f.write(str(messageid) + 'pre processing Completed')
        except Exception as e:
            #with open('home/project/' + str(cfg.filename) + '/' + str(cfg.messageid) + '/pre_processing_log.txt', 'w') as f:
                #messageid = cfg.messageid
                #f.write(str(messageid) + ' ' + str(e))
            #send_status_update(cfg.messageid, cfg.filename, cfg.requesttype,'error', str(e))
            raise e
            
    except Exception as e:
        cfg_cli = OmegaConf.from_cli()
        cfg_yml = OmegaConf.load(build_cfg_path('i3d'))
        # the latter arguments are prioritized
        cfg = OmegaConf.merge(cfg_yml, cfg_cli)
        filename = cfg.filename
        #messageid = cfg.messageid
        #requesttype = cfg.requesttype
        error_message = str(e)

        #send_status_update(messageid, filename, requesttype,'error', error_message)
        raise e
