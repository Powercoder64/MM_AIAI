import pdb
import sys
import os
import numpy as np
import torch.utils.data as data
import utils
from options import *
from config import *
from test import *
from model import *
#from tensorboard_logger import Logger
from thumos_features import *
#import requests
import json

# def send_status_update(messageid, filename, requesttype, output_path, response_type, comment=""):
#     url = "http://aiai-service-service.aiai-ml-curvex-dev.svc.cluster.local/aiai/api/model_run_status_update"
#     payload = json.dumps({
#         "messageid": messageid,
#         "filename": filename,
#         "requestType": requesttype,
#         "responseType": response_type,
#         "comment": comment
#     })
#     with open(output_path + 'neural_network_log.txt', 'w') as f:
#         f.write(str(messageid) + ' ' + str(payload))
#     headers = {'Content-Type': 'application/json'}
#     response = requests.post(url, headers=headers, data=payload)
#     print('send_status_update-call back resposne:  ' + str(response))


if __name__ == "__main__":
    try:
        args = parse_args()
        if 'filename=' in args.filename:
            args.filename = args.filename.split('=')[1]
        #if 'messageid=' in args.messageid:
            #args.messageid = args.messageid.split('=')[1]
        #if 'requesttype=' in args.requesttype:
            #args.requesttype = args.requesttype.split('=')[1]
        # if args.debug:
        #     pdb.set_trace()
        # os.makedirs('/home/project/' + args.filename, exist_ok=True)
        # os.makedirs('/home/project/' + args.filename + '/' + args.messageid, exist_ok=True)
        # os.makedirs('/home/project/' + args.filename + '/' + args.messageid + '/matrixfiles' , exist_ok=True)
        config = Config(args)
        # worker_init_fn = None
        output_path = config.output_path
        if config.seed >= 0:
            utils.set_seed(config.seed)
            worker_init_fn = np.random.seed(config.seed)

        net = BaS_Net(config.len_feature, config.num_classes, config.num_segments)
        net = net.cuda()

        test_loader = data.DataLoader(
            ThumosFeature(data_path=config.data_path, mode='test',
                          modal=config.modal, feature_fps=config.feature_fps,
                          num_segments=config.num_segments, len_feature=config.len_feature,
                          filename=config.filename, seed=config.seed, sampling='random'),

            batch_size=1,
            shuffle=False,
           )

        test_info = {"step": [], "test_acc": [], "average_mAP": [],
                     "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [],
                     "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [],
                     "mAP@0.7": [], "mAP@0.8": [], "mAP@0.9": []}

        logger = 'None'



        try:

            test(net, config, logger, test_loader, test_info, 0, model_file=config.model_file)
            #requesttype = args.requesttype
            #messageid = args.messageid
            filename = args.filename

            #send_status_update(messageid, filename, requesttype, output_path, 'processing-completed', '...')
            utils.save_best_record_thumos(test_info,
                                          os.path.join(config.output_path, "best_record.txt"))
            # with open(output_path + 'neural_network_log.txt', 'w') as f:
            #     f.write(str(messageid) + ' Processing Completed')
        except Exception as e:
            #requesttype = args.requesttype
            #messageid = args.messageid
            filename = args.filename
            #send_status_update(messageid, filename, requesttype, output_path, 'error', str(e))
            raise e
        
    except Exception as e:
        args = parse_args()
        if 'filename=' in args.filename:
            args.filename = args.filename.split('=')[1]
        #if 'messageid=' in args.messageid:
            #args.messageid = args.messageid.split('=')[1]
        #if 'requesttype=' in args.requesttype:
            #args.requesttype = args.requesttype.split('=')[1]
        # save the error message to a file called neural_network_log.txt
        #with open(config.output_path + 'neural_network_log.txt', 'w') as f:
            #f.write(str(messageid) + ' ' + str(e))
        #send_status_update(args.messageid, args.filename, args.requesttype, config.output_path, 'error', str(e))
        #with open(config.output_path + 'neural_network_log.txt', 'w') as f:
            #f.write(str(messageid) + ' ' + str(e))
        raise e






