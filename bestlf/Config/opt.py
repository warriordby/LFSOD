import argparse
import yaml
import os
import pprint
script_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

#可选参数
parser.add_argument('--model_name', type=str, default="agent_pvt.py-agent_pvt_v2_b2", help='model name')
parser.add_argument('--pretrained', type=str,
                    default='/root/autodl-tmp/LFSOD/bestlf/Pretain_weight/fpn_agent_pvt_s_12-16-28-28.pth', help='train from checkpoints')
parser.add_argument('--task', type=str, default='SS', help='training opt')
parser.add_argument('--root_save_path', type=str, default='./log', help='the root path to save models and logs')
parser.add_argument('--DDP', action='store_true', default=False, help='Single or Multi GPUs')
parser.add_argument('--resume', action='store_true', default=False, help='resume training processs')

opt = parser.parse_args()



config_name='SOD_yaml.yaml' if opt.task=='SOD' else 'SS_yaml.yaml'

config_path = os.path.join(script_dir, config_name)



# 继承配置
with open(config_path, 'r') as config_file:
    yaml_config=yaml.safe_load(config_file)

for key, value in yaml_config.items():
    if not hasattr(opt, key):# 防止opt被覆盖
        setattr(opt, key, value)


opt.save_path=opt.root_save_path+'/' +opt.task+'_log/log_'+opt.dataset_name+'_'+ opt.model_name.split('-')[1]
opt.checkpoints=opt.save_path+'/checkpoint'
if opt.task=='SS':
    opt.niters_per_epoch = opt.num_train_imgs // opt.batch_size  + 1
    opt.log_file = os.path.join(opt.save_path, "log.log")



import os
import importlib.util

def import_model_from_directory(model):

    file_name, model_name= model.split('-')
    model_directory='/root/autodl-tmp/LFSOD/bestlf/models/encoders'
    model_path=os.path.join(model_directory, file_name)

    if not os.path.isfile(model_path):
        print(f"File '{file_name}' not found in directory '{model_directory}'.")
        return None

    # 尝试从文件中导入指定的模块
    module_name = os.path.splitext(os.path.basename(model_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 检查模块是否包含指定的名称
    if hasattr(module, model_name):
        # 返回找到的模块对象
        return getattr(module, model_name)
    
    else:
        print(f"Model '{model_name}' not found in file '{file_name}'.")
    return None



if __name__=='__main__':
    pprint.pprint(vars(opt))
    # # 示例用法
    # model = import_model_from_directory(opt.model_name)
    # if model is not None:
    #     # 找到了指定的模块，可以继续使用
    #     print("Successfully imported model:", model)
    # else:
    #     print("Model not found in the specified directory.")


