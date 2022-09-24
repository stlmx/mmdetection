import torch
import onnx
import warnings
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core.export import build_model_from_cfg, preprocess_example_input

config_file = './configs/detr/detr_r50_8x2_150e_coco.py'

checkpoint_file = './checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'

device = 'cuda:0'

model = build_model_from_cfg(config_file, checkpoint_file)

input_config = {
        'input_shape': (1,3,800,800),
        'input_path': './demo/demo_800_800.jpg',
        'normalize_cfg': {
            'mean': (123.675, 116.28, 103.53),
            'std': (58.395, 57.12, 57.375)
            }    
    }
    # prepare input
one_img, one_meta = preprocess_example_input(input_config)
img_list, img_meta_list = [one_img], [[one_meta]]
output_file='yolov3.onnx'
model.forward = model.forward_dummy
torch.onnx.export(
            model,
            one_img,
            output_file,
            input_names=['input'],
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11)

import onnxsim
from mmdet import digit_version

dynamic_export = None

# get the custom op path
ort_custom_op_path = ''
try:
    from mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
except (ImportError, ModuleNotFoundError):
    warnings.warn('If input model has custom op from mmcv, \
            you may have to build mmcv with ONNXRuntime from source.')

min_required_version = '0.3.0'
assert digit_version(onnxsim.__version__) >= digit_version(
            min_required_version
        ), f'Requires to install onnx-simplify>={min_required_version}'

input_dic = {'input': img_list[0].detach().cpu().numpy()}
model_opt, check_ok = onnxsim.simplify(
            output_file,
            input_data=input_dic,
            
            dynamic_input_shape=dynamic_export)
if check_ok:
    onnx.save(model_opt, output_file)
    print(f'Successfully simplified ONNX model: {output_file}')
else:
    warnings.warn('Failed to simplify ONNX model.')
