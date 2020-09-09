import os, sys, shutil
import pdb
os.environ['CUDA_VISIBLE_DEVICES']='4'
import torch
import torchvision
import numpy as np


def torch_to_onnx():
    dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
    model = torchvision.models.alexnet(pretrained=True).cuda()

    # input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(15) ]
    input_names = [ "actual_input_1" ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


def load_onnx_model():
    import onnx

    # Load the ONNX model
    model = onnx.load("alexnet.onnx")
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)

    return model


def run_model():
    import onnxruntime as ort

    ort_session = ort.InferenceSession('alexnet.onnx')
    outputs = ort_session.run(None, {'actual_input_1': np.random.randn(10, 3, 224, 224).astype(np.float32)})
    print(outputs[0].shape)


def main():
    # onnx_model = load_onnx_model()
    run_model()


if __name__ == '__main__':
    main()