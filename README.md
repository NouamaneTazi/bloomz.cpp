# bloomz.cpp

Inference of HuggingFace's [BLOOM-like](https://huggingface.co/docs/transformers/model_doc/bloom) models in pure C/C++.

The repo was built on top of the amazing [llama.cpp](https://github.com/ggerganov/llama.cpp) repo to support [BLOOM](https://huggingface.co/docs/transformers/model_doc/bloom) models. It supports all models that can be loaded using `BloomForCausalLM.from_pretrained()`.

## Usage

Here are the step for the BloomZ-7B1 model:

```bash
# build this repo
git clone https://github.com/NouamaneTazi/bloomz.cpp
cd bloomz.cpp
make

python3 -m pip install torch numpy transformers

# download and convert the 7B1 model to ggml FP16 format
python3 convert-hf-to-ggml.py bigscience/bloomz-7b1 models 

# quantize the model to 4-bits
./quantize ./models/ggml-model-bloomz-7b1-f16.bin ./models/ggml-model-bloomz-7b1-f16-q4_0.bin 2

# run the inference
./main -m ./models/ggml-model-bloomz-7b1-f16-q4_0.bin -t 8 -n 128
```
