# bloomz.cpp

## Usage

Here are the step for the BloomZ-7B1 model:

```bash
# build this repo
git clone https://github.com/NouamaneTazi/bloomz.cpp
cd bloomz.cpp
make

python3 -m pip install torch numpy transformers

# convert the 7B1 model to ggml FP16 format
python3 convert-hf-to-ggml.py bigscience/bloomz-7b1 models 

# quantize the model to 4-bits
./quantize ./models/ggml-model-bloomz-7b1-f16.bin ./models/ggml-model-bloomz-7b1-f16-q4_0.bin 2

# run the inference
./main -m ./models/ggml-model-bloomz-7b1-f16-q4_0.bin -t 8 -n 128
```