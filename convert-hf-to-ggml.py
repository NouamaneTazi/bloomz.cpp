# Convert Hugging Face fine-tuned models to ggml format
#
# Usage:
#
#   git clone https://github.com/openai/whisper
#   git clone https://github.com/ggerganov/whisper.cpp
#   git clone https://huggingface.co/openai/whisper-medium
#
#   python3 ./whisper.cpp/models/convert-h5-to-ggml.py ./whisper-medium/ ./whisper .
#
# This script is similar to "convert-pt-to-ggml.py"
#
# For more info:
#
#   https://github.com/ggerganov/whisper.cpp/issues/157
#

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np

from transformers import BloomModel

conv_map = {
    'word_embeddings'       : 'tok_embeddings',
    "word_embeddings_layernorm": 'norm',
        'input_layernorm'        : 'attention_norm',
        'self_attention.query_key_value': 'attention.query_key_value',
        'self_attention.dense':          'attention.wo',
        'post_attention_layernorm': 'ffn_norm',
        'mlp.dense_h_to_4h'           : 'feed_forward.w1',
        'mlp.dense_4h_to_h'           : 'feed_forward.w2',
        'ln_f'                        : 'output_norm',
        'lm_head' : 'output',
        }

# ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# if len(sys.argv) < 4:
#     print("Usage: convert-h5-to-ggml.py dir_model path-to-whisper-repo dir-output [use-f32]\n")
#     sys.exit(1)

dir_model   = "/Users/nouamanetazi/projects/bloomz.cpp/models/"
dir_whisper = dir_model
dir_out     = dir_model

# model = BloomModel.from_pretrained(dir_model)

#code.interact(local=locals())

dir_tokenizer = dir_model

fname_out = dir_out + "/ggml-model-f32.bin"

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BloomForCausalLM

# model_name = "Muennighoff/bloom-tiny-random"
model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
# https://huggingface.co/bigscience/bloomz-7b1/blob/main/config.json
# config.bias_dropout_fusion = True
# config.skip_bias_add = True
# config.skip_bias_add_qkv = False
hparams = config.to_dict()
# model = AutoModelForCausalLM.from_config(config) # random weights
model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]
ftype = 0

fout = open(fname_out, "wb")

hparams["multiple_of"] = 1
fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
# fout.write(struct.pack("i", hparams["seq_length"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["multiple_of"]))
fout.write(struct.pack("i", hparams["n_head"]))
fout.write(struct.pack("i", hparams["n_layer"]))
fout.write(struct.pack("i", hparams["hidden_size"] // hparams["n_head"])) # rot (obsolete)
fout.write(struct.pack("i", ftype))

# Is this correct??
dot_token = tokenizer.encode(".")[0]
for i in range(hparams["vocab_size"]):
    # TODO: this is probably wrong - not sure how this tokenizer works
    text = tokenizer.decode([i]).encode('utf-8')
    # remove the first byte (it's always '.')
    # text = text[1:]
    fout.write(struct.pack("i", len(text)))
    fout.write(text)
    
list_vars = model.state_dict()
for name in list_vars.keys():
    src = name
    nn = name
    if name != "lm_head.weight":
        nn = nn.split(".")[1:]
    else:
        nn = nn.split(".")

    if nn[0] == "h":
        nn[0] = "layers"
        mapped = conv_map[".".join(nn[2:-1])]
        name = ".".join(nn[:2] + [mapped] + nn[-1:])
    else:
        mapped = conv_map[".".join(nn[:-1])]
        name = ".".join([mapped] + nn[-1:])

    

    if "query_key_value" in src:
        q, k, v = list_vars[src].split(list_vars[src].shape[0] // 3, dim=0)
        old_name = name
        for data, n in zip([q, k, v], ["wq", "wk", "wv"]):
            name = old_name
            name = name.replace("query_key_value", n)
            print(src, ' -> ', name)
            data = data.squeeze().numpy()
            data = data.astype(np.float16) # TODO: default type is fp32

            n_dims = len(data.shape)
            print(name, n_dims, data.shape)

            # default type is fp16
            ftype_cur = 1
            if ftype == 0 or n_dims == 1:
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype_cur = 0

            # header
            str = name.encode('utf-8')
            fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
            for i in range(n_dims):
                fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
            fout.write(str);

            # data
            data.tofile(fout)
        continue

    print(src, ' -> ', name)
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float16)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    # default type is fp16
    ftype_cur = 1
    if ftype == 0 or n_dims == 1:
        print("  Converting to float32")
        data = data.astype(np.float32)
        ftype_cur = 0

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str);

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
