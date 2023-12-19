# TODO: TODO's remain
# - make interrupt still make stats

# test cases:
# - prompt parsing test
#   + (person :2.0)(, :1.0)(masterpiece :1.0)(, :1.0)(best :1.1)(quality :1.1)(, :1.0)(blu:0.9090909090909091)(eyel:0.9090909090909091)(low :0.9090909090909091)(, :1.0)(blue :0.9090909090909091)(_ :0.9090909090909091)(yellow :0.9090909090909091)(and :1.0)(good :1.0)(, :1.0)(ok :1.0)(, :1.0)(one :1.1)(two :1.0)(, :1.0)(thre:1.1)(ef:1.1)(our :1.1)(, :1.0)(ok:1.0)(k :1.0)(, :3.0)
#   + animal, (ok:1.2), [phi], <lora:add_detail:0.5>, (example), ((example)), [example], (example:1.5), (example:0.25), \(example\)
#   + 14 steps, 64x64
# - face test
#   + an artistic sketch of a person's face, a portrait, SD
#   + cropped, out of frame, poor quality, worst quality, HD, abstract, symbol, people
# - typography test
#   + best quality, capital letter T, written, ink, inscribed, signature, artistic
#   + engraving, bold, italic, messy, scattered, blot

# import code
# code.interact(local=locals())

from collections.abc import Iterator
from copy import copy
from hashlib import md5
from pathlib import Path
import html
import itertools
import json
import modules.scripts as scripts
import open_clip.tokenizer
import os
import random
import sys
import time

from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
from modules import images, script_callbacks
from modules import prompt_parser
from modules.processing import StableDiffusionProcessing, Processed
from modules.processing import process_images, Processed
from modules.shared import opts, cmd_opts, state
import modules.shared as shared

from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm

from deap import cma 
from deap import base as deap_base 
from deap import creator as deap_creator 
from deap import tools as deap_tools 
import deap

# FLIP: https://github.com/NVlabs/flip
#
# flip_loss.py (at time of writing) is "hot off the press" and
# requires patching for this use case.
#
# TODO: move to install.py?
this_extension_directory = os.path.dirname(os.path.dirname(__file__))
flip_pytorch_path = os.path.join(this_extension_directory, 'flip/pytorch')
flip_loss_patched_path = os.path.join(flip_pytorch_path, 'flip_loss_patched.py')
if os.path.isfile(flip_loss_patched_path):
    print(f"flip_loss_patched.py found at: {flip_loss_patched_path}")
else:
    print('patching flip_loss.py for RGB color space..')
    flip_loss_path = os.path.join(flip_pytorch_path, 'flip_loss.py')
    cuda_enabled = torch.cuda.is_available()
    with open(flip_loss_path) as flip_loss:
        with open(flip_loss_patched_path, 'w') as flip_loss_patched:
            for line in flip_loss:
                # srgb2ycxcz needs to be replaced by linrgb2ycxcz,
                # but only in the LDRFLIPLoss.forward method.
                #
                # It only appears in single quotes ("''") in that method.
                line = line.replace("'srgb2ycxcz'", "'linrgb2ycxcz'")

                # if CUDA is not available, don't use it
                if not cuda_enabled:
                    line = line.replace(".cuda()", '')
                    line = line.replace(", device='cuda'", '')

                flip_loss_patched.write(line)

    print('patched flip_loss.py for RGB color space')

sys.path.append(flip_pytorch_path)
from flip_loss_patched import LDRFLIPLoss, color_space_transform, feature_detection, generate_spatial_filter, hunt_adjustment, hyab, redistribute_errors, spatial_filter
sys.path.remove(flip_pytorch_path)

####################################################################################
# END IMPORTS
####################################################################################


####################################################################################
# BEGIN FLIP LDRFlipLoss PATCHED (low-level image difference)
####################################################################################

# A small modification of LDRFlipLoss to optimize for batches of images where
# some images may be the reference or target of this loss function more than
# once. The original loss function performs a variety of independent computations
# on each image before comparing the results of those computations. Here, the
# independent parts are extracted into a function: "preprocess_ldrflip".
#
# [ꟻLIP](https://github.com/NVlabs/flip)
#
# This code is included under the following license:
#
# # BSD 3-Clause License
#
# Copyright (c) 2020-2023, NVIDIA Corporation & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
class SteppedLDRFLIPLoss(nn.Module):
    """ Class for computing LDR FLIP loss in steps """

    # TODO: enable these parameters in the UI
    def __init__(self):
        """
        :param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer,
                                  default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
        :param qc: float describing the q_c exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
        :param qf: float describing the q_f exponent in the LDR-FLIP feature pipeline (see FLIP paper for details)
        :param pc: float describing the p_c exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
        :param pt: float describing the p_t exponent in the LDR-FLIP color pipeline (see FLIP paper for details)
        :param eps: float containing a small value used to improve training stability
        """
        super().__init__()
        self.cuda_enabled = torch.cuda.is_available()
        self.pixels_per_degree = (0.7 * 3840 / 0.7) * np.pi / 180
        self.qc = 0.7
        self.qf = 0.5
        self.pc = 0.4
        self.pt = 0.95
        self.eps = 1e-15

        # --- Color pipeline ---
        # Spatial filtering
        self.s_a, self.radius_a = generate_spatial_filter(self.pixels_per_degree, 'A')
        self.s_rg, self.radius_rg = generate_spatial_filter(self.pixels_per_degree, 'RG')
        self.s_by, self.radius_by = generate_spatial_filter(self.pixels_per_degree, 'BY')
        self.radius = max(self.radius_a, self.radius_rg, self.radius_by)

        # Color metric precompute
        hunt_adjusted_green = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[1.0]], [[0.0]]]).unsqueeze(0), 'linrgb2lab'))
        hunt_adjusted_blue = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[0.0]], [[1.0]]]).unsqueeze(0), 'linrgb2lab'))
        self.cmax = torch.pow(hyab(hunt_adjusted_green, hunt_adjusted_blue, self.eps), self.qc).item()


    # this function extracts the independent computations per-image from compute_ldrflip
    def preprocess_ldrflip(self, image):
        image = pil_image_to_flip_tensor(self.cuda_enabled, image)

        # LDR-FLIP expects non-NaN values in [0,1] as input
        image = torch.clamp(image, 0, 1)

        # Transform to opponent color space
        image_opponent = color_space_transform(image, 'linrgb2ycxcz')
        filtered_image = spatial_filter(image_opponent, self.s_a, self.s_rg, self.s_by, self.radius)

        # Perceptually Uniform Color Space
        preprocessed_image = hunt_adjustment(color_space_transform(filtered_image, 'linrgb2lab'))

        # --- Feature pipeline ---
        # Extract and normalize Yy component
        img_y = (image_opponent[:, 0:1, :, :] + 16) / 116

        # Edge and point detection
        edges_image = feature_detection(img_y, self.pixels_per_degree, 'edge')
        points_image = feature_detection(img_y, self.pixels_per_degree, 'point')

        return (preprocessed_image, edges_image, points_image)


    # compute_ldrflip with independent per-image computations extracted into preprocess_ldrflip
    def compute_ldrflip_stepped(self, test, reference):
        """
        Computes the LDR-FLIP error map between two LDR images,
        assuming the images are observed at a certain number of
        pixels per degree of visual angle

        :param reference: reference tensor (with NxCxHxW layout with values in the YCxCz color space)
        :param test: test tensor (with NxCxHxW layout with values in the YCxCz color space)
        :return: tensor containing the per-pixel FLIP errors (with Nx1xHxW layout and values in the range [0, 1]) between LDR reference and test images
        """

        preprocessed_reference, edges_reference, points_reference = reference # self.preprocess_ldrflip(reference)
        preprocessed_test, edges_test, points_test = test # self.preprocess_ldrflip(test)

        # Color metric
        deltaE_hyab = hyab(preprocessed_reference, preprocessed_test, self.eps)
        power_deltaE_hyab = torch.pow(deltaE_hyab, self.qc)
        deltaE_c = redistribute_errors(power_deltaE_hyab, self.cmax, self.pc, self.pt)

        # Feature metric
        deltaE_f = torch.max(
            torch.abs(torch.norm(edges_reference, dim=1, keepdim=True) - torch.norm(edges_test, dim=1, keepdim=True)),
            torch.abs(torch.norm(points_test, dim=1, keepdim=True) - torch.norm(points_reference, dim=1, keepdim=True))
        )
        deltaE_f = torch.clamp(deltaE_f, min=self.eps)  # clamp to stabilize training
        deltaE_f = torch.pow(((1 / np.sqrt(2)) * deltaE_f), self.qf)

        # --- Final error ---
        return torch.pow(deltaE_c, 1 - deltaE_f)


    def forward(self, test, reference):
        """
        Computes the LDR-FLIP error map between two LDR images,
        assuming the images are observed at a certain number of
        pixels per degree of visual angle

        :param test: test tensor (with NxCxHxW layout with values in the range [0, 1] in the sRGB color space)
        :param reference: reference tensor (with NxCxHxW layout with values in the range [0, 1] in the sRGB color space)
        :return: float containing the mean FLIP error (in the range [0,1]) between the LDR reference and test images in the batch
        """

        deltaE = self.compute_ldrflip_stepped(test, reference)
        return torch.mean(deltaE)

####################################################################################
# END FLIP LDRFlipLoss PATCHED
####################################################################################

####################################################################################
# FLIP LOSS (high level image difference)
####################################################################################

# Loads an image and transforms it into a numpy array
# in the [0, 1] range with 1xCxHxW layout
def pil_image_to_flip_tensor(cuda_enabled: bool, img: Image) -> torch.Tensor:
    img = img.convert('RGB')
    img = np.asarray(img).astype(np.float32)
    img = np.rollaxis(img, 2)
    img = img / 255.0
    if cuda_enabled:
        img = torch.from_numpy(img).unsqueeze(0).cuda()
    else:
        img = torch.from_numpy(img).unsqueeze(0)
    return img

def pil_image_ldrflip_loss(cuda_enabled: bool, ldrflip_loss_fn, img_reference: Image, img_test: Image) -> float:
    ldr_reference = pil_image_to_flip_tensor(cuda_enabled, img_reference)
    ldr_test = pil_image_to_flip_tensor(cuda_enabled, img_test)
    ldrflip_loss: torch.Tensor = ldrflip_loss_fn(ldr_test, ldr_reference)
    return ldrflip_loss.item()

# Calculate the mean + abs(std) of
#   the pil_image_ldrflip_loss of
#   all of the pairs in the list that
#   have the expected width and height
#
# NOTE: std is used because the sqrt amplifies the variance for values < 1.
def pil_images_custom_ldrflip_loss(image_list: Iterator[Image], expected_width: int, expected_height: int) -> tuple[float, list[float], float]:
    cuda_enabled = torch.cuda.is_available()

    # TODO: DEBUG
    # ldrflip_loss_fn = LDRFLIPLoss()
    ldrflip_loss_fn_NEW = SteppedLDRFLIPLoss()

    # TODO: DEBUG
    # def loss_fn(img_tuple):
    #     img_reference, img_test = img_tuple
    #     return pil_image_ldrflip_loss(cuda_enabled, ldrflip_loss_fn, img_reference, img_test)

    def loss_fn_NEW(img_tuple):
        nonlocal cuda_enabled, ldrflip_loss_fn_NEW
        img_reference, img_test = img_tuple
        return ldrflip_loss_fn_NEW(img_reference, img_test).item()

    def img_filter(img):
        return img.width == expected_width and img.height == expected_height

    filtered_image_list = list(filter(img_filter, image_list))
    filtered_image_list_len = len(filtered_image_list)
    expected_total = (filtered_image_list_len * (filtered_image_list_len - 1)) / 2
    # TODO: DEBUG
    # combinations_list = list(itertools.combinations(filtered_image_list, 2))
    # actual_total = len(combinations_list)
    # assert expected_total == actual_total, f"pil_images_custom_ldrflip_loss: unexpected total number of combinations: {expected_total} != {actual_total}"

    # TODO: DEBUG
    # loss_list_start_time = time.time()
    # loss_list = list(tqdm.contrib.tmap(loss_fn, itertools.combinations(filtered_image_list, 2), total=expected_total))
    # loss_list_end_time = time.time()
    # loss_list_time = loss_list_end_time - loss_list_start_time
    # on len(combinations) == 120
    # loss_list_time
    # 23.747704029083252
    # print('loss_list_time')
    # print(loss_list_time)

    new_loss_list_start_time = time.time()
    preprocessed_filtered_image_list = tqdm.contrib.tmap(lambda x: ldrflip_loss_fn_NEW.preprocess_ldrflip(x), filtered_image_list)
    new_loss_list = list(tqdm.contrib.tmap(loss_fn_NEW, itertools.combinations(preprocessed_filtered_image_list, 2), total=expected_total))
    flip_execution_seconds = time.time() - new_loss_list_start_time
    # # on len(combinations) == 120
    # # new_loss_list_time
    # # 7.4162468910217285
    # print('new_loss_list_time')
    # print(new_loss_list_time)

    # TODO: DEBUG
    # print(loss_list)
    # print(new_loss_list)
    # assert loss_list == new_loss_list, "loss lists aren't equal"

    # TODO fix metric (multi metric/goal?)
    # return (np.mean(loss_list) + np.abs(np.std(loss_list)), loss_list)
    return (np.mean(new_loss_list) + np.abs(np.std(new_loss_list)), new_loss_list, flip_execution_seconds)


####################################################################################
# CLIP (tokenize)
####################################################################################

# From https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
class VanillaClip:
    def __init__(self, clip):
        self.clip = clip

    def vocab(self):
        return self.clip.tokenizer.get_vocab()

    def byte_decoder(self):
        return self.clip.tokenizer.byte_decoder

# From https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
class OpenClip:
    def __init__(self, clip):
        self.clip = clip
        self.tokenizer = open_clip.tokenizer._tokenizer

    def vocab(self):
        return self.tokenizer.encoder

    def byte_decoder(self):
        return self.tokenizer.byte_decoder


# Adapted from https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
def tokenize(text):
    clip = shared.sd_model.cond_stage_model.wrapped
    if isinstance(clip, FrozenCLIPEmbedder):
        clip = VanillaClip(shared.sd_model.cond_stage_model.wrapped)
    elif isinstance(clip, FrozenOpenCLIPEmbedder):
        clip = OpenClip(shared.sd_model.cond_stage_model.wrapped)
    else:
        raise RuntimeError(f'Unknown CLIP model: {type(clip).__name__}')

    tokens = shared.sd_model.cond_stage_model.tokenize([text])[0]
    vocab = {v: k for k, v in clip.vocab().items()}
    code = []
    ids = []
    current_ids = []
    byte_decoder = clip.byte_decoder()

    def dump(last=False):
        nonlocal code, ids, current_ids
        words = [vocab.get(x, "") for x in current_ids]

        # TODO: handle this? probably not..
        # try:
        word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode("utf-8")
        # except UnicodeDecodeError:
        #     if last:
        #         word = "❌" * len(current_ids)
        #     elif len(current_ids) > 4:
        #         id = current_ids[0]
        #         ids += [id]
        #         local_ids = current_ids[1:]
        #         code += [wordscode([id], "❌")]
        #
        #         current_ids = []
        #         for id in local_ids:
        #             current_ids.append(id)
        #             dump()
        #
        #         return
        #     else:
        #         return

        word = word.replace("</w>", " ")
        if word != '':
            code.append(word)

        ids += current_ids
        current_ids = []

    for token in tokens:
        token = int(token)
        current_ids.append(token)
        dump()

    dump(last=True)

    return code


# convert a list of [single_token_word, attention] pairs to a prompt string
def token_attention_to_text(token_attention: list[tuple[str, float]]) -> str:
    result = ''
    for token_word, attention_weight in token_attention:
        result += f"({token_word}:{attention_weight})"

    return result


# Convert a prompt string to a list of [single_token_word, attention] pairs
# and also return the token_attention_to_text result.
def to_token_attention_unchecked(text: str) -> list[tuple[str, float]]:
    parsed_attention = prompt_parser.parse_prompt_attention(text)
    token_attention = []
    for text_chunk, attention_weight in parsed_attention:
        if '|' in text_chunk:
            err = f"sd-prompt-pinning: '|' character unsupported for prompt pinning"
            raise RuntimeError(err)

        if ':' in text_chunk:
            err = f"sd-prompt-pinning: ':' character unsupported unless used like: (word:1.23), [other words:0.42], etc"
            raise RuntimeError(err)

        for token_word in tokenize(text_chunk):
            token_attention.append([token_word, attention_weight])

    return token_attention


# to_token_attention_unchecked with the assertion:
#   to_token_attention(text) == to_token_attention(token_attention_to_text(to_token_attention(text)[0]))
#
# Example:
# """
# token_attention, token_attention_text = to_token_attention(p.prompt)
# # [['person ', 2.0], [', ', 1.0], ['masterpiece ', 1.0], [', ', 1.0], ['best ', 1.1], ['quality ', 1.1], [', ', 1.0], ['blu', 0.9090909090909091], ['eyel', 0.9090909090909091], ['low ', 0.9090909090909091], [', ', 1.0], ['blue ', 0.9090909090909091], ['_ ', 0.9090909090909091], ['yellow ', 0.9090909090909091], ['and ', 1.0], ['good ', 1.0], [', ', 1.0], ['ok ', 1.0], [', ', 1.0], ['one ', 1.1], ['two ', 1.0], [', ', 1.0], ['thre', 1.1], ['ef', 1.1], ['our ', 1.1], [', ', 1.0], ['ok', 1.0], ['k ', 1.0], [', ', 3.0]]
# print('token_attention')
# print(token_attention)
# print()
#
# print('token_attention_text')
# print(token_attention_text)
# print()
#
# # (person :2.0)(, :1.0)(masterpiece :1.0)(, :1.0)(best :1.1)(quality :1.1)(, :1.0)(blu:0.9090909090909091)(eyel:0.9090909090909091)(low :0.9090909090909091)(, :1.0)(blue :0.9090909090909091)(_ :0.9090909090909091)(yellow :0.9090909090909091)(and :1.0)(good :1.0)(, :1.0)(ok :1.0)(, :1.0)(one :1.1)(two :1.0)(, :1.0)(thre:1.1)(ef:1.1)(our :1.1)(, :1.0)(ok:1.0)(k :1.0)(, :3.0)
# token_attention_words, token_attention_fragments = list(zip(*token_attention)) # unzip
#
# # ('person ', ', ', 'masterpiece ', ', ', 'best ', 'quality ', ', ', 'blu', 'eyel', 'low ', ', ', 'blue ', '_ ', 'yellow ', 'and ', 'good ', ', ', 'ok ', ', ', 'one ', 'two ', ', ', 'thre', 'ef', 'our ', ', ', 'ok', 'k ', ', ')
# print('token_attention_words')
# print(token_attention_words)
# print()
#
# # (2.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 0.9090909090909091, 0.9090909090909091, 0.9090909090909091, 1.0, 0.9090909090909091, 0.9090909090909091, 0.9090909090909091, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 3.0)
# print('token_attention_fragments')
# print(token_attention_fragments)
# print()
# """
def to_token_attention(text: str) -> tuple[list[tuple[str, float]], str]:
    token_attention = to_token_attention_unchecked(text)
    token_attention_text = token_attention_to_text(token_attention)
    token_attention_roundtrip = to_token_attention_unchecked(token_attention_text)
    assertion_msg = "roundtrip from token attention pairs failed"

    # DEBUG
    # if token_attention != token_attention_roundtrip:
    #     print('token_attention')
    #     print(token_attention)
    #     print('token_attention_roundtrip')
    #     print(token_attention_roundtrip)

    assert token_attention == token_attention_roundtrip, assertion_msg
    return token_attention, token_attention_text


####################################################################################
# DEAP-CMA-ES (optimization)
####################################################################################

# TODO: extract into functions/classes


############################################################################################################
# DETERMINISTIC-ENOUGH SAVING
############################################################################################################

# gets larges int(filename) + 1 for filename in path
def get_next_sequence_number(path: str) -> int:
    result = 0
    path_dir = Path(path)
    for file in path_dir.iterdir():
        # if not file.is_file(): continue
        try:
            result = max(int(file.name) + 1, result)
        except ValueError:
            pass
    return result

# - each pin run gets a folder (from highest_last_folder_plus_1)
# - each generation gets a deterministic subfilder (generation_number)
# - within a generation, the names don_t actually matter: see lambdas below
#
# p.outpath_samples + "/prompt_pins": all runs of this script
# p.outpath_samples + "/prompt_pins/00prompt_pin_number": a run of the script
# p.outpath_samples + "/prompt_pins/00prompt_pin_number/00generation_number": a generation of the script
class PromptPinFiles:
    def __init__(self, processing_instance: StableDiffusionProcessing):
        self.prompt_pins_path = os.path.join(processing_instance.outpath_samples, "prompt_pins")
        os.makedirs(self.prompt_pins_path, exist_ok=True)
        self.prompt_pin_number = get_next_sequence_number(self.prompt_pins_path)
        self.prompt_pin_path = os.path.join(self.prompt_pins_path, f"{self.prompt_pin_number:08}")
        os.makedirs(self.prompt_pin_path, exist_ok=True)
        self.image_width = processing_instance.width
        self.image_height = processing_instance.height

    # p.outpath_samples + "/prompt_pins/00prompt_pin_number/00generation_number": a generation of the script
    def get_generation_path(self, generation_number: int) -> str:
        generation_path = os.path.join(self.prompt_pin_path, f"{generation_number:08}")
        os.makedirs(generation_path, exist_ok=True)
        return generation_path

    # attention_weights are np.ndarray of float
    def get_instance_path(self, generation_number: int, attention_weights) -> str:
        attention_weight_hash = md5()
        attention_weight_hash.update(attention_weights.data.tobytes())
        attention_weight_hash = attention_weight_hash.hexdigest()
        instance_path = os.path.join(self.get_generation_path(generation_number), f"{attention_weight_hash}")
        os.makedirs(instance_path, exist_ok=True)
        return instance_path

    # make process_images save to the generation path
    # attention_weights are np.ndarray of float
    def set_processing_outdir_to_instance_path(self, processing_instance: StableDiffusionProcessing, generation_number: int, attention_weights) -> StableDiffusionProcessing:
        # check that samples are saved, or throw an error
        if not opts.samples_save:
            raise RuntimeError('sd-prompt-pinning: "Always save all generation images" option is required for this extension to run')
        if processing_instance.do_not_save_samples:
            raise RuntimeError('sd-prompt-pinning: do_not_save_samples was enabled, but must be disabled for this extension to run')

        # set outpath for the processing_instance to the instance_path
        processing_instance.outpath_samples = self.get_instance_path(generation_number, attention_weights)
        return processing_instance

    # get all images from instance (in generation)
    # - load all images in the instance folder
    # - filter down to the ones of the expected size (a la the FLIP test)
    # - return the batch
    def get_instance_images(self, processing_instance: StableDiffusionProcessing, generation_number: int, attention_weights) -> Iterator[Image]:
        def open_image(img_path):
            return Image.open(img_path, 'r') 

        def img_filter(img):
            return img.width == processing_instance.width and img.height == processing_instance.height

        instance_path = Path(self.get_instance_path(generation_number, attention_weights))
        return filter(img_filter, map(open_image, instance_path.glob('**/*.png')))


    # HTML Summary Display
    # Using https://tabulator.info
    def generate_html(self):
        index_path = os.path.join(self.prompt_pin_path, 'index.html')
        with open(index_path, "w") as html_file:
            html_file.write("<!DOCTYPE html>\n")
            html_file.write("<html>\n")
            html_file.write("<head>\n")
            html_file.write("  <link href=\"https://unpkg.com/tabulator-tables@5.5.2/dist/css/tabulator.min.css\" rel=\"stylesheet\">\n")
            html_file.write("  <script type=\"text/javascript\" src=\"https://unpkg.com/tabulator-tables@5.5.2/dist/js/tabulator.min.js\"></script>\n")
            html_file.write("</head>\n")
            html_file.write("<body>\n")
            html_file.write("<img src='cma_plot.png' title='CMA Stats'>\n")
            html_file.write("<br>\n")
            html_file.write("<div id=\"main-table\"></div>\n")
            html_file.write("<script type=\"text/javascript\">\n")
            html_file.write("  function main() {\n")
            html_file.write("    var table = new Tabulator(\"#main-table\", {\n")
            html_file.write("        layout:\"fitDataFill\",\n")
            html_file.write("        responsiveLayout:\"collapse\",\n")
            html_file.write("        pagination:true,\n")
            html_file.write("        paginationSize:5,\n")
            html_file.write("        paginationSizeSelector:[5, 10, 25, 50, 100, true],\n")
            html_file.write("        columns:[\n")
            html_file.write("        {formatter:\"responsiveCollapse\", width:30, minWidth:30, hozAlign:\"center\", resizable:true, headerSort:true},\n")
            html_file.write("        {title:\"Timestamp\", field:\"job_timestamp\", width:200, sorter:\"alphanum\"},\n")
            html_file.write("        {title:\"Loss\", field:\"calculated_loss\", hozAlign:\"left\", sorter:\"number\", width:150},\n")
            html_file.write("        {title:\"Generation\", field:\"generation_number\", hozAlign:\"left\", sorter:\"number\", width:50},\n")
            html_file.write("        {title:\"Instance\", field:\"instance_number\", hozAlign:\"left\", sorter:\"number\", width:50},\n")
            html_file.write(f"        {{title:\"GIF\", field:\"gif\", width:{self.image_width}, formatter:\"image\", formatterParams:{{\n")
            html_file.write(f"            width:\"{self.image_width}px\",\n")
            html_file.write(f"            height:\"{self.image_height}px\",\n")
            html_file.write(f"            urlPrefix:\"\",\n")
            html_file.write("            urlSuffix:\"/summary.gif\",\n")
            html_file.write("        }},\n")
            html_file.write(f"        {{title:\"Loss Plot\", field:\"loss_plot\", width:{self.image_width}, formatter:\"image\", formatterParams:{{\n")
            html_file.write(f"            width:\"{self.image_width}px\",\n")
            html_file.write(f"            height:\"{self.image_height}px\",\n")
            html_file.write(f"            urlPrefix:\"\",\n")
            html_file.write("            urlSuffix:\"/loss_plot.png\",\n")
            html_file.write("        }},\n")
            html_file.write(f"        {{title:\"Images\", field:\"images\", width:{self.image_width}, formatter:\"html\"}},\n")
            html_file.write("        {title:\"Prompt\", field:\"prompt\", width:200, sorter:\"alphanum\"},\n")
            html_file.write("        ],\n")
            html_file.write("    });\n")
            html_file.write("\n")
            html_file.write("    table.on(\"tableBuilt\", () => {\n")
            html_file.write("\n")

            generation_paths = list(filter(lambda path: path.is_dir(), Path(self.prompt_pin_path).glob('*')))
            generation_paths.sort()

            generation_num = 0
            for generation_path in generation_paths:
                instance_paths = list(filter(lambda path: path.is_dir(), Path(generation_path).glob('*')))
                instance_paths.sort()
                instance_num = 0
                for instance_path in instance_paths:
                    image_paths = list(Path(instance_path).glob('**/*.png'))
                    image_paths.sort()

                    job_timestamp = None
                    calculated_loss = None
                    prompt_text = None
                    neg_prompt_text = None
                    stats_json_path = os.path.join(instance_path, 'batch_stats.json')
                    with open(stats_json_path, 'r') as stats_json:
                        json_dict = json.load(stats_json)
                        job_timestamp = json_dict['job_timestamp']
                        calculated_loss = json_dict['calculated_loss']
                        prompt_text = json_dict['prompt']
                        neg_prompt_text = json_dict['negative_prompt']

                    def image_path_to_html_tag(image_path):
                        relative_image_path = str(Path(image_path).relative_to(Path(self.prompt_pin_path)))
                        return f"<img src='{relative_image_path}' width='{self.image_width}' height='{self.image_height}' title='{generation_num}' alt='{instance_num}'>"

                    relative_instance_path = str(Path(instance_path).relative_to(Path(self.prompt_pin_path)))
                    # TODO fix. overloads browser lol
                    # images_html = "\\n".join(map(image_path_to_html_tag, image_paths))
                    images_html = f"{html.escape(relative_instance_path)}"

                    html_file.write("      table.addRow({\n")
                    html_file.write(f"        \"job_timestamp\": \"{job_timestamp}\",\n")
                    html_file.write(f"        \"calculated_loss\": \"{calculated_loss}\",\n")
                    html_file.write(f"        \"generation_number\": \"{generation_num}\",\n")
                    html_file.write(f"        \"instance_number\": \"{instance_num}\",\n")
                    html_file.write(f"        \"gif\": \"{relative_instance_path}\",\n")
                    html_file.write(f"        \"loss_plot\": \"{relative_instance_path}\",\n")
                    html_file.write(f"        \"images\": \"{images_html}\",\n")
                    html_file.write(f"        \"prompt\": \"{prompt_text}\",\n")
                    html_file.write(f"        \"neg_prompt\": \"{neg_prompt_text}\",\n")
                    html_file.write("      });\n")
                    html_file.write("\n")

                    instance_num += 0

                generation_num += 1

            html_file.write("    });\n")
            html_file.write("  }\n")
            html_file.write("  window.onload = main;\n")
            html_file.write("</script>\n")
            html_file.write("</body>\n")
            html_file.write("</html>\n")


############################################################################################################
# END DETERMINISTIC-ENOUGH SAVING
############################################################################################################



####################################################################################
# AUTOMATIC1111 (script/GUI)
####################################################################################


class PromptPinningScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    # Extension title in menu UI
    def title(self):
        return "Prompt pinning"

    def show(self, is_img2img):
        return True
        # TODO: DEBUG
        # return scripts.AlwaysVisible

    # Setup menu ui detail
    def ui(self, is_img2img):
        # with gr.Accordion('Prompt pinning', open=False):
        with gr.Row():
                top_level_desc = gr.HTML('Prompt Pinning<br>')

                # TODO: placeholder not shown: calculate for user?
                # TODO: auto fill values (first get N from prompt..)

                cma_cli_log = gr.Checkbox(
                    True,
                    label="CMA Logging (CLI)"
                )

                # None when empty, else [..]
                user_target_images = gr.File(
                    label='Target images: leave blank to use the first generation as the target',
                    file_count='multiple',
                    file_types=['image'],
                    type='file',
                    container=True,
                )

                # key = 'cma_seed'
                cma_seed = gr.Textbox(
                    label="CMA Seed",
                    info='CMA seed: leave blank for one calculated from the seed and subseed'
                )

                # key = 'cma_number_of_generations'
                cma_number_of_generations = gr.Slider(
                    label="Number of Generations",
                    info='Defaults to int(16 * floor(log(N))) when 0',
                    minimum=0,
                    maximum=10000,
                    step=1,
                    value=0,
                )

                # key = 'cma_initial_population_centroid_radius'
                cma_initial_population_centroid_radius = gr.Slider(
                    label="Initial Population Centroid Radius",
                    info='Radius of the uniform distribution that the initial population is drawn from',
                    minimum=0.001,
                    maximum=2,
                    step=0.001,
                    value=1.0
                )

                # key = 'cma_initial_population_std'
                cma_initial_population_std = gr.Slider(
                    label="Initial Population STD",
                    info='Standard deviation of the initial population',
                    minimum=0.001,
                    maximum=2,
                    step=0.001,
                    value=1.0
                )

                # key = 'lambda_'
                cma_lambda = gr.Textbox(
                    label="Lambda",
                    placeholder='Default: int(4 + 3 * log(N))',
                    info='Number of children to produce at each generation, N is the individual’s size (integer).'
                )

                # key = 'mu'
                cma_mu = gr.Textbox(
                    label="Mu",
                    placeholder='Default: int(lambda_ / 2)',
                    info='The number of parents to keep from the lambda children (integer).'
                )

                # # key = 'cmatrix'
                # cma_cmatrix_desc = 'The initial covariance matrix of the distribution that will be sampled.'
                # cma_cmatrix_default = 'identity(N)'
                # cma_cmatrix = gr.Slider( label="Cmatrix",)

                # key = 'weights'
                cma_weights = gr.Dropdown(
                    ["superlinear", "linear" or "equal"],
                    label="Decrease speed",
                    value="superlinear"
                )

                # key = 'cs'
                cma_cs = gr.Textbox(
                    label="CS",
                    placeholder='Default: (mueff + 2) / (N + mueff + 3)',
                    value='',
                    info='Cumulation constant for step-size.'
                )

                # key = 'damps'
                cma_damps = gr.Textbox(
                    label="Damps",
                    placeholder='Default: 1 + 2 * max(0, sqrt(( mueff - 1) / (N + 1)) - 1) + cs',
                    value='',
                    info='Damping for step-size.'
                )

                # key = 'ccum'
                cma_ccum = gr.Textbox(
                    label="Covariant matrix cumulation",
                    placeholder='Default: 4 / (N + 4)',
                    value='',
                    info='Cumulation constant for covariance matrix.'
                )

                # key = 'ccov1'
                cma_ccov1 = gr.Textbox(
                    label="Rank-1 learning rate",
                    placeholder='Default: 2 / ((N + 1.3)^2 + mueff)',
                    value='',
                    info='Learning rate for rank-one update.'
                )

                # key = 'ccovmu'
                cma_ccovmu = gr.Textbox(
                    label="Rank-mu learning rate",
                    placeholder='Default: 2 * (mueff - 2 + 1 / mueff) / ((N + 2)^2 + mueff)',
                    value='',
                    info='Learning rate for rank-mu update.'
                )

        return [top_level_desc,
                cma_cli_log,
                user_target_images,
                cma_seed,
                cma_number_of_generations,
                cma_initial_population_centroid_radius,
                cma_initial_population_std,
                cma_lambda,
                cma_mu,
                cma_weights,
                cma_cs,
                cma_damps,
                cma_ccum,
                cma_ccov1,
                cma_ccovmu]


    # Extension main process
    # Type: (StableDiffusionProcessing, List<UI>) -> (Processed)
    # args are [StableDiffusionProcessing, UI1, UI2, ...]
    def run(self, processing_instance, *args, **kwargs):
        if not isinstance(processing_instance.prompt, str):
            raise RuntimeError(f"sd-prompt-pinning: non-string prompts unsupported: {type(processing_instance.prompt)}")

        if not isinstance(processing_instance.negative_prompt, str):
            raise RuntimeError(f"sd-prompt-pinning: non-string prompts unsupported: {type(processing_instance.negative_prompt)}")

        token_attention, token_attention_text = to_token_attention(processing_instance.prompt)
        # print('prompt')
        # print(processing_instance.prompt)
        # # best quality, capital letter T, written, ink, inscribed, signature, artistic
        # print()
        # print('token_attention')
        # print(token_attention)
        # # [['best ', 1.0], ['quality ', 1.0], [', ', 1.0], ['capital ', 1.0], ['letter ', 1.0], ['t ', 1.0], [', ', 1.0], ['written ', 1.0], [', ', 1.0], ['ink ', 1.0], [', ', 1.0], ['inscribed ', 1.0], [', ', 1.0], ['signature ', 1.0], [', ', 1.0], ['artistic ', 1.0]]
        # print()
        # print('token_attention_text')
        # print(token_attention_text)
        # # (best :1.0)(quality :1.0)(, :1.0)(capital :1.0)(letter :1.0)(t :1.0)(, :1.0)(written :1.0)(, :1.0)(ink :1.0)(, :1.0)(inscribed :1.0)(, :1.0)(signature :1.0)(, :1.0)(artistic :1.0)
        # print()

        neg_token_attention, neg_token_attention_text = to_token_attention(processing_instance.negative_prompt)
        # print('negative_prompt')
        # print(processing_instance.negative_prompt)
        # # engraving, bold, italic, messy, scattered, blot
        # print()
        # print('neg_token_attention')
        # print(neg_token_attention)
        # # [['engraving ', 1.0], [', ', 1.0], ['bold ', 1.0], [', ', 1.0], ['itali', 1.0], ['c ', 1.0], [', ', 1.0], ['messy ', 1.0], [', ', 1.0], ['scattered ', 1.0], [', ', 1.0], ['blot ', 1.0]]
        # print()
        # print('neg_token_attention_text')
        # print(neg_token_attention_text)
        # # (engraving :1.0)(, :1.0)(bold :1.0)(, :1.0)(itali:1.0)(c :1.0)(, :1.0)(messy :1.0)(, :1.0)(scattered :1.0)(, :1.0)(blot :1.0)
        # print()

        token_attention_words, token_attention_fragments = list(zip(*token_attention)) # unzip
        neg_token_attention_words, neg_token_attention_fragments = list(zip(*neg_token_attention)) # unzip
        len_token_attention = len(token_attention)
        len_neg_token_attention = len(neg_token_attention)
        N = len_token_attention + len_neg_token_attention
        # print('N')
        # print(N)
        # print()

        # TODO: cleanup arg and missing arg handling
        top_level_desc, cma_cli_log, user_target_images, cma_seed, cma_number_of_generations, *rest_args = args
        cma_initial_population_centroid_radius, cma_initial_population_std, *rest_args = rest_args
        cma_lambda, cma_mu, cma_weights, cma_cs, cma_damps, cma_ccum, *rest_args = rest_args
        cma_ccov1, cma_ccovmu = rest_args

        # convert -1 to a random number or None to cma_seed_value
        def possible_seed_to_seed(possible_seed):
            nonlocal cma_seed_value
            if possible_seed == -1:
                return int(random.randrange(4294967294))
            elif possible_seed is None and cma_seed_value is not None:
                return cma_seed
            else:
                return possible_seed

        def missing_arg(s):
            return s == 0 or s == 0.0 or s == '0' or s == '0.0' or s == ''

        if cma_seed == '0' or cma_seed == '':
            cma_seed_value = np.mod(np.prod(list(map(possible_seed_to_seed, [processing_instance.seed, processing_instance.subseed]))), 2**32)
        else:
            cma_seed_value = np.mod(int(cma_seed), 2**32)

        # set the numpy seed (for CMA)
        np.random.seed(cma_seed_value)

        if missing_arg(cma_number_of_generations):
            # Constant derived from rastrigin example, assuming proportional to lambda_ (4 + 3 * log(N))
            # (calculated as 250/lambda_ = ~17.6)
            number_of_generations = int(16 * np.floor(np.log(N)))
        else:
            number_of_generations = int(cma_number_of_generations)

        if missing_arg(cma_initial_population_centroid_radius):
            # radius of the initial population centroid
            initial_population_centroid_radius = 1.0
        else:
            initial_population_centroid_radius = float(cma_initial_population_centroid_radius)

        if missing_arg(cma_initial_population_std):
            # initial standard deviation of the CMA algorithm: would <= 1.0 be best, e.g. 0.1?
            initial_population_std = 1.0
        else:
            initial_population_std = float(cma_initial_population_std)

        strategy_kwargs = {}

        if missing_arg(cma_lambda):
            children_per_gen = int(4 + 3 * np.log(N))
        else:
            children_per_gen = int(cma_lambda)
            strategy_kwargs['lambda_'] = children_per_gen

        if missing_arg(cma_mu):
            pass
        else:
            strategy_kwargs['mu'] = int(cma_mu)

        strategy_kwargs['weights'] = cma_weights

        if missing_arg(cma_cs):
            pass
        else:
            strategy_kwargs['cs'] = float(cma_cs)

        if missing_arg(cma_damps):
            pass
        else:
            cma_damps = float(cma_damps)
            strategy_kwargs['damps'] = float(cma_damps)

        if missing_arg(cma_ccum):
            pass
        else:
            cma_ccum = float(cma_ccum)
            strategy_kwargs['ccum'] = float(cma_ccum)

        if missing_arg(cma_ccov1):
            pass
        else:
            cma_ccov1 = float(cma_ccov1)
            strategy_kwargs['ccov1'] = float(cma_ccov1)

        if missing_arg(cma_ccovmu):
            pass
        else:
            cma_ccovmu = float(cma_ccovmu)
            strategy_kwargs['ccovmu'] = float(cma_ccovmu)

        # set whole-run total number of iterations
        total_steps = processing_instance.steps * number_of_generations * children_per_gen
        shared.total_tqdm.updateTotal(total_steps)

        generation_number = 0
        prompt_pin_files = PromptPinFiles(processing_instance)
        most_recent_processed = None
        def evaluate_individual_vector(attention_weights) -> float:
            nonlocal generation_number, processing_instance, most_recent_processed, prompt_pin_files, token_attention, neg_token_attention, len_token_attention, len_neg_token_attention, user_target_images

            if shared.state.interrupted:
                print('evaluate_individual_vector: interrupted, returning NaN..')
                return float('NaN')

            pos_weights = attention_weights[:len_token_attention]
            neg_weights = attention_weights[len_token_attention:]
            assert len(neg_weights) == len_neg_token_attention, f"evaluate_individual_vector: unexpected len(neg_weights): {len(neg_weights)} != {len_neg_token_attention}"

            # new_token_attention_fragments = abs(weights + token_attention_fragments)
            new_token_attention = list(map(list, zip(token_attention_words, np.abs(np.add(pos_weights, token_attention_fragments)))))
            new_neg_token_attention = list(map(list, zip(neg_token_attention_words, np.abs(np.add(neg_weights, neg_token_attention_fragments)))))

            new_prompt_text = token_attention_to_text(new_token_attention)
            # TODO: DEBUG
            # print('new_prompt_text')
            # print(new_prompt_text)
            # print()

            new_neg_prompt_text = token_attention_to_text(new_neg_token_attention)
            # TODO: DEBUG
            # print('new_neg_prompt_text')
            # print(new_neg_prompt_text)
            # print()

            # Copy processing_instance
            new_processing_instance = copy(processing_instance)
            new_processing_instance = prompt_pin_files.set_processing_outdir_to_instance_path(processing_instance, generation_number, attention_weights)
            new_processing_instance.prompt = new_prompt_text
            new_processing_instance.negative_prompt = new_neg_prompt_text

            process_images_start_time = time.time()
            try:
                most_recent_processed = process_images(new_processing_instance)
            except Exception as e:
                errors.display(e, "generating image for prompt pinning")
                most_recent_processed = Processed(new_processing_instance, [], new_processing_instance.seed, "")
                process_images_seconds = float('NaN')

            process_images_end_time = time.time()
            process_images_seconds = process_images_end_time - process_images_start_time

            if user_target_images is None:
                target_images = prompt_pin_files.get_instance_images(processing_instance, 0, np.zeros(N, dtype=float))
            else:
                def open_image(temp_img):
                    return Image.open(temp_img.name, 'r') 

                target_images = map(open_image, user_target_images)

            # TODO: debug
            # print('target_images')
            # print(target_images)

            current_images = list(prompt_pin_files.get_instance_images(processing_instance, generation_number, attention_weights))
            # TODO: debug
            # print('current_images')
            # print(current_images)

            calculated_loss, calculated_loss_list, flip_execution_seconds = pil_images_custom_ldrflip_loss(itertools.chain(target_images, current_images), processing_instance.width, processing_instance.height)
            # TODO: debug
            # print('calculated_loss')
            # print(calculated_loss)

            ##############################################################################################################################
            # BEGIN GIF
            ##############################################################################################################################

            current_images_gif_path = os.path.join(prompt_pin_files.get_instance_path(generation_number, attention_weights), 'summary.gif')
            current_images[0].save(
                current_images_gif_path,
                save_all=True,
                append_images=current_images[1:], # drop(1)
                duration=50 * len(current_images),
                loop=0) # forever

            ##############################################################################################################################
            # END GIF
            ##############################################################################################################################

            ##############################################################################################################################
            # BEGIN STATS
            ##############################################################################################################################

            # Processed.js() results (including prompts, etc.)
            batch_stats = json.loads(most_recent_processed.js())

            # UI config
            batch_stats['cma_seed'] = cma_seed
            batch_stats['number_of_generations'] = number_of_generations
            batch_stats['initial_population_centroid_radius'] = initial_population_centroid_radius
            batch_stats['initial_population_std'] = initial_population_std
            batch_stats['strategy_kwargs'] = strategy_kwargs

            # results
            batch_stats['calculated_loss'] = calculated_loss
            batch_stats['calculated_loss_list'] = calculated_loss_list
            batch_stats['pos_weights'] = list(pos_weights)
            batch_stats['neg_weights'] = list(neg_weights)
            batch_stats['new_token_attention'] = list(new_token_attention)
            batch_stats['new_neg_token_attention'] = list(new_neg_token_attention)
            batch_stats['flip_execution_seconds'] = flip_execution_seconds
            batch_stats['process_images_seconds'] = process_images_seconds
            batch_stats_json_path = os.path.join(prompt_pin_files.get_instance_path(generation_number, attention_weights), 'batch_stats.json')

            with open(batch_stats_json_path, 'w') as json_file:
                json.dump(batch_stats, json_file, indent=2)

            ##############################################################################################################################
            # END STATS
            ##############################################################################################################################

            ##############################################################################################################################
            # BEGIN PLOT LOSSES
            ##############################################################################################################################

            x_axis = range(len(calculated_loss_list))
            fig, fig_axes = plt.subplots(1, 2, tight_layout=True)

            fig_axes[0].hist(calculated_loss_list)
            fig_axes[0].grid(True)
            fig_axes[0].set_title("Loss distribution")

            sorted_calculated_loss_list = calculated_loss_list
            sorted_calculated_loss_list.sort()
            fig_axes[1].plot(x_axis, sorted_calculated_loss_list)
            fig_axes[1].grid(True)
            fig_axes[1].set_title("All losses")

            loss_plot_path = os.path.join(prompt_pin_files.get_instance_path(generation_number, attention_weights), 'loss_plot.png')
            fig.savefig(loss_plot_path)
            plt.close('all')
            # TODO: DEBUG
            # print(f"saved loss plot to: {loss_plot_path}")

            ##############################################################################################################################
            # END PLOT LOSSES
            ##############################################################################################################################

            return (calculated_loss,)

        # we want to minimize the derived FLIP metric(s)
        deap_creator.create("FitnessMin", deap_base.Fitness, weights=(-1.0,))

        # individuals are numpy.ndarray's of shape=(number_of_tokens,)
        deap_creator.create("Individual", np.ndarray, fitness=deap_creator.FitnessMin)

        toolbox = deap_base.Toolbox()
        toolbox.register("evaluate", evaluate_individual_vector)

        # number of "winning" individuals in the hall of fame
        size_of_hall_of_fame = 1

        # the centroid (initial population) is set to a random vector in [-r, r]^N
        initial_population_centroid = np.random.uniform(-initial_population_centroid_radius, initial_population_centroid_radius, N)
        strategy = cma.Strategy(centroid=initial_population_centroid, sigma=initial_population_std, **strategy_kwargs)
        toolbox.register("generate", strategy.generate, deap_creator.Individual)
        toolbox.register("update", strategy.update)

        # setup the hall-of-fame (most fit individual records) and stats
        halloffame = deap_tools.HallOfFame(size_of_hall_of_fame, similar=np.allclose)    
        stats = deap_tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # deap_algorithms.eaGenerateUpdate(toolbox, ngen=number_of_generations, stats=stats, halloffame=halloffame)
        logbook = deap_tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # TODO: update naming
        NGEN = number_of_generations

        # Objects that will compile the data
        sigma = np.ndarray((NGEN,1))
        axis_ratio = np.ndarray((NGEN,1))
        diagD = np.ndarray((NGEN,N))
        fbest = np.ndarray((NGEN,1))
        best = np.ndarray((NGEN,N))
        std = np.ndarray((NGEN,N))

        # seed output with target
        initial_zero_array = np.zeros(N, dtype=float)
        if user_target_images is not None:
            print('skipping seeding initial output because user inputs provided')
        else:
            print('seeding initial output..')
            initial_output = evaluate_individual_vector(initial_zero_array)
            print('initial output:')
            print(initial_output)

        for gen in range(NGEN):
            generation_number = gen + 1
            print('(generation_number, number_of_generations)')
            print((generation_number, number_of_generations))
            print()

            # exit early if interrupted
            if shared.state.interrupted:
                if most_recent_processed is not None:
                    return Processed(most_recent_processed, [], most_recent_processed.seed, "")
                else:
                    return Processed(processing_instance, [], processing_instance.seed, "")

            # Generate a new population
            population = toolbox.generate()
            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                # break when nan (i.e. fitness calculation was interrupted)
                if np.isnan(fit):
                    break

                ind.fitness.values = fit

            # Update the strategy with the evaluated individuals
            toolbox.update(population)

            # Update the hall of fame and the statistics with the
            # currently evaluated population
            halloffame.update(population)
            record = stats.compile(population)
            logbook.record(evals=len(population), gen=gen, **record)

            if cma_cli_log:
                print('gen, evals, std, min, avg, max')
                print(logbook.stream)

            # Save more data along the evolution for latter plotting
            # diagD is sorted and sqrooted in the update method
            sigma[gen] = strategy.sigma
            axis_ratio[gen] = max(strategy.diagD)**2/min(strategy.diagD)**2
            diagD[gen, :N] = strategy.diagD**2
            fbest[gen] = halloffame[0].fitness.values
            best[gen, :N] = halloffame[0]
            std[gen, :N] = np.std(population, axis=0)

            ##############################################################################################################################
            # BEGIN PLOT LOSSES
            ##############################################################################################################################

            all_calculated_losses = []
            final_calculated_losses = []
            for stats_json_path in Path(prompt_pin_files.get_generation_path(generation_number)).glob('**/batch_stats.json'):
                with open(stats_json_path, 'r') as stats_json:
                    json_dict = json.load(stats_json)
                    all_calculated_losses.extend(json_dict['calculated_loss_list'])
                    final_calculated_losses.append(json_dict['calculated_loss'])

            fig, fig_axes = plt.subplots(2, 2, tight_layout=True)
            all_calculated_losses.sort()
            final_calculated_losses.sort()

            # histogram of all losses
            # [x.]
            # [..]
            fig_axes[0, 0].hist(all_calculated_losses, bins=20)
            fig_axes[0, 0].grid(True)
            fig_axes[0, 0].set_title("Image loss distribution")

            # histogram of all "final" losses
            # [..]
            # [x.]
            fig_axes[1, 0].hist(final_calculated_losses)
            fig_axes[1, 0].grid(True)
            fig_axes[1, 0].set_title("Individual loss distribution")

            # plot of all losses (sorted)
            # [.x]
            # [..]
            fig_axes[0, 1].plot(range(len(all_calculated_losses)), all_calculated_losses)
            fig_axes[0, 1].grid(True)
            fig_axes[0, 1].set_title("All image losses")

            # plot of all "final" losses (sorted)
            # [..]
            # [.x]
            fig_axes[1, 1].plot(range(len(final_calculated_losses)), final_calculated_losses)
            fig_axes[1, 1].grid(True)
            fig_axes[1, 1].set_title("All inidividual (batch) losses")

            loss_plot_path = os.path.join(prompt_pin_files.get_generation_path(generation_number), 'generation_loss_plot.png')
            fig.savefig(loss_plot_path)
            plt.close('all')
            # TODO: DEBUG
            # print(f"saved generation loss plot to: {loss_plot_path}")

            ##############################################################################################################################
            # END PLOT LOSSES
            ##############################################################################################################################


        ##############################################################################################################################
        # BEGIN FINAL CMA PLOT
        ##############################################################################################################################

        # The x-axis will be the number of evaluations
        x = list(range(0, strategy.lambda_ * NGEN, strategy.lambda_))
        avg, max_, min_ = logbook.select("avg", "max", "min")
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.semilogy(x, avg, "--b")
        plt.semilogy(x, max_, "--b")
        plt.semilogy(x, min_, "-b")
        plt.semilogy(x, fbest, "-c")
        plt.semilogy(x, sigma, "-g")
        plt.semilogy(x, axis_ratio, "-r")
        plt.grid(True)
        plt.title("blue: f-values, green: sigma, red: axis ratio")

        plt.subplot(2, 2, 2)
        plt.plot(x, best)
        plt.grid(True)
        plt.title("Object Variables")

        plt.subplot(2, 2, 3)
        plt.semilogy(x, diagD)
        plt.grid(True)
        plt.title("Scaling (All Main Axes)")

        plt.subplot(2, 2, 4)
        plt.semilogy(x, std)
        plt.grid(True)
        plt.title("Standard Deviations in All Coordinates")

        cma_plot_filepath = os.path.join(prompt_pin_files.prompt_pin_path, 'cma_plot.png')
        # plt.show()
        plt.savefig(cma_plot_filepath)
        plt.close('all')
        print(f"saved plot to: {cma_plot_filepath}")

        ##############################################################################################################################
        # END FINAL CMA PLOT
        ##############################################################################################################################

        print()
        print('Hall of Fame')
        print(halloffame)
        print()
        try:
            print('Hall of Fame: keys')
            print(halloffame.keys)
            print()
        except ValueError as e:
            print("Hall of Fame keys failed with: {e}")

        print('Logbook')
        print(logbook)
        print()

        # Processed.js() results (including prompts, etc.)
        final_stats = json.loads(most_recent_processed.js())

        # hall of fame
        # this needs multiple levels of unpacking to support json.dumps
        # (Individual has FitnessMin as a field and neither are supported)
        final_stats['halloffame.items'] = [item.__dict__['fitness'].__dict__ for item in halloffame.items]
        final_stats['halloffame.keys'] = [key.__dict__ for key in halloffame.keys]

        # logbook
        final_stats['logbook'] = logbook
        final_stats['logbook.header'] = logbook.header
        final_stats['final_generation_number'] = generation_number
        final_stats_json_path = os.path.join(prompt_pin_files.get_generation_path(generation_number), 'final_generation.json')

        with open(final_stats_json_path, 'w') as json_file:
            json.dump(final_stats, json_file, indent=2)

        # generate HTML summary file at prompt_pin_path/index.html
        prompt_pin_files.generate_html()

        return most_recent_processed

