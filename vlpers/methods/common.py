# Copyright (C) 2024 Istituto Italiano di Tecnologia.  All rights reserved.
#
# This work is licensed under the BSD 3-Clause License.
# See the LICENSE file located at the root directory.

import torch


def encode_text_img_retrieval(model, text, img_tokens, split_idxs):    
    b_size = img_tokens.shape[0]
    
    x = model.token_embedding(text).type(img_tokens.dtype)  # [batch_size, n_ctx, d_model]
    collect_ind = text == model.end_id 
    collect_ind = collect_ind.nonzero(as_tuple=False)[:, 1]
    
    # Since the padding index for new concepts is 0
    #  we save the start of sequence embedding and restore it after the substitution
    sos_embedding = x[:, 0]
    x = x.scatter(1, split_idxs[..., None].repeat([1, 1, x.size(2)]), img_tokens)
    x[:, 0] = sos_embedding
    
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x.type(model.dtype))
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)
    # take features from the eot embedding (eot_token is the highest number in each sequence)    
    x = x[torch.arange(x.size(0)), collect_ind] @ model.text_projection
    return x.type(img_tokens.dtype)


def get_concepts_pos(descriptions, value, max_concepts):
    concept_pos = torch.full([descriptions.size(0), max_concepts], fill_value=0, dtype=torch.int64, device=descriptions.device)

    indexes = torch.arange(descriptions.size(1), device=descriptions.device)[None].repeat(descriptions.size(0), 1)
    indexes = torch.masked_select(indexes, (descriptions == value))

    count = torch.sum(descriptions == value, dim=1)
    range_tensor = torch.arange(max_concepts, device=descriptions.device)[None].repeat(descriptions.size(0), 1)

    mask = range_tensor < count.unsqueeze(1)
    concept_pos[mask] = indexes
    
    return concept_pos