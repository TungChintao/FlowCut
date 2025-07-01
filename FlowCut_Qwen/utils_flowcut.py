import torch

def compute_score_llm(relation_attn, semantic_attn, value):
   
    relation_score = relation_attn/relation_attn.sum(dim=-1, keepdim=True)
    semantic_score = semantic_attn/semantic_attn.sum(dim=-1, keepdim=True)
   
    final_score = (relation_attn+semantic_attn) * torch.norm(value, p=1, dim=-1)

    return final_score

def compute_score_vit(relation_attn, value):
   
    final_score = relation_attn * torch.norm(value, p=1, dim=-1)

    return final_score


def adaptive_prune_ratio(attn,rpl,target_num):

    N = attn.shape[-1]
    
    entropy = -torch.sum(attn * torch.log(attn+1e-7), dim=-1)
    entropy_max = torch.log(torch.tensor(N)) 
    
    entropy_ratio = entropy/entropy_max

    r = (N-target_num)/rpl**0.5 * (1-entropy_ratio**2)

    return int(r.round())
