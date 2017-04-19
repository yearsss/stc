
import torch


# pro_list (decode_max_len, decode_token)
def beam_search(pro_list, beam_size, dec_dict):
    size = len(pro_list)
    sorted_pro_list = torch.sort(pro_list, 1, descending=True)
    sent = []
    sent_set = set()
    dfs(sorted_pro_list, 0, size, sent_set, sent, beam_size, dec_dict)



def dfs(sorted_pro_list, p, size, sent_set, sent, beam_size, dec_dict):
    if p == size:
        sent_set.add("".join(sent))
    else:
        for idx in sorted_pro_list[p][:beam_size]:
            sent.appenddec_dict[idx]
            dfs(sorted_pro_list, p+1, size, sent_set, sent, beam_size, dec_dict)
            sent.pop()
