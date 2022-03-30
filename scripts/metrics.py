import torch


def IoU(pred: torch.tensor,truth:torch.tensor,class_label:int) -> float:
    """
    pred:  [batch, height, width]
    truth: [batch, height, width]
    class_label: the 2-classes task positive task label

    return:
        inter_map: the intersection map for each sample. [batch,height, width]
        union_map: the union map for each sample. [batch,height,width]
        result: the IoU score for class `class_label`

    Example:
        IoU(pred,truth,0)
    """
    inter_map=torch.logical_and(pred==class_label,truth==class_label)
    union_map=torch.logical_or(pred==class_label,truth==class_label)
    result=torch.ones(pred.shape[0]) 
    non_zero_idx=torch.where(union_map.flatten(1).sum(-1)!=0) # Avoid nan
    result[non_zero_idx]=inter_map.flatten(1).sum(-1)[non_zero_idx]/union_map.flatten(1).sum(-1)[non_zero_idx]
    return inter_map,union_map,result

def compute_iou(output: torch.Tensor, truths: torch.Tensor) -> float:
    """
    Assume in the truths tensor, class id are stored as 0,1,2,... at each place of the tensor
    """
    output = output.detach()
    truths = truths.detach()
    
    ## EXERCISE #####################################################################
    #
    # Implement the IoU metric that is used by the benchmark to grade your results.
    #     
    # `output` is a tensor of dimensions [Batch, Classes, Height, Width]
    # `truths` is a tensor of dimensions [Batch, Height, Width]
    #
    # Tip: Peform a sanity check that tests your implementation on a user-defined 
    #      tensor for which you know what the output should be.
    #
    ################################################################################# 
    class_num=30 # 30 classes to be detected.
    output_index_map=output.argmax(dim=1).type(torch.uint8)
    results=torch.zeros(output.shape[0])
    for class_label in range(class_num): 
        _,_,value=IoU(output_index_map,truths,class_label)
        results+=value
    results/=class_num # average over class
    iou = results.mean().item() # average over batch, return native python float

    #################################################################################
    
    return iou