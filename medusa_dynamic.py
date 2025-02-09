import torch
import torch.nn.functional as F


NODE = 64  # number of nodes in the tree
DEPTH = 4  # depth of the tree


def initialize_medusa(input_ids, model, past_key_values):
    """
    Initializes the Medusa structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the Medusa logits, original model outputs, and logits.
    2. Sets the Medusa attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - medusa_logits (torch.Tensor): Logits from the Medusa heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    medusa_logits, outputs, logits = model(
        input_ids, past_key_values=past_key_values, output_orig=True, medusa_forward=True
    )

    return medusa_logits, logits


def update_medusa_mask(model, medusa_attn_mask):
    model.base_model.model.medusa_mask = medusa_attn_mask.unsqueeze(0).unsqueeze(0)


def reset_medusa_mode(
    model,
):
    """
    Resets the Medusa settings and the past key-values to their initial state.

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model.
    2. Resets the Medusa mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - None
    """
    model.base_model.model.medusa_mask = None
    model.base_model.model.medusa_mode = None


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def get_nucleus_one_token(logit, temperature, top_p):
    """
    Performs token sampling based on the nucleus (top-p) sampling method.

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC).
        temperature (float): A temperature parameter to control the randomness in sampling.
                             Higher values increase diversity, lower values make selections more deterministic.
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens


def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    """
    Implements token sampling based on the typical sampling method.

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
    threshold = torch.minimum(
        torch.ones_like(entropy) * posterior_threshold,
        torch.exp(-entropy) * posterior_alpha,
    )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens


class MultiTokenGenerator:
    def __init__(self, topk=32):
        self.topk = topk
        self.interval = [0]
        for i in range(DEPTH):
            self.interval.append(self.interval[-1] + topk ** (i + 1))

    def sortedIdx_to_tree_buffer(self, sortedIdx):
        tree_indices = torch.zeros(NODE, dtype=torch.int64, device=sortedIdx.device)
        position_ids = torch.zeros(NODE, dtype=torch.int64, device=sortedIdx.device)
        retrieve_indices = torch.zeros((NODE, DEPTH + 1), dtype=torch.int64, device=sortedIdx.device)
        retrieve_indices[:, 1:] = -1
        tree_attn_mask = torch.zeros((NODE, NODE), device=sortedIdx.device)
        tree_attn_mask[:, 0] = 1.0

        nodes = {}
        is_leaf = torch.ones(NODE, dtype=torch.bool, device=sortedIdx.device)
        is_leaf[0] = False

        for i, idx in enumerate(sortedIdx):
            idx = idx.item()
            nodes[idx] = i + 1

            if idx < self.interval[1]:
                tree_indices[i + 1] = idx + 1
                position_ids[i + 1] = 1
                retrieve_indices[i + 1, 1] = i + 1
                tree_attn_mask[i + 1, idx + 1] = 1.0

            elif idx < self.interval[2]:
                idx -= self.interval[1]
                tree_indices[i + 1] = idx % self.topk + self.topk + 1
                position_ids[i + 1] = 2
                parent = nodes[idx // self.topk]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 2] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:3]] = 1.0

            elif idx < self.interval[3]:
                idx -= self.interval[2]
                tree_indices[i + 1] = idx % self.topk + 2 * self.topk + 1
                position_ids[i + 1] = 3
                parent = nodes[idx // self.topk + self.topk]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 3] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:4]] = 1.0

            elif idx < self.interval[4]:
                idx -= self.interval[3]
                tree_indices[i + 1] = idx % self.topk + 3 * self.topk + 1
                position_ids[i + 1] = 4
                parent = nodes[idx // self.topk + self.interval[2]]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 4] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:5]] = 1.0

            else:
                idx -= self.interval[4]
                tree_indices[i + 1] = idx % self.topk + 4 * self.topk + 1
                position_ids[i + 1] = 5
                parent = nodes[idx // self.topk + self.interval[3]]
                is_leaf[parent] = False
                retrieve_indices[i + 1] = retrieve_indices[parent]
                retrieve_indices[i + 1, 5] = i + 1
                tree_attn_mask[i + 1, retrieve_indices[i + 1, 1:6]] = 1.0

        retrieve_indices = retrieve_indices[is_leaf]

        return tree_indices, position_ids, retrieve_indices, tree_attn_mask

    def generate_candidates(
        self,
        medusa_logits,
        logits,
        temperature=0,
        posterior_threshold=0.3,
        posterior_alpha=0.09,
        top_p=0.8,
        sampling='typical',
        fast=False,
    ):
        top_probs, top_indices = torch.topk(medusa_logits[:, 0, -1], self.topk, dim=-1)
        top_probs = top_probs.float()
        top_probs = F.softmax(top_probs, dim=-1)

        # Greedy decoding: Select the most probable candidate from the original logits.
        if temperature == 0 or fast:
            candidates_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
        else:
            if sampling == 'typical':
                candidates_logit = get_typical_one_token(
                    logits[:, -1], temperature, posterior_threshold, posterior_alpha
                ).squeeze(0)
            elif sampling == 'nucleus':
                candidates_logit = get_nucleus_one_token(logits[:, -1], temperature, top_p).squeeze(0)
            else:
                raise NotImplementedError

        # level 1
        p1_joint = top_probs[0]
        # level 2
        p2_joint = p1_joint.view(self.topk, 1) * top_probs[1].view(1, self.topk)
        # level 3
        p3_joint = p2_joint.view(self.topk, self.topk, 1) * top_probs[2].view(1, 1, self.topk)
        # level 4
        p4_joint = p3_joint.view(self.topk, self.topk, self.topk, 1) * top_probs[3].view(1, 1, 1, self.topk)

        if DEPTH == 5:
            p5_joint = p4_joint.view(self.topk, self.topk, self.topk, self.topk, 1) * top_probs[4].view(
                1, 1, 1, 1, self.topk
            )
            p_joint = torch.cat([p1_joint, p2_joint.view(-1), p3_joint.view(-1), p4_joint.view(-1), p5_joint.view(-1)])
        else:
            p_joint = torch.cat([p1_joint, p2_joint.view(-1), p3_joint.view(-1), p4_joint.view(-1)])

        _, tree_top_indices = torch.topk(p_joint, NODE - 1)
        tree_top_indices = tree_top_indices.sort()[0]

        # dynamic tree buffer
        tree_indices, position_ids, retrieve_indices, tree_attn_mask = self.sortedIdx_to_tree_buffer(tree_top_indices)

        medusa_buffers = {
            "medusa_attn_mask": tree_attn_mask,
            "tree_indices": tree_indices,
            "medusa_position_ids": position_ids,
            "retrieve_indices": retrieve_indices,
        }

        # Combine the selected candidate from the original logits with the topk medusa logits.
        candidates = torch.cat([candidates_logit, top_indices[:DEPTH].view(-1)], dim=-1)

        # Map the combined candidates to the tree indices to get tree candidates.
        tree_candidates = candidates[tree_indices]

        # Extend the tree candidates by appending a zero.
        tree_candidates_ext = torch.cat(
            [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0
        )
        # Retrieve the cartesian candidates using the retrieve indices.
        cart_candidates = tree_candidates_ext[retrieve_indices]

        return cart_candidates, tree_candidates.unsqueeze(0), medusa_buffers


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    medusa_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.

    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.

    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    position_ids = medusa_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates.
    # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.
    tree_medusa_logits, outputs, tree_logits = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        medusa_forward=True,
    )

    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]
    medusa_logits = tree_medusa_logits[:, 0, retrieve_indices]
    return medusa_logits, logits, outputs


def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :DEPTH] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples * n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask


def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :DEPTH] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples * n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-5), dim=-1)
    threshold = torch.minimum(
        torch.ones_like(entropy) * posterior_threshold,
        torch.exp(-entropy) * posterior_alpha,
    )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask


def evaluate_posterior(
    logits,
    candidates,
    temperature,
    posterior_threshold=0.3,
    posterior_alpha=0.09,
    top_p=0.8,
    sampling='typical',
    fast=True,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (candidates[:, 1:] == torch.argmax(logits[:, :DEPTH], dim=-1)).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length

    if sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:, :DEPTH] / temperature, dim=-1)
            candidates_prob = torch.gather(posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1)
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(
            logits, candidates, temperature, posterior_threshold, posterior_alpha
        )
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()

        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length

    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError


def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    medusa_logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits, medusa_logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - medusa_logits (torch.Tensor): Updated medusa logits.
    - new_token (int): Updated counter for the new tokens added.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat([input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1)
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits and medusa logits for the accepted tokens
    logits = logits[None, best_candidate, accept_length : accept_length + 1]
    medusa_logits = medusa_logits[:, None, best_candidate, accept_length : accept_length + 1]
    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, medusa_logits, new_token
