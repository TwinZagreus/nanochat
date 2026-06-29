"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794
CORE 评估: DCLM论文标准基准，测试模型的综合语言能力。
三种任务类型: multiple_choice(多选题), schema, language_modeling(语言建模)
评估流程: 渲染prompt → 批处理token序列 → forward_model → 找最低loss选项或比较预测token

TODOs: All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """渲染多选题的完整prompt: 每个选项渲染为一个独立prompt。
    Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """渲染schema问题的完整prompt: 每个context选项渲染为一个独立prompt。
    Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    渲染语言建模任务prompt: 返回两个prompt (不含continuation / 含continuation)。
    Render complete prompt for a language modeling task.

    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    注意: 模板中使用trim去除尾部空格, 因为某些数据集的context有尾部空格。
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    # 返回两个prompt: 不含continuation和含continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    # prompt_without的尾部空格会影响prompt_with中前缀token的检测 → 需要strip
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    查找多个token序列的公共前缀或公共后缀的长度。
    Find the length of the common prefix or suffix across token sequences

    Args:
        token_sequences: token序列列表
        direction: 'left' 查公共前缀长度 / 'right' 查公共后缀长度
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    # 找到第一个不同的位置, 该位置索引即为公共长度
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """将不等长token序列堆叠为B×T张量, 右侧padding至最长序列长度。
    Stack up a list of token sequences, pad to longest on the right

    Args:
        tokens: 不等长的token序列列表
        pad_token_id: 填充用的token id
    """
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    """多选题批处理: 所有prompt共享相同context前缀, 仅continuation不同。
    Batch multiple-choice prompts: all share the same context prefix, only continuations differ.

    Returns:
        tokens: token序列列表
        start_indices: 每个选项continuation的起始位置 (所有选项相同)
        end_indices: 每个选项continuation的结束位置
    """
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    # 多选题: context相同(公共前缀), continuation不同
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation / 确定每个选项continuation的起止位置
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    """Schema任务批处理: context各不相同, 但continuation相同 (公共后缀)。
    Batch schema prompts: contexts vary but continuation is the same (common suffix).

    Returns:
        tokens: token序列列表
        start_indices: 每个选项continuation的起始位置 (选项不同, 因为context长度不同)
        end_indices: 每个选项continuation的结束位置
    """
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    # Schema任务: context不同, continuation相同(公共后缀)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context / 确定每个context的continuation起止位置
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    """语言建模任务批处理: 输入两个prompt (不含/含continuation), 返回含continuation的序列。
    Batch LM prompts: two prompts (without/with continuation), returns the with-continuation sequence.

    Returns:
        [tokens_with]: 含continuation的完整token序列 (batch_size=1)
        [start_idx]: continuation起始位置
        [end_idx]: continuation结束位置
    """
    # In LM tasks, we have two prompts: without and with continuation
    # LM任务: 两个prompt (不含continuation和含continuation)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    # 只需含continuation的prompt, batch_size固定为1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    前向传播: 输入(B,T)token ids → 输出(B,T)逐位置交叉熵损失 + 逐位置argmax预测
    The last column of losses is set to nan because we don't have autoregressive targets there.
    最后一列设为nan(没有自回归目标→无法计算损失)
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """评估单个样本: 渲染prompt → 批处理 → 前向传播 → 判断正确/错误。
    Evaluate a single example, return True if correct, False otherwise

    Args:
        idx: 样本在数据集中的索引
        model: GPT模型
        tokenizer: 分词器
        data: 完整数据集列表
        device: 计算设备
        task_meta: 任务元数据 (task_type, num_fewshot, continuation_delimiter)
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item) / 采样fewshot示例 (排除当前样本)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type / 根据任务类型渲染prompt并批处理
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # 部分模型有最大序列长度限制 (如GPT-2), 超出部分需截断
    # In these cases, we have to truncate sequences to max length and adjust the indices
    # 截断时保留最后max_tokens个token, 并相应调整continuation的起止索引
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens / 保留末尾的max_tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down / 索引向下偏移
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged / 保持不变
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up all the sequences into a batch / 将所有序列堆叠为batch
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok / BOS作为padding token
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    # 前向传播: 获取每个位置的自回归损失和argmax预测
    losses, predictions = forward_model(model, input_ids)

    # See if the losses/predictions come out correctly / 判断预测是否正确
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1 / LM任务batch_size固定为1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        # predictions[i] 自回归地预测 input_ids[i+1]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        # MC/schema: 选择continuation平均损失最低的选项
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    评估整个任务: 遍历所有样本并按rank分派, 分布式环境下all_reduce汇总结果。
    Evaluate one task across many examples. Handles dispatch to all processes if run with torchrun.

    Args:
        model: GPT模型
        tokenizer: 分词器
        data: 任务数据集列表
        device: 计算设备
        task_meta: 任务元数据 (task_type, num_fewshot, continuation_delimiter)

    Returns:
        mean_correct: 整个任务的平均正确率 (0~1)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank / 按rank分派样本: rank i 处理 i, i+world_size, i+2*world_size, ...
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    # sync results across all the processes if running distributed / 分布式环境: 同步汇总各rank结果
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean / 计算平均正确率
    mean_correct = correct.mean().item()
    return mean_correct
