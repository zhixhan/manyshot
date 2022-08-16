from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class GPT2LMHeadModelMS(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # context examples
        self.context_kv = None
        self.context_mask = None
      
    
    def forward(
        self,
        input_ids = None,
        past_key_values = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print(use_cache)
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        ctx_kv = []
        for layer in transformer_outputs.past_key_values:
            ctx_kv.append([torch.cat(layer[0].split(1, dim=0), dim=-2).squeeze(0),
                            torch.cat(layer[1].split(1, dim=0), dim=-2).squeeze(0)])
        attention_mask = torch.cat(attention_mask.split(1, dim=0), dim=-1).squeeze(0)
        #ctx_keys = torch.stack([layer[0] for layer in list(transformer_outputs.past_key_values)]).transpose(0,1) #  batch x layer x head x seqlen x dim
        #ctx_values = torch.stack([layer[1] for layer in list(transformer_outputs.past_key_values)]).transpose(0,1)
        # print(attention_mask.shape) # batch x seqlen
        if self.context_kv is None:
            self.context_kv = ctx_kv
            self.context_mask = attention_mask
        else:
            for layer_idx in range(len(ctx_kv)):
                self.context_kv[layer_idx][0] = torch.cat([self.context_kv[layer_idx][0], ctx_kv[layer_idx][0]], dim=-2)
                self.context_kv[layer_idx][1] = torch.cat([self.context_kv[layer_idx][1], ctx_kv[layer_idx][1]], dim=-2)
            #self.context_keys = torch.cat([self.context_keys, ctx_keys], 0)
            #self.context_values = torch.cat([self.context_values, ctx_values], 0)
            self.context_mask = torch.cat([self.context_mask, attention_mask], 0)
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )