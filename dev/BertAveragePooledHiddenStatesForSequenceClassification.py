@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. Take hidden states from BertModel output and apply mean pooling 
    operation on top before attaching classification head""",
    BERT_START_DOCSTRING,
)
class BertAveragedPooledHiddenStatesForSequenceClassification(
    BertPreTrainedModel):

  def __init__(self, config, pooled_layers=None):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    self.init_weights()
    self.pooled_layers = pooled_layers

  @add_start_docstrings_to_callable(
      BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
  @add_code_sample_docstrings(
      tokenizer_class=_TOKENIZER_FOR_DOC,
      checkpoint="bert-base-uncased",
      output_type=SequenceClassifierOutput,
      config_class=_CONFIG_FOR_DOC,
  )
  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      position_ids=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=True,
      return_dict=None,
  ):
    r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    if self.pooled_layers is None:
      pooled_output = outputs.hidden_states[-1]
    elif len(self.pooled_layers) == 1:
      pooled_output = torch.mean(outputs.hidden_states[self.pooled_layers[0]],
                                 1)
    else:
      pooled_output = torch.cat(
          [outputs.hidden_states[pl] for pl in self.pooled_layers], 1)
      pooled_output = torch.mean(pooled_output, 1)

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.num_labels == 1:
        #  We are doing regression
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
      output = (logits,) + outputs[2:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
