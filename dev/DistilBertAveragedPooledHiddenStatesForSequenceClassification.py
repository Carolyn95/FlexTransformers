@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertAveragedPooledHiddenStatesForSequenceClassification(
    DistilBertPreTrainedModel):

  def __init__(self, config, pooled_layers=None):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.distilbert = DistilBertModel(config)
    self.pre_classifier = nn.Linear(config.dim, config.dim)
    self.classifier = nn.Linear(config.dim, config.num_labels)
    self.dropout = nn.Dropout(config.seq_classif_dropout)
    self.init_weights()
    self.pooled_layers = pooled_layers

  @add_start_docstrings_to_model_forward(
      DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
  @add_code_sample_docstrings(
      tokenizer_class=_TOKENIZER_FOR_DOC,
      checkpoint="distilbert-base-uncased",
      output_type=SequenceClassifierOutput,
      config_class=_CONFIG_FOR_DOC,
  )
  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      head_mask=None,
      inputs_embeds=None,
      labels=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    distilbert_output = self.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if self.pooled_layers is None:
      hidden_state = distilbert_output[0]
      pooled_output = hidden_state[:, 0]
    elif len(self.pooled_layers) == 1:
      hidden_state = distilbert_output.hidden_states[self.pooled_layers[0]]
      pooled_output = torch.mean(hidden_state, 1)
    else:
      pooled_output = torch.cat(
          [distilbert_output.hidden_states[pl] for pl in self.pooled_layers], 1)
      pooled_output = torch.mean(pooled_output, 1)

    pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
    pooled_output = self.dropout(pooled_output)  # (bs, dim)
    logits = self.classifier(pooled_output)  # (bs, num_labels)

    loss = None
    if labels is not None:
      if self.num_labels == 1:
        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
      else:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

    if not return_dict:
      output = (logits,) + distilbert_output[1:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=distilbert_output.hidden_states,
        attentions=distilbert_output.attentions,
    )
