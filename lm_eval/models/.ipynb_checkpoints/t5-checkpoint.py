import transformers
import torch
from lm_eval.base import BaseLM


class HFLM(BaseLM):

    def __init__(self, device='cuda', pretrained='t5', revision='main', subfolder=None, tokenizer=None, batch_size=1):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.t5 = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained, revision=revision + ("/" + subfolder if subfolder is not None else "")
        ).to(self.device)
        self.t5.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer, revision=revision, subfolder=subfolder)

        assert isinstance(self.tokenizer, (
            transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
            transformers.T5Tokenizer, transformers.T5TokenizerFast,
        )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if isinstance(self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
            assert self.tokenizer.encode('hello\n\nhello') == [31373, 198, 198, 31373], \
                self.tokenizer.encode('hello\n\nhello')

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def inputs_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.inputs_id
    
    @property
    def attention_mask(self):
        return self.tokenizer.attention_mask

    @property
    def max_length(self):
        try:
            return self.t5.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.t5.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        pass
    
    def tok_encode(self, string: str):
        pass
    
    def tok_decode(self, tokens):
        pass

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.t5(inps)[0][:, :, :50257]
    
    def _model_generate(self, context, max_length, inputs_id):
        return self.t5.generate(
            input_ids=inputs_id,
            attention_mask=attention_mask,
            do_sample=False
        )


# for backwards compatibility
T5LM = HFLM
