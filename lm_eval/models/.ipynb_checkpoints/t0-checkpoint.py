import transformers
import torch
from lm_eval.base import BaseLM,T5LM


class HFLM(T5LM):

    def __init__(self, device='cuda', pretrained='t0', revision='main', subfolder=None, tokenizer=None, batch_size=1):
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
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.t5.config.n_positions
        except AttributeError:
            #return self.t5.config.n_positions
            return 512 # https://github.com/huggingface/transformers/issues/8047

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

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)


    def _model_call(self, inps,dec_inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            # print("*****inps******")
            # tmp = self.t5(input_ids=inps,decoder_input_ids=dec_inps)[0]
            # print(tmp.shape)
            return self.t5(input_ids=inps,decoder_input_ids=dec_inps)[0][:, :, :32128] # https://huggingface.co/transformers/v3.1.0/model_doc/t5.html
            # return self.t5(inps)[0]

    def _model_generate(self, max_length, inputs_id, attention_mask):
        return self.t5.generate(
            input_ids=inputs_id,
            attention_mask=attention_mask,
            do_sample=False
        )


# for backwards compatibility
T0LM = HFLM
