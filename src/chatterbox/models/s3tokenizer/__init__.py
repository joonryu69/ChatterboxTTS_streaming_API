from .s3tokenizer import (
    S3_SR,
    S3_HOP,
    S3_TOKEN_HOP,
    S3_TOKEN_RATE,
    SPEECH_VOCAB_SIZE,
    S3Tokenizer,
)


SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1

    
def drop_invalid_tokens(x):
    """Drop SoS and EoS"""
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1), "only batch size of one allowed for now"
    
    # Ensure x is 1D for this logic
    if len(x.shape) == 2:
        x = x.squeeze(0)

    if SOS in x:
        # Find first SOS, convert to int, and take next token
        s = (x == SOS).nonzero(as_tuple=True)[0][0].item() + 1
    else:
        s = 0

    if EOS in x:
        # Find first EOS and convert to int
        e = (x == EOS).nonzero(as_tuple=True)[0][0].item()
    else:
        e = None

    x = x[s: e]
    return x
