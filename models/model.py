from models.attn import *
from models.decoder import *
from models.encoder import *
from models.embed import EncEmbedding, DecEmbedding


class ISDFormer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_len = args.pred_len

        self.enc_embedding = EncEmbedding(self.args)
        self.dec_embedding = DecEmbedding(self.args)

        # Encoder
        self.encoder = Encoder(
            attn_layers=
            [
                EncoderLayer(

                    AttentionLayer(WhoTalksAttention(True, attention_dropout=args.dropout), args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for _ in range(args.e_layers)
            ],
            conv_layers=
            [
                ConvLayer(
                    args.d_model
                ) for _ in range(args.e_layers - 1)
            ] if args.distil else None,
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullyAttention(attention_dropout=args.dropout),
                                   args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation,
                )
                for _ in range(args.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )

        self.projection = nn.Linear(args.d_model, 1, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # √
        enc_out, _ = self.encoder(enc_out)  # √

        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # √
        dec_out = self.decoder(dec_out, enc_out)  #
        dec_out = self.projection(dec_out)  #

        dec_out = dec_out.view(self.args.batch_size, self.args.pred_len, 1)

        return dec_out
