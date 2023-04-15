from model import *

def test()->None:

    torch.manual_seed(0)
    device = torch.device("cuda")

    ENC_INPUT_DIM = sp_e.GetPieceSize()
    DEC_INPUT_DIM = sp.GetPieceSize()
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 2048
    N_LAYERS = 6
    NUM_HEADS = 8
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(ENC_INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, NUM_HEADS, ENC_DROPOUT, device)
    dec = Decoder(DEC_INPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, NUM_HEADS, DEC_DROPOUT, device)

    model = Model(enc, dec)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CustomSchedule(optimizer, d_model = ENC_EMB_DIM)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
    
    checkpoint= torch.load('amp_checkpoint.pt')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    scheduler.load_state_dict(checkpoint['scheduler'])



    sequence = "For the second phase of the trials we just had different sizes, small, medium, large and extra-large. It's true."
    sequence2 = "Pour la seconde phase des essais nous avions simplement différentes tailles, petit, moyen, grand et extra-large C'est vrai."
    print("원문 : ",sequence)
    print("번역 원문: ",sequence2)
    translate_test(model, sequence, sequence2)
    translate(model, sequence)


    sequence = "Clarke helped to write the screenplay."
    sequence2 = "Clarke collabora à l’écriture du scénario."
    print("원문 : ",sequence)
    print("번역 원문: ",sequence2)
    translate_test(model, sequence, sequence2)
    translate(model, sequence)


    sequence = "Better to help in other ways and keep Ukraine as a proud, independent, non-EU nation."
    sequence2 = "Mieux vaut donc aider l'Ukraine par d'autres biais et qu'elle reste une nation fière, indépendante et hors de l'Union."
    print("원문 : ",sequence)
    print("번역 원문: ",sequence2)
    translate_test(model, sequence, sequence2)
    translate(model, sequence)

    sequence = "For translation tasks, the Transformer can be trained significantly faster than architectures based on convolutional layers."
    print("원문 : ",sequence)
    translate(model, sequence)

    print(max_encoder)
    print(max_decoder)

################################translate test########################################
def translate(model, sequence = ""):
    encoder_index=sp_e.encode(sequence, out_type=int)

    encoder_index = torch.tensor(encoder_index).unsqueeze(0).cuda()

    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode("", out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).cuda()

    output = decoder_input_index
    for i in range(max_decoder):
        output = model(encoder_index, output)
        output = torch.argmax(output, dim=2)

        if output[0][-1] == 3:
            break
        output = torch.cat([decoder_input_index, output], dim = 1)

    target = output.tolist()[0]

    print("teacher_ratio = 0, 번역문 : ",sp.DecodeIds(target))


def translate_test(model, sequence = "", sequence2 = ""):

    encoder_index=sp_e.encode(sequence, out_type=int)

    encoder_index = torch.tensor(encoder_index).unsqueeze(0).cuda()

    sp.SetEncodeExtraOptions('bos')
    decoder_input_index = sp.encode(sequence2, out_type=int)
    decoder_input_index = torch.tensor(decoder_input_index).unsqueeze(0).cuda()

    output = model(encoder_index, decoder_input_index)
    output = torch.argmax(output, dim=2)
    target = output.tolist()[0]

    print("teacher_ratio = 1, 번역문 : ",sp.DecodeIds(target))
#######################################################################################
