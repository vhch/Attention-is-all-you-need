from model import *

#trainning
def train(gpu, args):
    ############################################################
    print("gpu", gpu)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.manual_seed(0)
    ENC_INPUT_DIM = sp_e.GetPieceSize()
    DEC_INPUT_DIM = sp.GetPieceSize()
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 2048
    N_LAYERS = 6
    NUM_HEADS = 8
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(ENC_INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, NUM_HEADS, ENC_DROPOUT, gpu)
    dec = Decoder(DEC_INPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, NUM_HEADS, DEC_DROPOUT, gpu)


    model = Model(enc, dec)
    torch.cuda.set_device(gpu)
    model.cuda()

    loss_fn = nn.CrossEntropyLoss(ignore_index = 0).cuda(gpu)
    optimizer = optim.Adam(model.parameters(), lr=1)
    scheduler = CustomSchedule(optimizer, d_model = ENC_EMB_DIM)

#######################################################################################
    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
    # checkpoint= torch.load('amp_checkpoint.pt', map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # amp.load_state_dict(checkpoint['amp'])
    # scheduler.load_state_dict(checkpoint['scheduler'])
    model = DDP(model)
#######################################################################################
    dataset = CustomDataset(encoder_index, decoder_input_index, decoder_output_index)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)

    if gpu == 0:
        print("start")

    for epoch in range(args.epochs):
        loss_epoch = 0
        for encoder_input, decoder_input, decoder_output in dataloader:

            encoder_input = encoder_input.cuda()
            decoder_input = decoder_input.cuda()
            decoder_output = decoder_output.cuda()

            output = model(encoder_input, decoder_input)
            output=torch.transpose(output, 1, 2)
            loss = loss_fn(output, decoder_output)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            loss_epoch += loss
        scheduler.step()
        
        
        loss_epoch = loss_epoch / len(dataloader)
        if gpu == 0:
        #torch.save(model.module, 'weights_only.pth')
        #torch.save(model.module.state_dict(), 'weights_only.pth')
            checkpoint = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict(),
            'scheduler' : scheduler.state_dict()
            }
            torch.save(checkpoint, 'amp_checkpoint.pt')

            print('Epoch:', '%04d' % (epoch + 1), ' cost =', '{:.6f}'.format(loss_epoch))
