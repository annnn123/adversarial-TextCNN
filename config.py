

dataset_max_len = {'MR': 50,
                   'R8': 114,
                   'R52': 120,
                   '20NG': 324,
                   'Ohsumed': 216}



class config:


    #model config
    model_name = 'TextCNN'

    embedding_type = 'rand'  # rand, static, non_static, multichannel
    w2v_path = './pretrained_vec/glove.6B.300d.txt' # pretrained word vectors path
    pretrained_embeddings = None
    init_method = None
    embedding_size = 300
    padding_idx = 0

    # vocab_size = 5004
    # max_seq_len = 50
    # num_classes = 2

    kernel_sizes = [2, 3, 4]
    num_filters = 100
    in_channels = 1
    dropout_rate = 0.5


    #dataset config
    dataset = 'MR'
    train_data_path = f"./dataset/{dataset}/train.json"
    dev_data_path = None
    test_data_path = f"./dataset/{dataset}/test.json"
    vocab_path = f"./dataset/{dataset}/vocab.json"
    label_path = f"./dataset/{dataset}/labels.json"
    dev_ratio = 0.1

    max_seq_len = dataset_max_len[dataset]


    #train config
    seed = 2022

    batch_size = 128
    max_epoches = 200
    lr = 0.001

    log_every = 100
    valid_epoch = 1
    patience = 10
    model_save_dir = './model_results/'


    #adversarial training config
    adv_method = 'FGSM' #'', 'PGD', 'Free', 'FGSM'
    eps = 0.05
    alpha = 1.25 * eps
    PGD_steps = 5
    Free_num_replays = 5


