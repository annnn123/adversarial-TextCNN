import time

from config import *
from data_utils import *
from utils import *
from model import *
from eval import *
from adv_methods import *


def train(config):


    seed_torch(config.seed)


    vocab = Vocab.load(config.vocab_path)
    label2id = Label.load(config.label_path)
    config.vocab_size = len(vocab)
    config.padding_idx = vocab.padding_idx
    config.num_classes = len(label2id)

    if config.embedding_type != 'rand':
        config.pretrained_embeddings = load_pretrained_embeddings(vocab, config.w2v_path, config.embedding_size)


    if config.dev_data_path is not None:
        train_data = load_data(config.train_data_path)
        dev_data = load_data(config.dev_data_path)
        test_data = load_data(config.test_data_path)
    else:
        train_data = load_data(config.train_data_path)
        train_data, dev_data = train_dev_split(train_data, config.dev_ratio)
        test_data = load_data(config.test_data_path)
    print(f"[DataSize info] train size: {len(train_data)}, dev size: {len(dev_data)}, test size: {len(test_data)}")

    train_dataloader = build_dataloader_for_cls(train_data, vocab, label2id, batch_size=config.batch_size, max_seq_len=config.max_seq_len, shuffle=True)
    dev_dataloader = build_dataloader_for_cls(dev_data, vocab, label2id, batch_size=config.batch_size, max_seq_len=config.max_seq_len, shuffle=False)
    test_dataloader = build_dataloader_for_cls(test_data, vocab, label2id, batch_size=config.batch_size, max_seq_len=config.max_seq_len, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextCNN(config).to(device)


    if config.adv_method=='PGD':
        PGD_model = PGD(model, config.eps, config.alpha)
        best_model_save_path = config.model_save_dir + f"{config.model_name}_{config.adv_method}_eps{config.eps}_PGD_steps{config.PGD_steps}_reg_best_model.bin"
        print(f"{config.model_name} using {config.adv_method} adversarial training with eps={config.eps}, alpha={config.alpha}, PGD_steps={config.PGD_steps}.")
    elif config.adv_method=='Free':
        pass
    elif config.adv_method=='FGSM':
        FGSM_model = FGSM(model, config.eps, config.alpha)
        best_model_save_path = config.model_save_dir + f"{config.model_name}_{config.adv_method}_eps{config.eps}_reg_best_model.bin"
        print(f"{config.model_name} using {config.adv_method} adversarial training with eps={config.eps}, alpha={config.alpha}.")
    else:
        best_model_save_path = config.model_save_dir + f"{config.model_name}_best_model.bin"
        print(f"{config.model_name} without adversarial training.")



    if not config.adv_method:
        lr = config.lr
    else:
        lr = config.lr / 2.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    early_stopping = EarlyStopping(best_model_save_path, config.patience)

    train_time_ls = []
    print('Begin training ...')
    for epoch in range(1, config.max_epoches + 1):
        print(f"Epoch: {epoch}")
        model.train()

        cum_loss = cum_examples = 0
        report_loss = report_examples = 0
        start_time = time.time()
        report_start_time = time.time()
        for step, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            if config.adv_method == 'Free':
                pass

            else:
                model.zero_grad()  # reset gradients

                output = model(input_ids)
                loss = criterion(output, labels)
                loss.backward() # back propagation

                if config.adv_method == 'PGD':
                    PGD_model.backup_grad()
                    for t in range(config.PGD_steps+1):
                        PGD_model.attack(emb_name='embedding', first=(t==0))
                        if t < config.PGD_steps:
                            model.zero_grad()
                        else:
                            PGD_model.restore_grad()

                        tmp_output = model(input_ids)
                        loss_adv = criterion(tmp_output, labels)
                        loss_adv.backward()
                    PGD_model.restore_emb('embedding')

                elif config.adv_method == 'FGSM':
                    FGSM_model.backup_grad()
                    for t in range(2):
                        FGSM_model.attack(emb_name='embedding', first=(t==0))
                        if t == 0:
                            model.zero_grad()
                        else:
                            FGSM_model.restore_grad()

                        tmp_output = model(input_ids)
                        loss_adv = criterion(tmp_output, labels)
                        loss_adv.backward()
                    FGSM_model.restore_emb('embedding')

                else:
                    pass

                optimizer.step()    # update parameters of net

            report_loss += loss.item()
            cum_loss += loss.item()
            report_examples += 1
            cum_examples += 1

            if (step+1) % config.log_every == 0:
                print(f"epoch: {epoch}, step: {step+1}, avg loss: {report_loss / report_examples:.4f}, cost time: {time.time() - report_start_time:.2f}")
                report_loss = report_examples = 0.
                report_start_time = time.time()

        print(f"epoch: {epoch}, avg loss: {cum_loss / cum_examples:.4f}, cost time: {time.time() - start_time:.2f}")
        train_time_ls.append(time.time() - start_time)

        if epoch % config.valid_epoch == 0:
            print("Begin validation ...")
            dev_loss, dev_acc, dev_precision, dev_recall, dev_f1 = evaluate(model, dev_dataloader, device)
            print(f"[Validation] epoch: {epoch}, loss: {dev_loss:.4f}, accuracy: {dev_acc:.4f}")
            # model_save_path = os.path.join(config.model_save_dir, f"model_epoch_{epoch}.bin")
            # print(f"Save current model at {model_save_path}")
            # model.save(model_save_path)

            if early_stopping(-dev_loss, model):
                print("Early Stopping!")
                break

        print('=' * 60)

    print(f"Best Epoch: {epoch-config.patience}, avg cost time per epoch: {np.mean(train_time_ls):.2f}")


    print("Begin testing ...")
    model = TextCNN.load(best_model_save_path).to(device)
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_dataloader, device)
    print(f"[Test]: loss: {test_loss:.4f}, acc: {test_acc:.4f}, precision: {test_precision:.4f}, recall: {test_recall:.4f}, f1: {test_f1:.4f}")



if __name__ == '__main__':
    train(config)