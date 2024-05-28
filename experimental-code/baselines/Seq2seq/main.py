import torch.nn as nn
import torch.optim as optim
import random
import datetime
from LanguageUtil import *
from constant import *
from utils import *
from model import *


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, print_every=10):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    min_loss = 1000000

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    for epoch in range(1, EPOCHS + 1):
        print('epoch:', epoch)
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('epoch: %d, %d%% ,loss: %.4f' % (epoch, epoch / EPOCHS * 100, print_loss_avg))

        plot_losses.append(loss)

        # if loss < min_loss:
        #     min_loss = loss
        #     torch.save(encoder.state_dict(), './models/encoder.pth')
        #     torch.save(decoder.state_dict(), './models/decoder.pth')

    plot_data(x = range(EPOCHS), y = plot_losses, xlabel = "Epochs", ylabel = "Loss", label = "Loss Graph for target seq2seq model")


def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        # print('input_tensor:', input_tensor.shape)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, output_folder=None):
    # encoder.load_state_dict(torch.load('./models/encoder.pth'))
    # decoder.load_state_dict(torch.load('./models/decoder.pth'))
    encoder.eval()
    decoder.eval()
    lf_accuracy = 0
    ground_truth = []
    predicted = []
    n = len(pairs)
    for i in range(n):
        pair = pairs[i]
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words[:-1])
        ground_truth.append(pair[1])
        predicted.append(output_sentence)
        if output_sentence == pair[1]:
            lf_accuracy += 1
    print('lf_accuracy:', lf_accuracy / n)
    print('accuracy_num:', lf_accuracy)


    # for data in dataloader:
    #     input_tensor, target_tensor = data
    #     print('input_tensor:', input_tensor.shape)
    #     with torch.no_grad():
    #         encoder_outputs, encoder_hidden = encoder(input_tensor)
    #         decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
    #         _, topi = decoder_outputs.topk(1)
    #         decoded_ids = topi.squeeze()

    #         decoded_words = []
    #         for idx in decoded_ids:
    #             if idx.item() == EOS_token:
    #                 decoded_words.append('<EOS>')
    #                 break
    #             decoded_words.append(output_lang.index2word[idx.item()])

    #         ground_truth.append(' '.join([output_lang.index2word[idx.item()] for idx in target_tensor[0] if idx.item() not in [0, 1]]))
    #         predicted.append(' '.join(decoded_words))


    with open(f'{output_folder}/results_{datetime.datetime.now().strftime("%d%H%M%S")}.txt', 'w') as f:
        for i in range(len(ground_truth)):
            if len(pairs[i]) == 2:
                f.write('table_id: ' + pairs[i][2] + '\n')
            f.write('ground_truth: ' + ground_truth[i] + '\n')
            f.write('predicted: ' + predicted[i] + '\n\n')
        f.write('lf_accuracy: ' + str(lf_accuracy / n) + '\n')

    # if (lf_accuracy > 0):
        torch.save(encoder.state_dict(), f'./models/encoder_{datetime.datetime.now().strftime("%d%H%M%S")}.pth')
        torch.save(decoder.state_dict(), f'./models/decoder_{datetime.datetime.now().strftime("%d%H%M%S")}.pth')
        # exit()


if __name__ == '__main__':
    # for _ in range(2000):
        input_lang, output_lang, train_dataloader, test_pairs = get_dataloader(BATCH_SIZE, DATASETS)

        encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

        train(train_dataloader, encoder, decoder, print_every=5)

        # input_lang_test, output_lang_test, pairs = prepareData('eng', 'query', 'test')
        # input_lang_test, output_lang_test, test_dataloader = get_dataloader(BATCH_SIZE, 'test')
        evaluateRandomly(encoder, decoder, input_lang, output_lang, test_pairs, OUTPUT_FOLDER)
        # evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, TEST_SIZE)