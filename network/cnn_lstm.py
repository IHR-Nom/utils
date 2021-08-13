from torch import nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True, dropout=0.5, num_layers=2)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, cnn_out_size, n_channel, n_class, n_hidden, n_rnn=2, leaky_relu=False):
        super(CRNN, self).__init__()
        # assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 1024]

        cnn = nn.Sequential()

        def conv_relu(i, batch_normalization=False):
            n_in = n_channel if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_normalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(n_out))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        self.cnn = cnn

        self.rnn = BidirectionalLSTM(cnn_out_size, n_hidden, n_class)

    def forward(self, x):
        # conv features
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        conv = conv.view(b, -1, w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)

        return output


