# ECE-147-Project

UCLA ECE-147-Project

| model name     | input                   | epoch | train acc | valid acc | num params  |
|----------------|-------------------------|-------|-----------|-----------|-------------|
| LSTM           | raw                     | 30    | 0.9       | 0.31      | 40K         |
| RNN            | raw                     | 30    | 0.87      | 0.27      | 14K         |
| ConvLSTM       | raw                     | 30    | 0.86      | 0.36      | 58K         |
| DeepConvNet    | raw                     | 87    | 0.96      | 0.68      | 27K         |
| ShallowConvNet | raw                     | 30    | 0.83      | 0.57      | 46K         |
| ConvLSTM       | max, avg, subsampling   | 30    | 0.9       | 0.66      | 27K         |
| ATCNet         | raw                     | 30    | 0.83      | 0.57      | 123K        |
