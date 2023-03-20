# ECE-147-Project

To achieve > 70% test acc:
Select EEGNet_Modified model and train for 200 epochs. The last checkpoint should reach a test acc of 72.9%.


| model name     | input                   | epoch | train acc | valid acc | num params  |  test acc  |
|----------------|-------------------------|-------|-----------|-----------|-------------|------------|
| LSTM           | raw                     | 30    | 0.9       | 0.31      | 40K         |   	  |
| RNN            | raw                     | 30    | 0.87      | 0.27      | 14K         |   	  |
| ConvLSTM       | raw                     | 30    | 0.86      | 0.36      | 58K         |   	  |
| DeepConvNet    | raw                     | 87    | 0.96      | 0.68      | 27K         | 0.64   	  |
| ShallowConvNet | raw                     | 30    | 0.83      | 0.57      | 46K         | 0.55   	  |
| ConvLSTM       | max, avg, subsampling   | 30    | 0.9       | 0.66      | 27K         |    	  |
| ATCNet         | raw                     | 30    | 0.83      | 0.67      | 123K        | 0.67   	  |
| ViT            | raw                     | 30    | 0.26      | 0.25      | 628K        |   	  |
| EEGNet_Modified| raw                     | 200   | 0.95      | 0.72      | 17.7K       | 0.73   	  |
| EEGNet_Modified| trim                    | 200   | 0.93      | 0.74      | 17.7K       | 0.74   	  |


